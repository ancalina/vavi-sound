#!/usr/bin/env python3
"""
SMAF (MMF) -> MIDI converter with MA-3/MA-5/MA-7 support.

This script reads a Mobile Standard/Application File (SMAF) containing a
score track, extracts the first Mtsq chunk, interprets VM3/VM5-style event
streams (as used by MA-3/MA-5/MA-7 handsets), and emits a single-track
Standard MIDI file (format 0).

The implementation mirrors key time-base behaviors from the vavi-sound
SMAF parser so that duration and gate times are scaled independently. That
helps keep percussion tracks aligned on MA-5/MA-7 material whose gate times
use a different time base from event durations.
"""

from __future__ import annotations

import argparse
import io
import struct
import sys
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_TEMPO_BPM = 120


def read_varlen(buf: bytes, idx: int) -> Tuple[int, int]:
    """Read a variable-length quantity (same style as MIDI)."""
    value = 0
    while True:
        if idx >= len(buf):
            return value, idx
        b = buf[idx]
        idx += 1
        value = (value << 7) | (b & 0x7F)
        if not (b & 0x80):
            break
    return value, idx


def extract_mtsq_and_timebase(mmf_data: bytes) -> Tuple[bytes, int, int]:
    """
    Find MTR and Mtsq chunks in the MMF and extract:
    - Mtsq payload (VM3/VM5 event stream)
    - duration time base (time_d) from the MTR header
    - gate time base (time_g) from the MTR header (falls back to time_d)
    """
    time_d = 1
    time_g = 1

    # --- MTR (score track header) ---
    idx_mtr = mmf_data.find(b"MTR")
    if idx_mtr != -1:
        # Chunk layout for MTR:
        # [0:3]  = 'MTR'
        # [3]    = subtype (track index etc.)
        # [4:8]  = 32-bit big-endian size of payload
        # [8:...] payload: [0]=fmt, [1]=seq_type, [2]=time_d, [3]=time_g, ...
        size_mtr = struct.unpack(">I", mmf_data[idx_mtr + 4 : idx_mtr + 8])[0]
        payload = mmf_data[idx_mtr + 8 : idx_mtr + 8 + size_mtr]
        if len(payload) >= 4:
            fmt = payload[0]
            seq_type = payload[1]
            time_d = payload[2] or 1
            time_g = payload[3] or time_d
            # Some MA-7 files pad the header; ignore any trailing bytes.
            # print(f"MTR: fmt={fmt}, seq_type={seq_type}, time_d={time_d}, time_g={time_g}")

    # --- Mtsq (sequence data) ---
    idx_mtsq = mmf_data.find(b"Mtsq")
    if idx_mtsq == -1:
        raise RuntimeError("No Mtsq chunk found in this MMF file.")

    # Chunk layout for Mtsq:
    # [0:4] = 'Mtsq'
    # [4:8] = 32-bit big-endian size of payload
    # [8:...] payload = VM3/VM5 event stream
    size_mtsq = struct.unpack(">I", mmf_data[idx_mtsq + 4 : idx_mtsq + 8])[0]
    payload = mmf_data[idx_mtsq + 8 : idx_mtsq + 8 + size_mtsq]
    if len(payload) < size_mtsq:
        raise RuntimeError("Truncated Mtsq payload.")

    return payload, time_d, time_g


def parse_vm3_events(buf: bytes) -> List[Tuple[Any, ...]]:
    """
    Parse VM3/VM5-style event stream into a list of abstract events.

    Returns a list of tuples like:
      ("prog", abs_tick, ch, program)
      ("ctrl", abs_tick, ch, controller, value)
      ("aftertouch", abs_tick, ch, note, pressure)
      ("ch_pressure", abs_tick, ch, pressure)
      ("pitch_bend", abs_tick, ch, lsb, msb)
      ("note_on", abs_tick, ch, note, vel, gate)
      ("note_on80", abs_tick, ch, note, vel, -1)
      ("sysex", abs_tick, data_bytes)
      ("ff", abs_tick, n)
    """
    events: List[Tuple[Any, ...]] = []
    abs_tick = 0
    i = 0

    while i < len(buf):
        delta, i = read_varlen(buf, i)
        abs_tick += delta

        if i >= len(buf):
            break

        status = buf[i]
        i += 1
        st_hi = status & 0xF0
        ch = status & 0x0F

        if status == 0xF0:
            # SysEx: read until 0xF7
            start = i
            while i < len(buf) and buf[i] != 0xF7:
                i += 1
            if i < len(buf):
                i += 1  # include F7
            events.append(("sysex", abs_tick, buf[start:i]))

        elif status == 0xFF:
            # In mmfplay this is just a marker with one following byte.
            if i >= len(buf):
                break
            n = buf[i]
            i += 1
            events.append(("ff", abs_tick, n))

        elif st_hi == 0x80:
            # "Note on" without explicit gate time (mmfplay uses -1)
            if i >= len(buf):
                break
            note = buf[i]
            i += 1
            vel, i = read_varlen(buf, i)
            events.append(("note_on80", abs_tick, ch, note, vel, -1))

        elif st_hi == 0x90:
            # Note on with gate time
            if i >= len(buf):
                break
            note = buf[i]
            i += 1
            vel, i = read_varlen(buf, i)
            gate, i = read_varlen(buf, i)
            events.append(("note_on", abs_tick, ch, note, vel, gate))

        elif st_hi == 0xA0:
            # Polyphonic aftertouch (key pressure)
            if i + 1 > len(buf):
                break
            note = buf[i]
            pressure = buf[i + 1]
            i += 2
            events.append(("aftertouch", abs_tick, ch, note, pressure))

        elif st_hi == 0xB0:
            # Controller change
            if i + 1 > len(buf):
                break
            ctr = buf[i]
            val = buf[i + 1]
            i += 2
            events.append(("ctrl", abs_tick, ch, ctr, val))

        elif st_hi == 0xC0:
            # Program change
            if i >= len(buf):
                break
            prg = buf[i]
            i += 1
            events.append(("prog", abs_tick, ch, prg))

        elif st_hi == 0xD0:
            # Channel pressure (aftertouch)
            if i >= len(buf):
                break
            pressure = buf[i]
            i += 1
            events.append(("ch_pressure", abs_tick, ch, pressure))

        elif st_hi == 0xE0:
            # Pitch bend (LSB, MSB)
            if i + 1 > len(buf):
                break
            lsb = buf[i]
            msb = buf[i + 1]
            i += 2
            events.append(("pitch_bend", abs_tick, ch, lsb, msb))

        else:
            # Unknown or end of sequence
            break

    return events


def write_varlen(value: int) -> bytes:
    """Encode an integer as a MIDI-style variable-length quantity."""
    buffer = value & 0x7F
    result = [buffer]
    value >>= 7
    while value > 0:
        buffer = (value & 0x7F) | 0x80
        result.insert(0, buffer)
        value >>= 7
    return bytes(result)


def _midi_events_from_vm3_events(
    events: Iterable[Tuple[Any, ...]],
    time_d: int,
    time_g: int,
) -> List[Tuple[int, int, bytes]]:
    """Translate parsed VM events into MIDI events with scaled ticks."""
    midi_events: List[Tuple[int, int, bytes]] = []
    order_counter = 0

    # Default gate per channel, used when note_on80 doesn't have explicit gate.
    last_gate_per_ch: Dict[int, int] = {ch: 48 for ch in range(16)}

    duration_base = max(1, time_d)
    gate_base = max(1, time_g)

    for ev in events:
        kind = ev[0]

        if kind in {"sysex", "ff"}:
            # Skip sys-ex and vendor markers; they are not mapped.
            continue

        if kind == "prog":
            _, tick, ch, prg = ev
            status = 0xC0 | (ch & 0x0F)
            midi_bytes = bytes([status, prg & 0x7F])
            midi_events.append((tick * duration_base, order_counter, midi_bytes))
            order_counter += 1

        elif kind == "aftertouch":
            _, tick, ch, note, pressure = ev
            status = 0xA0 | (ch & 0x0F)
            midi_bytes = bytes([status, note & 0x7F, pressure & 0x7F])
            midi_events.append((tick * duration_base, order_counter, midi_bytes))
            order_counter += 1

        elif kind == "ctrl":
            _, tick, ch, ctr, val = ev
            status = 0xB0 | (ch & 0x0F)
            midi_bytes = bytes([status, ctr & 0x7F, val & 0x7F])
            midi_events.append((tick * duration_base, order_counter, midi_bytes))
            order_counter += 1

        elif kind in {"note_on", "note_on80"}:
            if kind == "note_on":
                _, tick, ch, note, vel, gate = ev
                gate_ticks = max(1, gate * gate_base)
                last_gate_per_ch[ch] = gate_ticks
            else:  # "note_on80"
                _, tick, ch, note, vel, _ = ev
                gate_ticks = max(1, last_gate_per_ch.get(ch, 48))

            tick = tick * duration_base

            # Note on
            status_on = 0x90 | (ch & 0x0F)
            midi_events.append(
                (tick, order_counter, bytes([status_on, note & 0x7F, max(1, vel) & 0x7F]))
            )
            order_counter += 1

            # Note off (using 0x80, could also use NoteOn with vel=0)
            tick_off = tick + gate_ticks
            status_off = 0x80 | (ch & 0x0F)
            midi_events.append((tick_off, order_counter, bytes([status_off, note & 0x7F, 64])))
            order_counter += 1

        elif kind == "ch_pressure":
            _, tick, ch, pressure = ev
            status = 0xD0 | (ch & 0x0F)
            midi_bytes = bytes([status, pressure & 0x7F])
            midi_events.append((tick * duration_base, order_counter, midi_bytes))
            order_counter += 1

        elif kind == "pitch_bend":
            _, tick, ch, lsb, msb = ev
            status = 0xE0 | (ch & 0x0F)
            midi_bytes = bytes([status, lsb & 0x7F, msb & 0x7F])
            midi_events.append((tick * duration_base, order_counter, midi_bytes))
            order_counter += 1

    midi_events.sort(key=lambda x: (x[0], x[1]))
    return midi_events


def build_midi(events: List[Tuple[Any, ...]], time_d: int, time_g: int, tempo_bpm: int = DEFAULT_TEMPO_BPM) -> bytes:
    """
    Convert parsed VM3/VM5 events to a single-track MIDI (format 0).

    time_d is the duration time base and scales event deltas.
    time_g is the gate time base and scales note lengths.
    """
    midi_events = _midi_events_from_vm3_events(events, time_d, time_g)

    ticks_per_quarter = max(tempo_bpm * max(1, time_d), 24)

    track_stream = io.BytesIO()

    # Tempo meta event at start (tempo in microseconds per quarter note)
    tempo_us = int(round(60_000_000 / max(1, tempo_bpm)))
    track_stream.write(write_varlen(0))
    track_stream.write(bytes([0xFF, 0x51, 0x03, (tempo_us >> 16) & 0xFF, (tempo_us >> 8) & 0xFF, tempo_us & 0xFF]))

    last_tick = 0
    for tick, _, data_bytes in midi_events:
        delta = tick - last_tick
        if delta < 0:
            delta = 0
        track_stream.write(write_varlen(delta))
        track_stream.write(data_bytes)
        last_tick = tick

    # End-of-track meta event
    track_stream.write(write_varlen(0))
    track_stream.write(b"\xFF\x2F\x00")

    track_data = track_stream.getvalue()

    # Build MIDI file with one track
    midi_stream = io.BytesIO()
    midi_stream.write(b"MThd")
    midi_stream.write(struct.pack(">I", 6))  # header length
    midi_stream.write(struct.pack(">H", 0))  # format 0
    midi_stream.write(struct.pack(">H", 1))  # one track
    midi_stream.write(struct.pack(">H", ticks_per_quarter))

    midi_stream.write(b"MTrk")
    midi_stream.write(struct.pack(">I", len(track_data)))
    midi_stream.write(track_data)

    return midi_stream.getvalue()


def mmf_to_midi(in_path: str, out_path: str, tempo_bpm: int = DEFAULT_TEMPO_BPM) -> None:
    """High-level helper: read MMF, convert, and write MIDI."""
    with open(in_path, "rb") as f:
        mmf_data = f.read()

    mtsq, time_d, time_g = extract_mtsq_and_timebase(mmf_data)
    events = parse_vm3_events(mtsq)
    midi_bytes = build_midi(events, time_d=time_d, time_g=time_g, tempo_bpm=tempo_bpm)

    with open(out_path, "wb") as f:
        f.write(midi_bytes)


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert SMAF (MMF) to MIDI.")
    parser.add_argument("input", help="Input .mmf file")
    parser.add_argument("output", help="Output .mid file")
    parser.add_argument(
        "--tempo",
        type=int,
        default=DEFAULT_TEMPO_BPM,
        help="MIDI tempo in BPM (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    args = _parse_args(argv)
    mmf_to_midi(args.input, args.output, tempo_bpm=args.tempo)
    print(f"Converted '{args.input}' -> '{args.output}' (tempo={args.tempo} BPM)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
