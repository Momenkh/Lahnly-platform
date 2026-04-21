"""
AI Music Transcription Platform
End-to-end pipeline: audio file -> guitar tabs + playback + visualization

Usage:
    python main.py <audio_file> [options]

Pipeline stages:
    1  Instrument separation  (Demucs — extracts guitar stem)
    2  Pitch extraction       (basic-pitch polyphonic ML model)
    3  Note cleaning          (filters, merges, polyphony limit)
    4  Quantization           (tempo detection, snap to 16th-note grid)
    5  Key / scale analysis   (Krumhansl-Schmuckler)
    5b Key-context octave correction
    6  Guitar mapping         (string + fret assignment)
    7  Chord detection        (groups simultaneous notes, names chords)
    8  Tab generation         (ASCII guitar tab — solo notes only)
    9  Audio preview          (synthesized WAV + optional playback)
   10  Fretboard diagram      (PNG visualization)
   11  Chord sheet            (PNG chord box diagrams + progression)

Options:
    --guitar-type T   lead | acoustic | rhythm (default: rhythm)
                      lead     = electric solo — lower confidence, polyphony 3, min-note 40ms
                      acoustic = fingerpicked/strummed — middle-ground thresholds
                      rhythm   = full chords — strictest confidence, polyphony 6
    --start MM:SS     Only process notes from this time onward
    --end   MM:SS     Only process notes up to this time
    --bpm-override N  Force BPM to N instead of auto-detecting
    --force-tempo     Re-detect BPM even if a saved tempo exists
    --no-play         Skip audio playback (still saves 09_preview.wav)
    --no-viz          Skip fretboard and chord sheet images
    --no-separate     Skip separation (use raw mix for pitch detection)
    --no-quantize     Skip tempo detection and note quantization
    --from-stage N    Resume from stage N (2-11); requires prior run outputs
"""

import argparse
import os
import sys

from pipeline.config import set_outputs_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Guitar tab transcription pipeline")
    parser.add_argument("audio_file", nargs="?", help="Path to audio file (WAV, FLAC, M4A, MP3)")
    parser.add_argument("--no-play",      action="store_true", help="Skip audio playback")
    parser.add_argument("--no-viz",       action="store_true", help="Skip visualization")
    parser.add_argument("--no-separate",  action="store_true", help="Skip separation (use raw mix)")
    parser.add_argument("--no-quantize",  action="store_true", help="Skip tempo detection and quantization")
    parser.add_argument("--force-tempo",  action="store_true", help="Re-detect BPM even if saved tempo exists")
    parser.add_argument(
        "--guitar-type",
        choices=["lead", "acoustic", "rhythm"],
        default="rhythm",
        help="Guitar type: lead (solo/single-note), acoustic, rhythm (default, full chords)",
    )
    parser.add_argument(
        "--bpm-override",
        type=float,
        default=None,
        metavar="BPM",
        help="Force a specific BPM instead of auto-detecting",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        metavar="MM:SS",
        help="Only process notes starting from this timestamp (e.g. 5:25 or 325)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        metavar="MM:SS",
        help="Only process notes up to this timestamp (e.g. 6:10 or 370)",
    )
    parser.add_argument(
        "--from-stage",
        type=int,
        default=2,
        metavar="N",
        help="Resume from stage N (2-11). Requires prior run to have saved outputs.",
    )
    return parser.parse_args()


def _parse_time(s: str) -> float:
    """Parse MM:SS or plain seconds string to float seconds."""
    s = s.strip()
    if ":" in s:
        parts = s.split(":")
        return int(parts[0]) * 60 + float(parts[1])
    return float(s)


def _load_saved_tempo() -> dict | None:
    """Load 04_tempo.json if it exists, return None otherwise."""
    from pipeline.quantization import load_quantization
    try:
        _, tempo = load_quantization()
        return tempo
    except FileNotFoundError:
        return None


def main():
    args = parse_args()

    if args.audio_file:
        out_dir = set_outputs_dir(args.audio_file)
        print(f"Outputs -> {out_dir}")

    if args.from_stage <= 1 and not args.audio_file:
        print("Error: audio_file is required unless --from-stage > 1")
        sys.exit(1)

    if args.audio_file and not os.path.isfile(args.audio_file):
        print(f"Error: file not found: {args.audio_file}")
        sys.exit(1)

    # ── Stage 1: Instrument Separation ───────────────────────────────────────
    if not args.no_separate and args.from_stage <= 1:
        from pipeline.separation import separate_guitar
        pitch_input = separate_guitar(args.audio_file)
    elif not args.no_separate and args.from_stage <= 2:
        from pipeline.separation import get_stem_path
        stem = get_stem_path()
        if os.path.isfile(stem):
            print("[Stage 1] Skipped — using saved guitar stem")
            pitch_input = stem
        else:
            from pipeline.separation import separate_guitar
            print("[Stage 1] No saved stem found — running separation")
            pitch_input = separate_guitar(args.audio_file)
    else:
        if args.no_separate:
            print("[Stage 1] Skipped — using raw mix for pitch detection")
        pitch_input = args.audio_file

    # ── Stage 2: Pitch Extraction ─────────────────────────────────────────────
    if args.from_stage <= 2:
        from pipeline.pitch_extraction import extract_pitches
        raw_notes = extract_pitches(pitch_input, guitar_type=args.guitar_type)
    elif args.from_stage <= 3:
        from pipeline.pitch_extraction import load_raw_notes
        print("[Stage 2] Skipped — loading saved raw notes")
        raw_notes = load_raw_notes()
    # else: raw_notes not needed — Stage 3 loads its own saved output

    # ── Stage 3: Note Cleaning ────────────────────────────────────────────────
    if args.from_stage <= 3:
        from pipeline.note_cleaning import clean_notes
        cleaned_notes = clean_notes(raw_notes, guitar_type=args.guitar_type)
    else:
        from pipeline.note_cleaning import load_cleaned_notes, load_clean_meta
        print("[Stage 3] Skipped — loading saved cleaned notes")
        cleaned_notes = load_cleaned_notes()

        # Guitar-type mismatch warning
        meta = load_clean_meta()
        saved_type = meta.get("guitar_type")
        if saved_type and saved_type != args.guitar_type:
            print(f"[Warning] Saved cleaned notes used --guitar-type={saved_type} "
                  f"but current run specifies '{args.guitar_type}'.")
            print(f"[Warning] Results may be inconsistent — re-run with --from-stage 3 "
                  f"to re-clean with '{args.guitar_type}' settings.")

    # ── Stage 4: Tempo Detection & Quantization ───────────────────────────────
    if not args.no_quantize:
        from pipeline.quantization import quantize_notes

        if args.from_stage <= 4:
            # Decide whether to reuse a saved BPM
            reuse_tempo = None
            if args.from_stage >= 3 and not args.force_tempo and args.bpm_override is None:
                reuse_tempo = _load_saved_tempo()
                if reuse_tempo:
                    print(f"[Stage 4] Saved BPM found ({reuse_tempo['bpm']:.1f}) — "
                          f"reusing (--force-tempo to override)")

            cleaned_notes, tempo_info = quantize_notes(
                cleaned_notes, pitch_input,
                bpm_override=args.bpm_override,
                reuse_tempo=reuse_tempo,
            )
        else:
            from pipeline.quantization import load_quantization
            try:
                print("[Stage 4] Skipped — loading saved quantization")
                cleaned_notes, tempo_info = load_quantization()
            except FileNotFoundError:
                print("[Stage 4] No saved quantization — running now")
                cleaned_notes, tempo_info = quantize_notes(
                    cleaned_notes, pitch_input,
                    bpm_override=args.bpm_override,
                )
    else:
        print("[Stage 4] Skipped — quantization disabled")
        tempo_info = None

    # ── Stage 5: Key / Scale Analysis ────────────────────────────────────────
    if args.from_stage <= 5:
        from pipeline.music_theory import analyze_key
        key_info = analyze_key(cleaned_notes)
    else:
        from pipeline.music_theory import load_key_analysis
        try:
            print("[Stage 5] Skipped — loading saved key analysis")
            key_info = load_key_analysis()
        except FileNotFoundError:
            from pipeline.music_theory import analyze_key
            print("[Stage 5] No saved key analysis — running now")
            key_info = analyze_key(cleaned_notes)

    # ── Stage 5b: Key-context feedback (octave correction + confidence filter) ──
    from pipeline.note_cleaning import apply_key_octave_correction, apply_key_confidence_filter
    cleaned_notes = apply_key_octave_correction(cleaned_notes, key_info)
    cleaned_notes = apply_key_confidence_filter(cleaned_notes, key_info, conf_cutoff=0.35)

    # ── Stage 6: Guitar Mapping ───────────────────────────────────────────────
    if args.from_stage <= 6:
        from pipeline.guitar_mapping import map_to_guitar
        mapped_notes = map_to_guitar(cleaned_notes, key_info=key_info, guitar_type=args.guitar_type)
    else:
        from pipeline.guitar_mapping import load_mapped_notes
        print("[Stage 6] Skipped — loading saved mapped notes")
        mapped_notes = load_mapped_notes()

    # ── Time range filter (applied to mapped notes so all downstream is scoped) ─
    t_start_s = _parse_time(args.start) if args.start else None
    t_end_s   = _parse_time(args.end)   if args.end   else None

    if t_start_s is not None or t_end_s is not None:
        before = len(mapped_notes)
        lo = t_start_s or 0.0
        hi = t_end_s or float("inf")
        mapped_notes = [n for n in mapped_notes if lo <= n["start"] <= hi]
        label = f"{args.start or '0:00'} - {args.end or 'end'}"
        print(f"[Filter] Time range {label}: {len(mapped_notes)} notes "
              f"(removed {before - len(mapped_notes)})")

    # ── Melody isolation ──────────────────────────────────────────────────────
    # For lead guitar, separate the top voice (melody) from harmony notes.
    # Tabs are generated from melody only; harmony feeds chord detection.
    from pipeline.guitar_mapping import isolate_melody
    from pipeline.settings import MELODY_MIN_PITCH
    melody_notes, harmony_notes = isolate_melody(
        mapped_notes,
        min_pitch=MELODY_MIN_PITCH.get(args.guitar_type, 0),
    )

    # ── Stage 7: Chord Detection ──────────────────────────────────────────────
    # Feed harmony notes (and any chord-like groups from full mapped set) into
    # chord detection. Using harmony_notes means fast melodic runs don't
    # accidentally register as chords.
    if args.from_stage <= 7:
        from pipeline.chord_detection import detect_chords
        _, chord_groups = detect_chords(
            mapped_notes, tempo_info=tempo_info, key_info=key_info
        )
    else:
        from pipeline.chord_detection import load_chord_detection
        try:
            print("[Stage 7] Skipped — loading saved chord detection")
            _, chord_groups = load_chord_detection()
        except FileNotFoundError:
            from pipeline.chord_detection import detect_chords
            print("[Stage 7] No saved chord data — running now")
            _, chord_groups = detect_chords(
                mapped_notes, tempo_info=tempo_info, key_info=key_info
            )

    # ── Stage 8: Tab Generation ───────────────────────────────────────────────
    # Tabs use melody_notes only — clean single-voice representation
    if args.from_stage <= 8:
        from pipeline.tab_generation import generate_tabs
        tab_str = generate_tabs(melody_notes, chord_groups=chord_groups, tempo_info=tempo_info)
    else:
        from pipeline.tab_generation import load_tabs
        print("[Stage 8] Skipped — loading saved tabs")
        tab_str = load_tabs()

    print("\n" + "=" * 60)
    bpm_label = f"  |  {tempo_info['bpm']:.1f} BPM" if tempo_info else ""
    range_label = ""
    if t_start_s is not None or t_end_s is not None:
        range_label = f"  |  {args.start or '0:00'}-{args.end or 'end'}"
    print(f"GUITAR TABS  --  {key_info['key_str']}{bpm_label}{range_label}")
    print("=" * 60)
    print(tab_str[:2000])
    if len(tab_str) > 2000:
        print("  ... (truncated — see 08_tabs.txt for full tab)")
    print("=" * 60 + "\n")

    # ── Stage 9: Audio Export + Playback ─────────────────────────────────────
    if args.from_stage <= 9:
        from pipeline.audio_playback import save_audio, play_notes
        save_audio(melody_notes)
        if not args.no_play:
            play_notes(melody_notes)
    else:
        print("[Stage 9] Skipped")

    # ── Stage 10: Fretboard Visualization ────────────────────────────────────
    if not args.no_viz and args.from_stage <= 10:
        from pipeline.visualization import plot_fretboard
        img_path = plot_fretboard(
            mapped_notes, key_info=key_info,
            save=True, show=False,
        )
        if img_path:
            print(f"[Stage 10] Fretboard diagram saved to: {img_path}")
    else:
        print("[Stage 10] Skipped — visualization disabled")

    # ── Stage 11: Chord Sheet ─────────────────────────────────────────────────
    if not args.no_viz and args.from_stage <= 11:
        from pipeline.chord_sheet import plot_chord_sheet
        chord_path = plot_chord_sheet(
            chord_groups, tempo_info=tempo_info, save=True, show=False
        )
        if chord_path:
            print(f"[Stage 11] Chord sheet saved to: {chord_path}")
    else:
        print("[Stage 11] Skipped — visualization disabled")

    print("\nDone. All outputs saved to: outputs/")


if __name__ == "__main__":
    main()
