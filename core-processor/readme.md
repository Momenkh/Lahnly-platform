# AI Music Transcription Platform

An end-to-end pipeline that takes a guitar audio recording and produces guitar tabs, chord sheets, fretboard diagrams, and a synthesized audio preview.

---

## What It Does

Given any audio file (MP3, WAV, FLAC, M4A), the system:

1. Isolates the guitar from the mix using a neural source separation model
2. Detects every note played using a polyphonic ML pitch detector
3. Cleans, filters, and quantizes the notes to a musical grid
4. Detects the song's key and scale
5. Maps every note to a real guitar string and fret
6. Identifies chords from simultaneous notes
7. Generates readable ASCII guitar tabs (solo lines only)
8. Synthesizes a preview WAV so you can verify the transcription by ear
9. Renders a fretboard diagram image (piano-roll style, per string)
10. Produces a chord reference sheet with fingering diagrams and chord progression

---

## Usage

```bash
# Full pipeline
python main.py song.mp3

# Skip audio playback (still saves 09_preview.wav)
python main.py song.mp3 --no-play

# Skip image output
python main.py song.mp3 --no-viz

# Resume from stage 3 — reuse saved stem and raw notes
python main.py song.mp3 --from-stage 3

# Skip quantization (useful for free-tempo / rubato recordings)
python main.py song.mp3 --no-quantize

# Choose guitar type (see section below)
python main.py song.mp3 --guitar-type lead
python main.py song.mp3 --guitar-type rhythm    # default
python main.py song.mp3 --guitar-type acoustic
```

---

## Guitar Types

The `--guitar-type` flag tunes confidence thresholds and polyphony limits to match the style of playing. Choosing the wrong type is the most common cause of missing notes.

| Type | Best For | Confidence Floor | Min Note | Max Polyphony |
|------|----------|-----------------|----------|---------------|
| `lead` | Electric solos, fast runs, bends, hammer-ons | Low (~0.12) | 40 ms | 3 |
| `acoustic` | Fingerpicking, strummed acoustic | Medium (~0.18) | 50 ms | 5 |
| `rhythm` | Full chords, electric rhythm guitar | Higher (~0.20) | 60 ms | 6 |

**Why this matters:** Lead guitar notes — especially bends and fast passages — receive lower frame-level confidence scores from the ML model because the pitch smears across frames. Using `--guitar-type lead` lowers the threshold to let those notes through, and tightens the polyphony limit to 3 so harmonics don't crowd out the real solo notes.

---

## Pipeline Stages

All outputs are saved to `outputs/<song_name>/`.

| Stage | Name | Output Files |
|-------|------|-------------|
| 1 | Instrument Separation | `01_guitar_stem.wav`, `01_stem_meta.json` |
| 2 | Pitch Extraction | `02_raw_notes.json` |
| 3 | Note Cleaning | `03_cleaned_notes.json` |
| 4 | Tempo & Quantization | `04_quantized_notes.json`, `04_tempo.json` |
| 5 | Key / Scale Detection | `05_key_analysis.json` |
| 6 | Guitar Mapping | `06_mapped_notes.json` |
| 7 | Chord Detection | `07_chords.json` |
| 8 | Tab Generation | `08_tabs.txt` |
| 9 | Audio Preview | `09_preview.wav` |
| 10 | Fretboard Diagram | `10_fretboard.png` |
| 11 | Chord Sheet | `11_chord_sheet.png` |

Use `--from-stage N` to skip stages 1 through N-1 and reuse their saved outputs. For example, `--from-stage 3` reuses the guitar stem and raw notes but re-runs everything from cleaning onward.

---

## Pipeline Flow

```
Audio File (MP3/WAV/FLAC/M4A)
        |
        v
  Stage 1: Separation
    Demucs htdemucs_6s (GPU-accelerated)
    Extracts guitar stem from full mix
    Falls back: htdemucs_ft -> htdemucs -> raw mix
    Outputs: stem_confidence (0-1) used by all later stages
        |
        v
  Stage 2: Pitch Extraction
    Spotify basic-pitch (polyphonic ML model)
    Onset/frame thresholds adapt to stem_confidence + guitar type
    Per-note confidence = mean frame-level detection probability
        |
        v
  Stage 3: Note Cleaning
    1. Duration filter   — drops notes shorter than min_duration (type-dependent)
    2. Confidence filter — adaptive threshold from stem quality + guitar type
    3. Bass filter       — stricter gate below E3 (relaxed for htdemucs_6s stem)
    4. Merge nearby      — merges same-pitch notes separated by tiny gaps
    5. Polyphony limit   — evicts lowest-confidence / highest-pitch notes
        |
        v
  Stage 4: Tempo & Quantization
    librosa beat tracking on the densest 60s window of note activity
    Half/double-time BPM correction (tests bpm/2, bpm, bpm*2)
    Snaps note start times to nearest 16th-note grid (35% tolerance)
        |
        v
  Stage 5: Key / Scale Detection
    Krumhansl-Schmuckler algorithm across all 24 major/minor keys
    Dual-weighted pitch histogram: 50% duration + 50% onset count
    Returns top-3 key candidates with confidence scores
        |
        v
  Stage 6: Guitar Mapping
    Sliding hand-window model (4-fret span)
    Prefers open strings that are in-key
    Long notes stay on current string; short notes move freely
        |
        +---------------------------+
        |                           |
        v                           v
  Stage 7: Chord Detection     (mapped notes)
    Groups notes within strum      |
    window (40% of quarter note)   |
    Names chords via template      |
    matching with penalty scoring  |
    Key-aware tie-breaking         |
        |                           |
        +----------+                |
        |          |                |
        v          v                v
  Stage 8      Stage 11        Stage 9 & 10
  Tab          Chord Sheet     Audio Preview (09_preview.wav)
  Generation   (diagrams +     Fretboard Diagram (10_fretboard.png)
  (08_tabs.txt) progression)
```

---

## Output Example

```
GUITAR TABS  --  D# minor  |  123.0 BPM

[0:00]
e |----------------------------------------------------------------|
B |--------4-----------------------------4-------------------------|
G |------------4-----------7---------------7-----------------------|
D |----------------------------------------------------------------|
A |--------------------------------------------------------------6-|
E |----------------------------------------------------------------|

[0:03]
e |----------------------------------------------------------------|
B |--------------------6-----------4-------------------------------|
G |----------------------------------------------------------------|
D |----------------------------------------------------------------|
A |----6----------------------------------------6------------------|
E |--------------------------7-----------4-------------------------|
```

---

## Architecture Notes

- **Separation:** Demucs `htdemucs_6s` (6-stem model with a dedicated guitar stem). Falls back to `htdemucs_ft` → `htdemucs` → raw mix if the guitar stem is silent. Input is loudness-normalised to -16 LUFS before separation.

- **Pitch detection:** Spotify basic-pitch (polyphonic transformer, CUDA-capable). Falls back to librosa pyin (monophonic) if unavailable. Minimum note length is 40–60 ms depending on guitar type.

- **Confidence:** Per-note frame-level detection probability from the basic-pitch model output array — not velocity. Propagated through all downstream stages and used for visualization shading.

- **Key detection:** Krumhansl-Schmuckler algorithm with dual-weighted pitch class histogram (50% duration + 50% onset count). Returns top-3 candidates.

- **BPM:** librosa beat tracking on the densest 60-second window of the recording, with automatic half/double-time correction.

- **Chord naming:** Template matching against 15 chord types with penalty scoring (`matches - 0.3 × extra pitch classes`) and key-aware tie-breaking.

---

## Requirements

```
torch torchaudio demucs
basic-pitch
librosa
numpy soundfile
matplotlib pygame
pyloudnorm av
```

GPU (CUDA) is used automatically if available. CPU fallback works but separation is significantly slower.
