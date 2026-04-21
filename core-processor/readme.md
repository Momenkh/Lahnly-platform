# core-processor

The audio processing engine behind the Lahnly platform. Takes any audio file and runs it through an end-to-end pipeline that separates instrument layers, extracts notes, analyzes music theory, and produces playable output — tabs, chord sheets, diagrams, and a synthesized preview.

> **Current scope:** The pipeline is built around guitar output (tabs, fretboard diagrams). The underlying stages — separation, pitch extraction, note cleaning, key detection, chord detection — are instrument-agnostic by design, so extending the output layer to other instruments is the planned next step.

---

## What It Does

Given any audio file (MP3, WAV, FLAC, M4A), the pipeline:

1. Separates the mix into instrument layers using a neural source separation model
2. Picks the target layer and extracts every note using a polyphonic ML pitch detector
3. Cleans, filters, and quantizes the notes to a musical grid
4. Detects the song's key, scale, and tempo
5. Maps every note to a playable position on the target instrument
6. Identifies chords from simultaneous notes
7. Generates tabs, chord sheets, a fretboard diagram, and a synthesized audio preview

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

# Choose instrument profile (see section below)
python main.py song.mp3 --guitar-type lead
python main.py song.mp3 --guitar-type rhythm    # default
python main.py song.mp3 --guitar-type acoustic
```

---

## Instrument Profiles

The `--guitar-type` flag tunes the pitch detection confidence thresholds and polyphony limits to match the playing style. Choosing the wrong profile is the most common cause of missing or noisy notes.

| Profile | Best For | Confidence Floor | Min Note | Max Polyphony |
|---------|----------|-----------------|----------|---------------|
| `lead` | Electric solos, fast runs, bends, hammer-ons | Low (~0.12) | 40 ms | 3 |
| `acoustic` | Fingerpicking, strummed acoustic | Medium (~0.18) | 50 ms | 5 |
| `rhythm` | Full chords, electric rhythm guitar | Higher (~0.20) | 60 ms | 6 |

**Why this matters:** Lead guitar notes — especially bends and fast passages — receive lower frame-level confidence scores from the ML model because the pitch smears across frames. The `lead` profile lowers the threshold to let those notes through and tightens the polyphony cap so harmonics don't crowd out the real solo notes.

> **Planned:** This flag will be replaced by a general `--instrument` option as support for other instruments (piano, bass, etc.) is added. The confidence and polyphony logic already lives in a settings layer designed to accommodate this.

---

## Pipeline Stages

All outputs are saved to `outputs/<song_name>/`.

| Stage | Name | Output Files |
|-------|------|-------------|
| 1 | Instrument Separation | `01_stem.wav`, `01_stem_meta.json` |
| 2 | Pitch Extraction | `02_raw_notes.json` |
| 3 | Note Cleaning | `03_cleaned_notes.json` |
| 4 | Tempo & Quantization | `04_quantized_notes.json`, `04_tempo.json` |
| 5 | Key / Scale Detection | `05_key_analysis.json` |
| 6 | Instrument Mapping | `06_mapped_notes.json` |
| 7 | Chord Detection | `07_chords.json` |
| 8 | Tab Generation | `08_tabs.txt` |
| 9 | Audio Preview | `09_preview.wav` |
| 10 | Fretboard Diagram | `10_fretboard.png` |
| 11 | Chord Sheet | `11_chord_sheet.png` |

Use `--from-stage N` to skip stages 1 through N-1 and reuse their saved outputs. For example, `--from-stage 3` reuses the stem and raw notes but re-runs everything from cleaning onward.

---

## Pipeline Flow

```
Audio File (MP3/WAV/FLAC/M4A)
        |
        v
  Stage 1: Separation
    Demucs htdemucs_6s (GPU-accelerated)
    Separates full mix into instrument layers
    Falls back: htdemucs_ft -> htdemucs -> raw mix
    Outputs: stem_confidence (0-1) propagated to all later stages
        |
        v
  Stage 2: Pitch Extraction
    Polyphonic ML pitch detector (librosa pyin, basic-pitch planned)
    Onset/frame thresholds adapt to stem_confidence + instrument profile
    Per-note confidence = mean frame-level detection probability
        |
        v
  Stage 3: Note Cleaning
    1. Duration filter   — drops notes shorter than min_duration
    2. Confidence filter — adaptive threshold from stem quality + profile
    3. Bass filter       — stricter gate below E3
    4. Merge nearby      — merges same-pitch notes with tiny gaps
    5. Polyphony limit   — evicts lowest-confidence / highest-pitch extras
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
  Stage 6: Instrument Mapping
    (Guitar) Sliding hand-window model (4-fret span)
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
TABS  --  D# minor  |  123.0 BPM

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

- **Separation:** Demucs `htdemucs_6s` (6-stem model). Falls back to `htdemucs_ft` → `htdemucs` → raw mix if the target stem is silent. Input is loudness-normalised to -16 LUFS before separation.

- **Pitch detection:** librosa pyin (monophonic fallback, currently active). Basic-pitch (polyphonic transformer, CUDA-capable) is the planned upgrade once Python 3.14 wheels are available.

- **Confidence:** Per-note frame-level detection probability — not velocity. Propagated through all downstream stages and used for visualization shading and key-context filtering.

- **Key detection:** Krumhansl-Schmuckler algorithm with dual-weighted pitch class histogram (50% duration + 50% onset count). Returns top-3 candidates.

- **BPM:** librosa beat tracking on the densest 60-second window of the recording, with automatic half/double-time correction.

- **Chord naming:** Template matching against 15 chord types with penalty scoring (`matches - 0.3 × extra pitch classes`) and key-aware tie-breaking.

---

## Future Plans

The core processing stages (separation, pitch extraction, note cleaning, key/chord analysis) already work on any pitched instrument. What changes between instruments is the **mapping layer** (Stage 6) and the **output layer** (Stages 8–11). The planned additions are:

- **Multi-instrument output** — piano roll, sheet music, bass tabs, etc. alongside or instead of guitar tabs
- **Cross-instrument translation** — take a note sequence from one instrument and re-map it to another (e.g. a guitar solo transcribed as piano notation). Drums are out of scope for translation since they are unpitched.
- **YouTube ingestion** — accept a URL instead of a local file; download and preprocess automatically
- **Layer selection** — let the user choose which separated layer to transcribe (e.g. bass instead of guitar)
- **basic-pitch upgrade** — swap the pitch extractor backend from librosa pyin to the polyphonic basic-pitch model once Python 3.14 wheels ship

---

## Requirements

```
torch torchaudio demucs
librosa
numpy soundfile
matplotlib pygame-ce
scipy av
```

GPU (CUDA) is used automatically if available. CPU fallback works but separation is significantly slower.
