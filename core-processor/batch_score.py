"""
Batch accuracy benchmark — auto-discovers all audio files in the current directory,
infers guitar role from the filename, lets the pipeline auto-detect the type from
the separated stem, and appends scores to benchmark_results.json.

Role inference rules (case-insensitive filename match):
  contains "lead"/"solo"  → role=lead   (single run)
  contains "rhythm"       → role=rhythm (single run)
  no match (full song)    → TWO runs: role=rhythm + role=lead

Guitar type is always auto-detected by the pipeline from the stem — the batch
script never passes --type so the detector runs on every song.

Usage:
  python batch_score.py                    # run all discovered songs
  python batch_score.py --role lead        # run only lead-role songs
  python batch_score.py --role rhythm      # run only rhythm-role songs
  python batch_score.py --from-stage 2    # skip separation (reuse existing stems)
  python batch_score.py --no-save         # print results only, don't write to JSON
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
RESULTS_FILE     = "benchmark_results.json"
SCORE_RE         = re.compile(r"\[Score\] Chroma similarity \(stem vs preview\): ([0-9.]+)")
MODE_RE          = re.compile(r"Mode\s*:\s*(\w+)_(\w+)")


def infer_roles(filename: str) -> list[str]:
    """
    Returns the guitar role(s) to run for this file.
    Type is always auto-detected by the pipeline from the stem.
    Labelled files → single role.  Unlabelled → ["rhythm", "lead"] (two runs).
    """
    name = filename.lower()
    if any(k in name for k in ("lead", "solo")):
        return ["lead"]
    if "rhythm" in name:
        return ["rhythm"]
    return ["rhythm", "lead"]


def discover_songs(role_filter: str | None = None) -> list[tuple[str, str]]:
    """Returns list of (filename, guitar_role) pairs to run."""
    songs = []
    seen = set()
    for ext in AUDIO_EXTENSIONS:
        for f in glob.glob(f"*{ext}") + glob.glob(f"*{ext.upper()}"):
            if f in seen:
                continue
            seen.add(f)
            for role in infer_roles(f):
                if role_filter is None or role == role_filter:
                    songs.append((f, role))
    return sorted(songs)


def load_results() -> dict:
    if os.path.isfile(RESULTS_FILE):
        with open(RESULTS_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    return {
        "description": (
            "Chroma similarity benchmark results (stem vs synthesized preview). "
            "Scale: 0.55=poor, 0.70=usable, 0.85=very good."
        ),
        "runs": [],
    }


def save_results(data: dict) -> None:
    with open(RESULTS_FILE, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def run_song(filename: str, guitar_role: str, extra_args: list[str]) -> tuple[float | None, str | None, str | None]:
    """Run the pipeline for one song+role. Type is auto-detected by the pipeline.
    Returns (score, detected_guitar_type, detected_guitar_role) — all None on error.
    """
    cmd = [
        sys.executable, "main.py", filename,
        "--role", guitar_role,
        "--score", "--no-play", "--no-viz",
    ] + extra_args

    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    output = proc.stdout + proc.stderr

    keywords = ["[Auto]", "Mode :", "[Stage", "[Score", "[Warning", "Error", "Traceback"]
    for line in output.splitlines():
        if any(k in line for k in keywords):
            print("  " + line)

    score_m = SCORE_RE.search(output)
    mode_m  = MODE_RE.search(output)

    score         = float(score_m.group(1)) if score_m else None
    detected_type = mode_m.group(1) if mode_m else None
    detected_role = mode_m.group(2) if mode_m else None

    return score, detected_type, detected_role


def print_summary(runs: list[dict]) -> None:
    if not runs:
        return

    print("\n" + "=" * 80)
    print(f"  {'Song':<40} {'Mode':<18} {'Score':>6}  {'Timestamp'}")
    print("=" * 80)
    for r in runs:
        ts   = r.get("timestamp", r.get("date", "—"))
        mode = r.get("mode", f"{r.get('guitar_type','?')}_{r.get('guitar_role','?')}")
        print(f"  {r['song'][:40]:<40} {mode:<18} {r['score']:>6.3f}  {ts}")

    scores = [r["score"] for r in runs]
    print("-" * 80)
    print(f"  Total runs: {len(scores)}   Overall avg: {sum(scores)/len(scores):.3f}")

    by_mode: dict[str, list[float]] = {}
    for r in runs:
        mode = r.get("mode", f"{r.get('guitar_type','?')}_{r.get('guitar_role','?')}")
        by_mode.setdefault(mode, []).append(r["score"])
    for mode, sc in sorted(by_mode.items()):
        print(f"    {mode:<18}: avg {sum(sc)/len(sc):.3f}  ({len(sc)} runs)")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Batch chroma-similarity benchmark")
    parser.add_argument(
        "--role", choices=["lead", "rhythm"], default=None,
        help="Only run songs of this role (lead or rhythm). Default: all roles.",
    )
    parser.add_argument(
        "--from-stage", type=int, default=1, metavar="N",
        help="Resume each song from stage N (e.g. 2 to skip separation)",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Print results only — do not append to benchmark_results.json",
    )
    args = parser.parse_args()

    songs = discover_songs(role_filter=args.role)
    if not songs:
        print("No matching audio files found.")
        return

    extra_args = ["--from-stage", str(args.from_stage)] if args.from_stage > 1 else []

    results_data = load_results()
    new_runs: list[dict] = []

    print("=" * 80)
    label = f"role={args.role}" if args.role else "all roles"
    print(f"  Discovered {len(songs)} run(s) — {label}  (type: auto-detected)")
    print("=" * 80)

    for filename, guitar_role in songs:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        print(f"\n[role={guitar_role}] {filename}", flush=True)
        score, detected_type, detected_role = run_song(filename, guitar_role, extra_args)

        if score is not None:
            guitar_type = detected_type or "unknown"
            actual_role = detected_role or guitar_role
            mode        = f"{guitar_type}_{actual_role}"
            print(f"  -> Score: {score:.3f}  (detected: {mode})")
            entry = {
                "song":        filename,
                "guitar_type": guitar_type,
                "guitar_role": actual_role,
                "mode":        mode,
                "score":       score,
                "timestamp":   ts,
            }
            new_runs.append(entry)
            if not args.no_save:
                results_data["runs"].append(entry)
                save_results(results_data)
        else:
            print("  -> ERROR: no score extracted")

    print_summary(new_runs if new_runs else results_data.get("runs", []))


if __name__ == "__main__":
    main()
