"""Combine an EMG file and a label file into one file: label first, then the EMG values.

Each output line looks like:
    <label>,<ch1>,<ch2>,<ch3>,<ch4>
e.g. "1,0.32,20.72,0.28,0.22"

The original numbers are copied as text, so no precision is lost.

Usage (from this folder):
    python combine_files.py                       # combine the day1 and day2 pairs below
    python combine_files.py EMG LABELS OUTPUT     # combine any pair of files
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (EMG file, label file, combined output file)
DEFAULT_PAIRS = [
    ("emg_output.txt",      "relabel_output.txt",      "labeled_emg_output.txt"),
    ("emg_output_day2.txt", "relabel_output_day2.txt", "labeled_emg_output_day2.txt"),
]


def combine(emg_file, label_file, out_file):
    emg_lines = (HERE / emg_file).read_text().splitlines()
    label_lines = (HERE / label_file).read_text().splitlines()
    # Ignore blank trailing lines.
    emg_lines = [l.strip() for l in emg_lines if l.strip()]
    label_lines = [l.strip() for l in label_lines if l.strip()]
    if len(emg_lines) != len(label_lines):
        raise ValueError(
            f"{emg_file} has {len(emg_lines)} rows but {label_file} has {len(label_lines)}"
        )
    with open(HERE / out_file, "w", newline="\n") as f:
        for lab, emg in zip(label_lines, emg_lines):
            f.write(f"{int(float(lab))},{emg}\n")
    print(f"{emg_file} + {label_file} -> {out_file} ({len(emg_lines):,} rows)")


if __name__ == "__main__":
    if len(sys.argv) == 4:
        combine(*sys.argv[1:])
    elif len(sys.argv) == 1:
        for pair in DEFAULT_PAIRS:
            combine(*pair)
    else:
        sys.exit(__doc__)

