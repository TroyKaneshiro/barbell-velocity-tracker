"""
One-off fix: new_dataset/data.yaml ended up with 4 classes
(['..', '0', 'Bar', 'Barbell']) instead of the single 'plate' class the
project actually uses. This rewrites every label file's class id to 0 and
resets data.yaml to nc: 1, names: ['plate'].

Usage:
    python collapse_classes.py [--root new_dataset]
"""

import argparse
import os


def collapse_labels(root):
    changed_files = 0
    changed_lines = 0
    for split in ("train", "valid", "test"):
        labels_dir = os.path.join(root, split, "labels")
        if not os.path.isdir(labels_dir):
            continue
        for fname in os.listdir(labels_dir):
            if not fname.endswith(".txt"):
                continue
            path = os.path.join(labels_dir, fname)
            with open(path) as f:
                lines = f.readlines()
            new_lines = []
            file_changed = False
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    new_lines.append(line)
                    continue
                parts = stripped.split()
                if parts[0] != "0":
                    changed_lines += 1
                    file_changed = True
                parts[0] = "0"
                new_lines.append(" ".join(parts) + "\n")
            if file_changed:
                changed_files += 1
                with open(path, "w") as f:
                    f.writelines(new_lines)
    return changed_files, changed_lines


def rewrite_data_yaml(root):
    path = os.path.join(root, "data.yaml")
    with open(path) as f:
        lines = f.readlines()
    out = []
    for line in lines:
        if line.startswith("nc:"):
            out.append("nc: 1\n")
        elif line.startswith("names:"):
            out.append("names: ['plate']\n")
        else:
            out.append(line)
    with open(path, "w") as f:
        f.writelines(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="new_dataset")
    args = ap.parse_args()

    changed_files, changed_lines = collapse_labels(args.root)
    rewrite_data_yaml(args.root)

    print(f"Rewrote class id to 0 in {changed_lines} box(es) across {changed_files} label file(s).")
    print(f"Updated {args.root}/data.yaml -> nc: 1, names: ['plate']")


if __name__ == "__main__":
    main()
