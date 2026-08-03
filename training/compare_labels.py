"""
Compare YOLO label sets between two Roboflow dataset exports (e.g. dataset/
vs new_dataset/) to find which images have actually been re-labelled.

Roboflow re-hashes filenames (the `.rf.<hash>` part) on every export, so the
same source image gets a different filename each version. This script joins
old vs new labels by the filename with the `.rf.<hash>` suffix stripped, and
uses the image's md5 to disambiguate cases where that base name isn't unique
(duplicate/augmented copies of the same source image within one export).

Usage:
    python compare_labels.py [--old dataset] [--new new_dataset] [--tol 0.003]
"""

import argparse
import hashlib
import os
import re
import sys
from collections import defaultdict

RF_HASH_RE = re.compile(r"\.rf\.[0-9a-f]+")


def strip_hash(filename):
    return RF_HASH_RE.sub("", filename)


def md5_of(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def parse_boxes(label_path, tol_decimals):
    """Returns (coords, classes): coords ignores class id (so box-position
    changes are detected even if class ids got remapped between exports),
    classes is the raw set of class ids seen in the file."""
    coords = []
    classes = set()
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            classes.add(int(parts[0]))
            nums = tuple(round(float(x), tol_decimals) for x in parts[1:5])
            coords.append(nums)
    return sorted(coords), classes


def collect(dataset_dir, splits=("train", "valid")):
    """Return base_name -> list of dicts with label/image paths, keyed within
    each dataset by base filename (rf-hash stripped)."""
    entries = defaultdict(list)
    for split in splits:
        labels_dir = os.path.join(dataset_dir, split, "labels")
        images_dir = os.path.join(dataset_dir, split, "images")
        if not os.path.isdir(labels_dir):
            continue
        for fname in os.listdir(labels_dir):
            if not fname.endswith(".txt"):
                continue
            base = strip_hash(fname)
            label_path = os.path.join(labels_dir, fname)
            image_name = fname[:-4] + ".jpg"
            image_path = os.path.join(images_dir, image_name)
            if not os.path.isfile(image_path):
                # fall back: try any extension with same stem
                stem = fname[:-4]
                image_path = None
                if os.path.isdir(images_dir):
                    for cand in os.listdir(images_dir):
                        if cand.startswith(stem):
                            image_path = os.path.join(images_dir, cand)
                            break
            entries[base].append({
                "split": split,
                "filename": fname,
                "label_path": label_path,
                "image_path": image_path,
            })
    return entries


def match_pairs(old_entries, new_entries):
    """Yield (base, old_entry_or_None, new_entry_or_None, ambiguous)."""
    all_bases = set(old_entries) | set(new_entries)
    for base in sorted(all_bases):
        olds = old_entries.get(base, [])
        news = new_entries.get(base, [])

        if not olds:
            for n in news:
                yield base, None, n, False
            continue
        if not news:
            for o in olds:
                yield base, o, None, False
            continue

        if len(olds) == 1 and len(news) == 1:
            yield base, olds[0], news[0], False
            continue

        # Ambiguous: multiple copies of this base on one or both sides
        # (augmented duplicates). Try to pair by identical image md5;
        # anything left over is reported as ambiguous.
        old_by_md5 = {}
        for o in olds:
            if o["image_path"] and os.path.isfile(o["image_path"]):
                old_by_md5.setdefault(md5_of(o["image_path"]), []).append(o)

        remaining_news = list(news)
        used_olds = set()
        for n in news:
            if not n["image_path"] or not os.path.isfile(n["image_path"]):
                continue
            h = md5_of(n["image_path"])
            candidates = [o for o in old_by_md5.get(h, []) if id(o) not in used_olds]
            if candidates:
                o = candidates[0]
                used_olds.add(id(o))
                remaining_news.remove(n)
                yield base, o, n, False

        leftover_olds = [o for o in olds if id(o) not in used_olds]
        for o in leftover_olds:
            yield base, o, None, True
        for n in remaining_news:
            yield base, None, n, True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old", default="dataset", help="old dataset root (default: dataset)")
    ap.add_argument("--new", default="new_dataset", help="new dataset root (default: new_dataset)")
    ap.add_argument("--tol", type=int, default=3, help="decimal places to round box coords before comparing (default: 3)")
    ap.add_argument("--list-changed", action="store_true", help="print every changed filename, not just the count")
    ap.add_argument("--out-dir", default=None, help="if set, write changed/unchanged/new/removed/ambiguous filename lists to this directory")
    args = ap.parse_args()

    old_entries = collect(args.old)
    new_entries = collect(args.new)

    changed, unchanged, new_only, old_only, ambiguous = [], [], [], [], []
    class_mismatches = []

    for base, o, n, is_ambiguous in match_pairs(old_entries, new_entries):
        if is_ambiguous:
            ambiguous.append((base, o, n))
            continue
        if o is None:
            new_only.append((base, n))
            continue
        if n is None:
            old_only.append((base, o))
            continue

        old_coords, old_classes = parse_boxes(o["label_path"], args.tol)
        new_coords, new_classes = parse_boxes(n["label_path"], args.tol)
        if old_coords == new_coords:
            unchanged.append((base, o, n))
        else:
            changed.append((base, o, n))
        if old_classes != new_classes:
            class_mismatches.append((base, old_classes, new_classes))

    total_common = len(changed) + len(unchanged)
    print(f"Old dataset : {args.old}  ({sum(len(v) for v in old_entries.values())} labels)")
    print(f"New dataset : {args.new}  ({sum(len(v) for v in new_entries.values())} labels)")
    print()
    print(f"Matched (comparable) pairs : {total_common}")
    print(f"  Unchanged (still old box)  : {len(unchanged)}")
    print(f"  Changed (relabelled)      : {len(changed)}")
    print(f"New-only images (added)     : {len(new_only)}")
    print(f"Old-only images (missing)   : {len(old_only)}")
    print(f"Ambiguous (dup/augmented)   : {len(ambiguous)}")
    print()
    print(f"Progress estimate: {len(changed)}/{total_common} of matched images "
          f"({(len(changed)/total_common*100 if total_common else 0):.1f}%) show a different box than the old export.")

    if class_mismatches:
        print()
        print(f"WARNING: {len(class_mismatches)} label pairs use different class ids "
              f"between old and new (box coords ignored above, but this affects training!).")
        all_new_classes = set()
        for _, _, nc in class_mismatches:
            all_new_classes |= nc
        print(f"  Class ids seen in mismatched new labels: {sorted(all_new_classes)}")
        print(f"  Check new_dataset/data.yaml `names` — it currently lists more than one class;")
        print(f"  if this project should be single-class ('plate'), the extra classes are likely")
        print(f"  an annotation/export mixup, not intentional relabeling.")

    if args.list_changed:
        print("\n--- Changed filenames (new label path) ---")
        for base, o, n in changed:
            print(n["filename"])

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        def dump(name, rows, get_name):
            with open(os.path.join(args.out_dir, name), "w") as f:
                for row in rows:
                    f.write(get_name(row) + "\n")
        dump("changed.txt", changed, lambda r: r[2]["filename"])
        dump("unchanged.txt", unchanged, lambda r: r[2]["filename"])
        dump("new_only.txt", new_only, lambda r: r[1]["filename"])
        dump("old_only.txt", old_only, lambda r: r[1]["filename"])
        dump("ambiguous.txt", ambiguous, lambda r: (r[1] or r[2])["filename"])
        print(f"\nWrote filename lists to {args.out_dir}/")


if __name__ == "__main__":
    main()
