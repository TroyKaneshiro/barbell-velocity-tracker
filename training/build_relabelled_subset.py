"""
Build a YOLO dataset containing only the images whose labels actually
changed between two Roboflow exports (plus any brand-new images), so you
can train on just the relabelled subset instead of the full mixed export.

Roboflow can only export the whole project, never "just what changed", so
the workflow is: relabel some images in Roboflow -> export the full
project again -> run this script against the previous export (--old) and
the new one (--new) to pull out only what actually changed.

Reuses the matching logic from compare_labels.py (filename with the
`.rf.<hash>` suffix stripped, disambiguated by image md5 for duplicates).

Usage:
    python build_relabelled_subset.py [--old dataset] [--new new_dataset] [--out relabelled_dataset]
"""

import argparse
import os
import shutil

from compare_labels import collect, match_pairs, parse_boxes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old", default="dataset", help="previous export (baseline to diff against)")
    ap.add_argument("--new", default="new_dataset", help="latest export")
    ap.add_argument("--out", default="relabelled_dataset", help="output dataset dir")
    ap.add_argument("--tol", type=int, default=3, help="decimal places to round box coords before comparing")
    args = ap.parse_args()

    if os.path.isdir(args.out):
        shutil.rmtree(args.out)

    old_entries = collect(args.old)
    new_entries = collect(args.new)

    kept = []
    stats = {"changed": 0, "new_only": 0, "unchanged_skipped": 0, "old_only_skipped": 0, "ambiguous_skipped": 0}

    for base, o, n, is_ambiguous in match_pairs(old_entries, new_entries):
        if is_ambiguous:
            stats["ambiguous_skipped"] += 1
            continue
        if n is None:
            # Image existed before but is gone from the new export
            # (e.g. deleted for being low-res/grainy/bad angle) - not relabelled data.
            stats["old_only_skipped"] += 1
            continue
        if o is None:
            kept.append(n)
            stats["new_only"] += 1
            continue

        old_coords, _ = parse_boxes(o["label_path"], args.tol)
        new_coords, _ = parse_boxes(n["label_path"], args.tol)
        if old_coords == new_coords:
            stats["unchanged_skipped"] += 1
        else:
            kept.append(n)
            stats["changed"] += 1

    for entry in kept:
        images_dir = os.path.join(args.out, entry["split"], "images")
        labels_dir = os.path.join(args.out, entry["split"], "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        shutil.copy(entry["label_path"], os.path.join(labels_dir, entry["filename"]))
        if entry["image_path"] and os.path.isfile(entry["image_path"]):
            shutil.copy(entry["image_path"], os.path.join(images_dir, os.path.basename(entry["image_path"])))
        else:
            print(f"WARNING: no image file found for {entry['filename']}, skipping copy")

    with open(os.path.join(args.out, "data.yaml"), "w") as f:
        f.write(
            f"path: {os.path.abspath(args.out)}\n"
            "train: train/images\n"
            "val: valid/images\n"
            "\n"
            "nc: 1\n"
            "names: ['plate']\n"
        )

    n_train = sum(1 for e in kept if e["split"] == "train")
    n_valid = sum(1 for e in kept if e["split"] == "valid")
    print(f"Kept {len(kept)} images ({n_train} train / {n_valid} valid) -> {args.out}/")
    print(f"  changed (relabelled): {stats['changed']}")
    print(f"  new-only (added):     {stats['new_only']}")
    print(f"  skipped, unchanged:   {stats['unchanged_skipped']}")
    print(f"  skipped, old-only:    {stats['old_only_skipped']}")
    print(f"  skipped, ambiguous:   {stats['ambiguous_skipped']}")


if __name__ == "__main__":
    main()
