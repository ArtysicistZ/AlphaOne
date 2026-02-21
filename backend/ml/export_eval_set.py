"""Export the eval/test split to CSV for manual label audit."""

import csv
import os
import sys

from sklearn.model_selection import train_test_split

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.train_base import load_labeled_data, apply_entity_replacement, ID2LABEL

SEED = 42


def main():
    texts, subjects, labels = load_labeled_data()
    print(f"Total labeled pairs: {len(texts)}")

    # Apply entity replacement (same as training)
    modified_texts = [
        apply_entity_replacement(text, subject)
        for text, subject in zip(texts, subjects)
    ]

    # Same split as training: 90/10, seed=42, stratified
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        list(zip(texts, modified_texts, subjects)), labels,
        test_size=0.1, random_state=SEED, stratify=labels,
    )

    # Unpack
    test_original = [t[0] for t in test_texts]
    test_modified = [t[1] for t in test_texts]
    test_subjects = [t[2] for t in test_texts]

    out_path = os.path.join(os.path.dirname(__file__), "eval_set_audit.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "original_text", "modified_text", "subject", "label", "correct?", "notes"])
        for i, (orig, mod, subj, label) in enumerate(
            zip(test_original, test_modified, test_subjects, test_labels)
        ):
            writer.writerow([i, orig, mod, subj, ID2LABEL[label], "", ""])

    print(f"Exported {len(test_labels)} eval examples to: {out_path}")

    # Print distribution
    from collections import Counter
    dist = Counter(ID2LABEL[l] for l in test_labels)
    for label, count in sorted(dist.items()):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
