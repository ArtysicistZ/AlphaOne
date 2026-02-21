"""
Diagnostic: check the training data quality for entity replacement training.

Reports:
  - Total labeled pairs
  - Single-entity vs multi-entity sentence breakdown
  - Conflicting-label pairs (same sentence, different subjects, different labels)
  - Label distribution per category

Usage:
    cd backend
    python -m ml.data.check_training_data
"""

from collections import Counter, defaultdict

from sqlalchemy import func

from app.database.session import init_db, SessionLocal
from app.database.models import TrainingSentence, TrainingSentenceSubject


def main():
    init_db()
    db = SessionLocal()

    try:
        # ── 1. Total labeled pairs ──────────────────────────────────
        rows = (
            db.query(
                TrainingSentenceSubject.sentence_id,
                TrainingSentenceSubject.subject,
                TrainingSentenceSubject.sentiment_label,
            )
            .filter(TrainingSentenceSubject.sentiment_label.isnot(None))
            .all()
        )

        total_pairs = len(rows)
        print(f"Total labeled pairs: {total_pairs}")
        print()

        # ── 2. Group by sentence_id ─────────────────────────────────
        sentence_map: dict[int, list[tuple[str, str]]] = defaultdict(list)
        for sentence_id, subject, label in rows:
            sentence_map[sentence_id].append((subject, label))

        total_sentences = len(sentence_map)
        single_entity = sum(1 for v in sentence_map.values() if len(v) == 1)
        multi_entity = sum(1 for v in sentence_map.values() if len(v) > 1)

        print(f"Unique sentences:    {total_sentences}")
        print(f"  Single-entity:     {single_entity} ({100*single_entity/total_sentences:.1f}%)")
        print(f"  Multi-entity:      {multi_entity} ({100*multi_entity/total_sentences:.1f}%)")
        print()

        # ── 3. Multi-entity breakdown ───────────────────────────────
        entity_counts = Counter(len(v) for v in sentence_map.values() if len(v) > 1)
        print("Multi-entity distribution:")
        for n_entities, count in sorted(entity_counts.items()):
            print(f"  {n_entities} entities: {count} sentences")
        print()

        # ── 4. Conflicting labels ───────────────────────────────────
        # Sentences where different subjects have different sentiment labels
        conflicting = 0
        same_label = 0
        conflicting_examples = []

        for sentence_id, pairs in sentence_map.items():
            if len(pairs) < 2:
                continue
            labels_in_sentence = set(label for _, label in pairs)
            if len(labels_in_sentence) > 1:
                conflicting += 1
                if len(conflicting_examples) < 5:
                    conflicting_examples.append((sentence_id, pairs))
            else:
                same_label += 1

        print(f"Multi-entity label analysis:")
        print(f"  Conflicting labels:  {conflicting} ({100*conflicting/max(multi_entity,1):.1f}% of multi-entity)")
        print(f"  Same labels:         {same_label} ({100*same_label/max(multi_entity,1):.1f}% of multi-entity)")
        print()

        # How many total pairs come from conflicting sentences?
        conflicting_pair_count = sum(
            len(pairs) for pairs in sentence_map.values()
            if len(pairs) > 1 and len(set(l for _, l in pairs)) > 1
        )
        print(f"  Pairs from conflicting sentences: {conflicting_pair_count} ({100*conflicting_pair_count/total_pairs:.1f}% of all pairs)")
        print()

        # ── 5. Show conflicting examples ────────────────────────────
        if conflicting_examples:
            print("Conflicting-label examples:")
            for sentence_id, pairs in conflicting_examples:
                # Fetch the sentence text
                sentence = db.query(TrainingSentence.normalized_text).filter(
                    TrainingSentence.id == sentence_id
                ).scalar()
                print(f"  Text: {sentence[:100]}{'...' if sentence and len(sentence) > 100 else ''}")
                for subject, label in pairs:
                    print(f"    {subject}: {label}")
                print()

        # ── 6. Overall label distribution ───────────────────────────
        label_dist = Counter(label for _, label in
                             [(s, l) for pairs in sentence_map.values() for s, l in pairs])
        print("Label distribution:")
        for label, count in label_dist.most_common():
            print(f"  {label}: {count} ({100*count/total_pairs:.1f}%)")

    finally:
        db.close()


if __name__ == "__main__":
    main()
