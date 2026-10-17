"""Subject-wise splits. A subject's nights never appear on both sides of a split."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import Night


@dataclass
class Fold:
    name: str
    train: list[Night]
    test: list[Night]


def subject_kfold(nights: list[Night], k: int = 5, seed: int = 0) -> list[Fold]:
    """K folds over subjects (shuffled with `seed`). Subjects are keyed by
    dataset + subject_id so pooled datasets can't collide."""
    subjects = sorted({_subject(n) for n in nights})
    if len(subjects) < k:
        raise ValueError(f"need at least {k} subjects for {k}-fold, got {len(subjects)}")
    order = np.random.default_rng(seed).permutation(len(subjects))
    groups = np.array_split(np.array(subjects, dtype=object)[order], k)
    folds = []
    for i, held_out in enumerate(groups):
        held = set(held_out)
        folds.append(Fold(
            name=f"fold{i}",
            train=[n for n in nights if _subject(n) not in held],
            test=[n for n in nights if _subject(n) in held],
        ))
    return folds


def leave_one_subject_out(nights: list[Night]) -> list[Fold]:
    subjects = sorted({_subject(n) for n in nights})
    return [
        Fold(f"loso:{s}", [n for n in nights if _subject(n) != s], [n for n in nights if _subject(n) == s])
        for s in subjects
    ]


def cross_dataset(train: list[Night], test: list[Night]) -> list[Fold]:
    a = {n.dataset for n in train}
    b = {n.dataset for n in test}
    return [Fold(f"{'+'.join(sorted(a))}->{'+'.join(sorted(b))}", train, test)]


def _subject(n: Night) -> str:
    return f"{n.dataset}/{n.subject_id}"
