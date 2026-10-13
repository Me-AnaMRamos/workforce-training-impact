"""Generate a synthetic DL2/DL3 workforce skeleton for Jan 2018--Dec 2020.

Only the total number of workers (2,044), trained (1,656), untrained (388),
levels (DL2/DL3), and observation window are treated as paper-established here.
All other allocation/distribution choices are SIMULATION ASSUMPTIONS and must be
revised if supported details are recovered from the paper.

Run from project root:
    python src/generate_workforce.py
    python src/generate_workforce.py --seed 123 --output data/synthetic/workforce.csv

Dependencies: numpy, pandas
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Paper-established constraints supplied for this replication exercise.
N_WORKERS = 2044
N_TRAINED = 1656
N_UNTRAINED = 388
START_MONTH = pd.Period("2018-01", freq="M")
END_MONTH = pd.Period("2020-12", freq="M")

# SIMULATION ASSUMPTIONS (not empirically identified by the supplied counts).
FACTORY_IDS = ("F01", "F02")
N_SUPERVISORS = 345  # One synthetic supervisor per synthetic cell.
PROB_DL2 = 0.50
TRAINING_COHORTS = tuple(pd.period_range("2018-07", "2019-06", freq="M"))

PROB_NO_EXIT_BY_2020_12 = 0.65


def random_month(rng: np.random.Generator, first: pd.Period, last: pd.Period) -> pd.Period:
    """Draw a month uniformly from [first, last], inclusive."""
    if last < first:
        raise ValueError(f"Invalid month interval: {first} > {last}")
    return first + int(rng.integers(0, last.ordinal - first.ordinal + 1))


def make_workforce(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if N_TRAINED + N_UNTRAINED != N_WORKERS:
        raise ValueError("Training totals must equal workforce total")

    # Exactly 1,656 ultimately trained and 388 never trained in this window.
    trained = np.zeros(N_WORKERS, dtype=bool)
    trained[rng.choice(N_WORKERS, size=N_TRAINED, replace=False)] = True

    # Factory / level / cell / supervisor assignments are synthetic placeholders.
    factory = rng.choice(FACTORY_IDS, size=N_WORKERS)
    cell_number = rng.integers(1, CELLS_PER_FACTORY + 1, size=N_WORKERS)
    cell_id = [f"{f}_C{c:02d}" for f, c in zip(factory, cell_number)]
    supervisor_id = [f"{f}_S{c:02d}" for f, c in zip(factory, cell_number)]
    level = rng.choice(["DL2", "DL3"], size=N_WORKERS,
                       p=[PROB_DL2, 1 - PROB_DL2])

    rows = []
    for i in range(N_WORKERS):
        if trained[i]:
            # This is an illustrative INDIVIDUAL cohort assignment, NOT a
            # claim that the experiment assigned training at worker level.
            cohort = TRAINING_COHORTS[int(rng.integers(len(TRAINING_COHORTS)))]
            # Ensure worker was employed before the training month.
            hire_month = random_month(rng, START_MONTH, cohort - 1)
            earliest_exit_month = cohort
        else:
            cohort = None
            hire_month = random_month(rng, START_MONTH, END_MONTH)
            earliest_exit_month = hire_month

        # Month-boundary dates are a modeling convention, not exact observed dates.
        hire_date = hire_month.to_timestamp(how="start").date()
        if rng.random() < PROB_NO_EXIT_BY_2020_12:
            exit_date = pd.NaT  # Still employed at the end of observation.
        else:
            exit_month = random_month(rng, earliest_exit_month, END_MONTH)
            exit_date = exit_month.to_timestamp(how="end").date()

        rows.append({
            "worker_id": f"W{i + 1:04d}",
            "factory_id": factory[i],
            "level": level[i],
            "cell_id": cell_id[i],
            "supervisor_id": supervisor_id[i],
            "hire_date": hire_date,
            "exit_date": exit_date,
            "ultimately_trained": int(trained[i]),
            "training_cohort": str(cohort) if cohort is not None else None,
        })

    workforce = pd.DataFrame(rows)
    validate(workforce)
    return workforce


def validate(df: pd.DataFrame) -> None:
    assert len(df) == N_WORKERS
    assert df.worker_id.nunique() == N_WORKERS
    assert int(df.ultimately_trained.sum()) == N_TRAINED
    assert int((df.ultimately_trained == 0).sum()) == N_UNTRAINED
    assert set(df.level.unique()) <= {"DL2", "DL3"}
    assert df.loc[df.ultimately_trained == 1, "training_cohort"].notna().all()
    assert df.loc[df.ultimately_trained == 0, "training_cohort"].isna().all()
    assert (df.groupby("cell_id").factory_id.nunique() == 1).all()
    assert (df.groupby("cell_id").supervisor_id.nunique() == 1).all()
    hire = pd.to_datetime(df.hire_date)
    exit_ = pd.to_datetime(df.exit_date)
    assert (hire >= START_MONTH.to_timestamp()).all()
    assert (hire <= END_MONTH.to_timestamp(how="end")).all()
    assert (exit_.dropna() >= hire.loc[exit_.notna()]).all()
    assert (exit_.dropna() <= END_MONTH.to_timestamp(how="end")).all()
    cohort = pd.to_datetime(df.training_cohort, errors="coerce")
    assert (hire.loc[cohort.notna()] < cohort.dropna()).all()
    assert (exit_.loc[cohort.notna()].dropna() >= cohort.loc[exit_.notna() & cohort.notna()]).all()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path,
                        default=Path("data/synthetic/workforce.csv"))
    args = parser.parse_args()
    df = make_workforce(seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, date_format="%Y-%m-%d")
    print(f"Saved {len(df):,} unique workers to {args.output}")
    print(df["ultimately_trained"].value_counts().sort_index().rename(
        index={0: "Untrained", 1: "Trained"}).to_string())
    print("Workers by level:", df.level.value_counts().to_dict())
    print("Training cohorts:", df.training_cohort.value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
