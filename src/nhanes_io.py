from pathlib import Path
import pyreadstat
import pandas as pd

RAW_DIR = Path("data/raw_xpt")

def load_xpt_folder(raw_dir, key: str = "SEQN") -> dict[str, pd.DataFrame]:
    """
    Loads all .xpt/.XPT files in raw_dir into a dict of pandas DataFrames.
    Dict keys are file stems, e.g. 'DEMO_J' from 'DEMO_J.xpt'.
    """
    raw_dir = Path(raw_dir)   # <-- ADD THIS LINE

    xpt_files = sorted(
        list(raw_dir.glob("*.xpt")) +
        list(raw_dir.glob("*.XPT"))
    )

    if not xpt_files:
        raise FileNotFoundError(f"No XPT files found in {raw_dir.resolve()}")

    tables: dict[str, pd.DataFrame] = {}
    missing_key = []

    for path in xpt_files:
        df, meta = pyreadstat.read_xport(path)
        name = path.stem  # DEMO_J
        tables[name] = df

        if key not in df.columns:
            missing_key.append(name)

        print(f"Loaded {name}: shape={df.shape}")

    if missing_key:
        print(f"\nWarning: {len(missing_key)} table(s) missing '{key}': {missing_key}")

    return tables

KEY = "SEQN"

def left_join_many(
    base: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    table_names: list[str],
    key: str = KEY,
) -> pd.DataFrame:
    df = base.copy()

    for name in table_names:
        t = tables[name].copy()

        # If multiple rows per person, collapse to 1 row/person (numeric means)
        if t[key].duplicated().any():
            t = t.groupby(key, as_index=False).mean(numeric_only=True)

        # Avoid column collisions (except key)
        overlapping = set(df.columns) & set(t.columns) - {key}
        if overlapping:
            t = t.rename(columns={c: f"{c}_{name.lower()}" for c in overlapping})

        before = df.shape[1]
        df = df.merge(t, on=key, how="left")
        after = df.shape[1]

        print(f"Joined {name}: cols {before} -> {after}, matched rows = {df[key].notna().sum()}")

    return df

