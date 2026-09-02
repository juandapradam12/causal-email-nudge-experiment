"""Data loading and preprocessing for the email nudge experiment."""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "datos_prueba_tecnica.csv"

CATEGORICAL_COLS = ["grupo", "tipo_tarjeta", "formacion"]
BINARY_COLS = ["or", "ctor", "sexo", "uso_app", "tarjeta_debito"]
NUMERIC_COLS = ["edad", "inve"]

GROUP_LABELS = {
    "ctrl": "Control (sin nudge)",
    "trat1": "Tratamiento 1 (nudge 1)",
    "trat2": "Tratamiento 2 (nudge 2)",
}


def load_data(path: Path | str | None = None) -> pd.DataFrame:
    """Load and type the experiment dataset."""
    df = pd.read_csv(path or DATA_PATH)

    df["grupo"] = pd.Categorical(
        df["grupo"],
        categories=["ctrl", "trat1", "trat2"],
        ordered=True,
    )
    for col in BINARY_COLS:
        df[col] = df[col].astype(int)

    df["tipo_tarjeta"] = df["tipo_tarjeta"].astype("category")
    df["formacion"] = df["formacion"].astype("category")

    return df


def treatment_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode treatment groups (control as reference)."""
    return pd.get_dummies(df["grupo"], prefix="grupo", drop_first=False)
