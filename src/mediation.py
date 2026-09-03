"""Funnel mediation: decompose treatment effects on ctor via open rate."""

from __future__ import annotations

import pandas as pd


def funnel_rates(df: pd.DataFrame, group: str) -> dict[str, float]:
    """Open rate, click rate, and click-given-open for one arm."""
    sub = df.loc[df["grupo"] == group]
    open_rate = float(sub["or"].mean())
    click_rate = float(sub["ctor"].mean())
    openers = sub.loc[sub["or"] == 1]
    click_given_open = float(openers["ctor"].mean()) if len(openers) else float("nan")
    return {
        "group": group,
        "n": float(len(sub)),
        "open_rate": open_rate,
        "click_rate": click_rate,
        "click_given_open": click_given_open,
    }


def funnel_mediation(
    df: pd.DataFrame,
    treatment: str,
    control: str = "ctrl",
) -> pd.DataFrame:
    """Decompose ATE on ``ctor`` into opening vs post-open conversion.

    Because ``ctor`` is nested in ``or`` (ctor=0 whenever or=0):

        E[ctor | T] = P(or=1 | T) × P(ctor=1 | or=1, T)

    The Kitagawa–Blinder–Oaxaca style split (treatment weights on the
    conversion path) is:

        Δctor = CTO_c × Δor  +  OR_t × ΔCTO

    where:
    - **vía apertura** ``CTO_c × Δor``: efecto por abrir más, fijando la
      conversión post-apertura del control;
    - **vía conversión** ``OR_t × ΔCTO``: efecto por convertir mejor entre
      quienes abren, fijando la apertura del tratamiento.

    Returns one row with total effect and both path contributions (and shares).
    """
    c = funnel_rates(df, control)
    t = funnel_rates(df, treatment)

    delta_or = t["open_rate"] - c["open_rate"]
    delta_cto = t["click_given_open"] - c["click_given_open"]
    delta_ctor = t["click_rate"] - c["click_rate"]

    via_open = c["click_given_open"] * delta_or
    via_convert = t["open_rate"] * delta_cto
    reconstructed = via_open + via_convert

    share_open = via_open / delta_ctor if abs(delta_ctor) > 1e-12 else float("nan")
    share_convert = via_convert / delta_ctor if abs(delta_ctor) > 1e-12 else float("nan")

    return pd.DataFrame(
        [
            {
                "comparison": f"{treatment} vs {control}",
                "open_rate_control": c["open_rate"],
                "open_rate_treatment": t["open_rate"],
                "delta_or": delta_or,
                "cto_control": c["click_given_open"],
                "cto_treatment": t["click_given_open"],
                "delta_cto": delta_cto,
                "ate_ctor": delta_ctor,
                "effect_via_open": via_open,
                "effect_via_conversion": via_convert,
                "reconstructed": reconstructed,
                "share_via_open": share_open,
                "share_via_conversion": share_convert,
            }
        ]
    )


def all_funnel_mediations(
    df: pd.DataFrame,
    comparisons: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Run funnel mediation for standard treatment pairs."""
    comparisons = comparisons or [
        ("trat1", "ctrl"),
        ("trat2", "ctrl"),
        ("trat2", "trat1"),
    ]
    return pd.concat(
        [funnel_mediation(df, treatment=t, control=c) for t, c in comparisons],
        ignore_index=True,
    )
