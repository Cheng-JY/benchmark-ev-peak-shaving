r"""
python src\scripts\shortcut_reduce_ch_avail.py "data\data_20260928_2\ch_profiles_agg_home_DEU_20260928_2026_cal2026_R0_0.1.csv"  --output "data\data_20260928_2\ch_profiles_agg_home_DEU_20260928_2026_cal2026_R0_0.1.csv"

"""


from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_COL = "ch_avail_refuel_cp"

# Fixed experiment settings
REDUCE_FACTORS = [round(x / 10, 2) for x in range(1, 10)]
VALLEY_FACTOR = 0.80
SHAPE = 1.0
PEAK_BASE_SHARE = 0.70


def factor_tag(value: float) -> str:
    """Format a numeric availability factor for use in a column name.
    
    Parameters
    ----------
    value : float
        Value used by this operation.
    
    Returns
    -------
    str
        Result computed as ``f'{float(value):.2f}'``."""
    return f"{float(value):.2f}"


def build_column_names(source_col: str, factor: float) -> tuple[str, str]:
    """Build column names.
    
    Parameters
    ----------
    source_col : str
        Source col used by this operation.
    factor : float
        Factor used by this operation.
    
    Returns
    -------
    tuple[str, str]
        Computed result for the selected input and execution branch."""
    tag = factor_tag(factor)

    if source_col.endswith("_cp"):
        prefix = source_col[:-3]
        return (
            f"{prefix}_reduced_flat{tag}_cp",
            f"{prefix}_reduced_peak{tag}_cp",
        )

    return (
        f"{source_col}_reduced_flat{tag}",
        f"{source_col}_reduced_peak{tag}",
    )


def add_reduction_sweep(
    df: pd.DataFrame,
    *,
    source_col: str = SOURCE_COL,
    overwrite: bool = True,
) -> pd.DataFrame:
    """
    Generate reduced_flat and reduced_peak columns for factors 0.10 ... 0.90.

    Fixed parameters:
        valley_factor   = 0.80
        shape           = 1.0
        peak_base_share = 0.70

    Flat:
        flat = source * reduce_factor

    Peak:
        threshold = PEAK_BASE_SHARE * max(source)

        z = clip(source / threshold, 0, 1)

        factor_peak = VALLEY_FACTOR
                      - (VALLEY_FACTOR - reduce_factor) * z**SHAPE

        if source >= threshold:
            factor_peak = reduce_factor

        peak = source * factor_peak
    """
    if source_col not in df.columns:
        raise KeyError(
            f"Source column {source_col!r} not found.\n"
            f"Available columns:\n{list(df.columns)}"
        )

    values = pd.to_numeric(df[source_col], errors="coerce")

    finite = values.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]

    max_val = float(np.max(finite)) if finite.size else np.nan
    threshold = PEAK_BASE_SHARE * max_val

    for reduce_factor in REDUCE_FACTORS:
        flat_col, peak_col = build_column_names(
            source_col,
            reduce_factor,
        )

        if not overwrite and (
            flat_col in df.columns or peak_col in df.columns
        ):
            print(f"Skip existing factor {reduce_factor:.2f}")
            continue

        # Flat reduction
        df[flat_col] = values * reduce_factor

        # Peak-weighted reduction
        lower = min(reduce_factor, VALLEY_FACTOR)
        upper = max(reduce_factor, VALLEY_FACTOR)

        if (
            np.isfinite(max_val)
            and max_val > 0
            and np.isfinite(threshold)
            and threshold > 0
        ):
            z = (values / threshold).clip(lower=0.0, upper=1.0)

            factor_series = (
                VALLEY_FACTOR
                - (VALLEY_FACTOR - reduce_factor)
                * np.power(z, SHAPE)
            )

            factor_series = pd.Series(
                factor_series,
                index=values.index,
                dtype=float,
            )

            factor_series = factor_series.mask(
                values >= threshold,
                reduce_factor,
            )

        else:
            factor_series = pd.Series(
                reduce_factor,
                index=values.index,
                dtype=float,
            )

        factor_series = factor_series.clip(
            lower=lower,
            upper=upper,
        )

        df[peak_col] = values * factor_series

        valid = values.replace(0, np.nan)
        peak_ratio = df[peak_col] / valid

        print(
            f"{reduce_factor:.2f}: "
            f"flat={reduce_factor:.2f}, "
            f"peak factor range="
            f"{peak_ratio.min(skipna=True):.4f} .. "
            f"{peak_ratio.max(skipna=True):.4f}"
        )

    print()
    print(f"max({source_col}) = {max_val:.6f}")
    print(f"peak threshold     = {threshold:.6f}")
    print(f"valley factor      = {VALLEY_FACTOR}")
    print(f"shape              = {SHAPE}")
    print(f"peak base share    = {PEAK_BASE_SHARE}")

    return df


def update_csv(
    csv_path: Path,
    *,
    output_path: Path | None = None,
    source_col: str = SOURCE_COL,
    overwrite: bool = True,
) -> Path:
    """Update csv.
    
    Parameters
    ----------
    csv_path : Path
        Csv path used by this operation.
    output_path : Path | None
        Output path used by this operation.
    source_col : str
        Source col used by this operation.
    overwrite : bool
        Overwrite used by this operation.
    
    Returns
    -------
    Path
        Result computed as ``target``.
    
    Raises
    ------
    FileNotFoundError
        If an explicit validation or processing check in this operation fails."""
    csv_path = csv_path.expanduser().resolve()

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path, sep=";")

    df = add_reduction_sweep(
        df,
        source_col=source_col,
        overwrite=overwrite,
    )

    target = (
        output_path.expanduser().resolve()
        if output_path is not None
        else csv_path
    )

    target.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        target,
        sep=";",
        decimal=".",
        index=False,
        float_format="%.6f",
    )

    print()
    print(f"Saved: {target}")

    return target


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser.
    
    Returns
    -------
    argparse.ArgumentParser
        Result computed as ``parser``."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate reduced_flat and reduced_peak columns "
            "for reduce factors 0.10 ... 0.90 using fixed "
            "experiment parameters."
        )
    )

    parser.add_argument(
        "csv",
        type=Path,
        help="Input aggregation CSV",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output CSV. "
            "If omitted, the input CSV is overwritten."
        ),
    )

    parser.add_argument(
        "--source-col",
        default=SOURCE_COL,
        help=f"Source availability column (default: {SOURCE_COL})",
    )

    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not overwrite already existing reduced columns.",
    )

    return parser


def main() -> None:
    """Run the command-line workflow for this module.
    
    Returns
    -------
    None"""
    args = build_parser().parse_args()

    update_csv(
        args.csv,
        output_path=args.output,
        source_col=args.source_col,
        overwrite=not args.keep_existing,
    )


if __name__ == "__main__":
    main()