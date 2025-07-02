# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "pandas",
#     "numpy",
#     "matplotlib",
#     "copairs",
# ]
# ///
"""Unified phenotypic analysis script for activity and consistency assessment."""

import logging
import argparse
from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from copairs import map
from copairs.matching import assign_reference_index

logger = logging.getLogger(__name__)


def load_and_prepare_data(
    input_file: Union[str, Path], remove_constant_columns: bool = True
) -> pd.DataFrame:
    """Load data and optionally remove constant columns."""
    df = pd.read_csv(input_file)

    if remove_constant_columns:
        df = df.loc[:, df.nunique() > 1]

    return df


def compute_phenotypic_activity(
    df: pd.DataFrame,
    compound_col: str,
    control_query: str,
    null_size: int = 1000000,
    p_threshold: float = 0.05,
    seed: int = 0,
    reference_col: str = "Metadata_reference_index",
) -> pd.DataFrame:
    """Compute phenotypic activity by comparing compounds to controls."""
    # Add reference index for controls
    df_activity = assign_reference_index(
        df,
        control_query,
        reference_col=reference_col,
        default_value=-1,
    )

    # Define positive pairs: replicates of same compound
    pos_sameby = [compound_col, reference_col]
    pos_diffby = []

    # Define negative pairs: compound vs control
    neg_sameby = []
    neg_diffby = [compound_col, reference_col]

    # Extract metadata and profiles
    metadata = df_activity.filter(regex="^Metadata")
    profiles = df_activity.filter(regex="^(?!Metadata)").values

    # Calculate average precision
    activity_ap = map.average_precision(
        metadata, profiles, pos_sameby, pos_diffby, neg_sameby, neg_diffby
    )

    # Remove controls from results
    control_mask = df_activity.query(control_query).index
    activity_ap = activity_ap[~activity_ap.index.isin(control_mask)]

    # Calculate mean average precision and p-values
    activity_map = map.mean_average_precision(
        activity_ap,
        [compound_col],
        null_size=null_size,
        threshold=p_threshold,
        seed=seed,
    )

    activity_map["-log10(p-value)"] = -activity_map["corrected_p_value"].apply(np.log10)

    return activity_map, activity_ap


def compute_phenotypic_consistency(
    df: pd.DataFrame,
    compound_col: str,
    target_col: str,
    activity_results: Optional[pd.DataFrame] = None,
    aggregate_replicates: bool = True,
    null_size: int = 1000000,
    p_threshold: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Compute phenotypic consistency by comparing compounds with same targets."""
    # Filter to active compounds if activity results provided
    if activity_results is not None:
        active_compounds = activity_results.query("below_corrected_p")[compound_col]
        df = df.query(f"{compound_col} in @active_compounds").copy()
        logger.info(f"Filtered to {len(active_compounds)} active compounds")

    # Filter out rows with null targets
    df = df.dropna(subset=[target_col]).copy()
    
    # Aggregate replicates if requested
    if aggregate_replicates:
        feature_cols = [c for c in df.columns if not c.startswith("Metadata")]
        
        # Handle multi-label targets
        if (
            df[target_col].dtype == "object"
            and df[target_col].str.contains("|", regex=False).any()
        ):
            # Group by compound and aggregate
            df_grouped = df.groupby(compound_col, as_index=False).agg(
                {
                    **{col: "median" for col in feature_cols},
                    target_col: "first",  # Keep the target info
                }
            )
            # Convert target strings to lists for multi-label handling
            df_grouped[target_col] = df_grouped[target_col].str.split("|")
            df = df_grouped
        else:
            # Simple case: group by both compound and target
            df = df.groupby([compound_col, target_col], as_index=False)[feature_cols].median()

    # Check if multi-label
    is_multilabel = df[target_col].dtype == "object" and any(
        isinstance(x, list) for x in df[target_col]
    )

    # Define positive pairs: compounds sharing targets
    pos_sameby = [target_col]
    pos_diffby = []

    # Define negative pairs: compounds with different targets
    neg_sameby = []
    neg_diffby = [target_col]

    # Extract metadata and profiles
    metadata = df.filter(regex="^Metadata")
    profiles = df.filter(regex="^(?!Metadata)").values

    # Calculate average precision
    if is_multilabel:
        target_aps = map.multilabel.average_precision(
            metadata,
            profiles,
            pos_sameby=pos_sameby,
            pos_diffby=pos_diffby,
            neg_sameby=neg_sameby,
            neg_diffby=neg_diffby,
            multilabel_col=target_col,
        )
    else:
        target_aps = map.average_precision(
            metadata, profiles, pos_sameby, pos_diffby, neg_sameby, neg_diffby
        )

    # Calculate mean average precision and p-values
    target_maps = map.mean_average_precision(
        target_aps, pos_sameby, null_size=null_size, threshold=p_threshold, seed=seed
    )

    target_maps["-log10(p-value)"] = -target_maps["corrected_p_value"].apply(np.log10)

    return target_maps, target_aps


def plot_results(
    results: pd.DataFrame,
    mode: str,
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (8, 6),
) -> None:
    """Plot mAP vs -log10(p-value) results."""
    active_ratio = results.below_corrected_p.mean()

    plt.figure(figsize=figsize)
    plt.scatter(
        data=results,
        x="mean_average_precision",
        y="-log10(p-value)",
        c="below_corrected_p",
        cmap="tab10",
        s=30,
        alpha=0.7,
    )

    title = (
        "Phenotypic Activity Assessment"
        if mode == "activity"
        else "Phenotypic Consistency Assessment"
    )
    label = "active" if mode == "activity" else "consistent"

    plt.title(title)
    plt.xlabel("mAP")
    plt.ylabel("-log10(p-value)")
    plt.axhline(-np.log10(0.05), color="black", linestyle="--", alpha=0.5)
    plt.text(
        0.65,
        1.5,
        f"Phenotypically {label} = {100 * active_ratio:.1f}%",
        va="center",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {output_path}")
    else:
        plt.show()


def main():
    """Run phenotypic analysis from command line."""
    parser = argparse.ArgumentParser(
        description="Unified phenotypic analysis for activity and consistency assessment"
    )

    parser.add_argument(
        "mode",
        choices=["activity", "consistency"],
        help="Analysis mode: 'activity' or 'consistency'",
    )

    parser.add_argument(
        "input_file", type=Path, help="Path to input CSV file with profiles"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Output directory for results (default: current directory)",
    )

    parser.add_argument(
        "--compound-col",
        default="Metadata_broad_sample",
        help="Column identifying compounds (default: Metadata_broad_sample)",
    )

    parser.add_argument(
        "--control-query",
        default="Metadata_broad_sample == 'DMSO'",
        help="Query to identify control samples (default: Metadata_broad_sample == 'DMSO')",
    )

    parser.add_argument(
        "--target-col",
        default="Metadata_target",
        help="Column identifying targets for consistency mode (default: Metadata_target)",
    )

    parser.add_argument(
        "--activity-results",
        type=Path,
        help="Path to activity results CSV for filtering in consistency mode",
    )

    parser.add_argument(
        "--aggregate-replicates",
        action="store_true",
        help="Aggregate replicates by median in consistency mode",
    )

    parser.add_argument(
        "--null-size",
        type=int,
        default=1000000,
        help="Size of null distribution for p-value calculation (default: 1000000)",
    )

    parser.add_argument(
        "--p-threshold",
        type=float,
        default=0.05,
        help="P-value threshold for significance (default: 0.05)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0)",
    )

    parser.add_argument(
        "--plot", action="store_true", help="Generate and save plot of results"
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading data from {args.input_file}")
    df = load_and_prepare_data(args.input_file)
    logger.info(f"Loaded {len(df)} profiles with {len(df.columns)} features")

    # Run analysis based on mode
    if args.mode == "activity":
        logger.info("Running phenotypic activity analysis")
        results, ap_scores = compute_phenotypic_activity(
            df,
            compound_col=args.compound_col,
            control_query=args.control_query,
            null_size=args.null_size,
            p_threshold=args.p_threshold,
            seed=args.seed,
        )

        # Save results
        output_file = args.output_dir / "activity_map.csv"
        results.to_csv(output_file, index=False)
        logger.info(f"Saved activity results to {output_file}")

        ap_file = args.output_dir / "activity_ap.csv"
        ap_scores.to_csv(ap_file, index=False)
        logger.info(f"Saved AP scores to {ap_file}")

    else:  # consistency mode
        # Load activity results if provided
        activity_results = None
        if args.activity_results:
            logger.info(f"Loading activity results from {args.activity_results}")
            activity_results = pd.read_csv(args.activity_results)

        logger.info("Running phenotypic consistency analysis")
        results, ap_scores = compute_phenotypic_consistency(
            df,
            compound_col=args.compound_col,
            target_col=args.target_col,
            activity_results=activity_results,
            aggregate_replicates=args.aggregate_replicates,
            null_size=args.null_size,
            p_threshold=args.p_threshold,
            seed=args.seed,
        )

        # Save results
        output_file = args.output_dir / "consistency_map.csv"
        results.to_csv(output_file, index=False)
        logger.info(f"Saved consistency results to {output_file}")

        ap_file = args.output_dir / "consistency_ap.csv"
        ap_scores.to_csv(ap_file, index=False)
        logger.info(f"Saved AP scores to {ap_file}")

    # Generate plot if requested
    if args.plot:
        plot_file = args.output_dir / f"{args.mode}_plot.png"
        plot_results(results, args.mode, plot_file)

    # Print summary
    n_significant = results.below_corrected_p.sum()
    n_total = len(results)
    print(f"\nAnalysis complete!")
    print(f"Total groups analyzed: {n_total}")
    print(
        f"Significant groups (p < {args.p_threshold}): {n_significant} ({100 * n_significant / n_total:.1f}%)"
    )
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
