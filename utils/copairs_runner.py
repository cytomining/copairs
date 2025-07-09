# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "pandas",
#     "numpy",
#     "copairs",
#     "pyyaml",
#     "matplotlib",
#     "seaborn",
# ]
# ///

"""Generic runner for copairs analyses with configuration support."""

import logging
from typing import Any, Dict, Union, Optional
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from copairs import map
from copairs.matching import assign_reference_index

logger = logging.getLogger(__name__)


class CopairsRunner:
    """Generic runner for copairs analyses.

    This runner supports:
    - Loading data from CSV/Parquet files
    - Preprocessing steps (filtering, reference assignment, metadata merging, aggregation)
    - Running average precision calculations
    - Running mean average precision with significance testing
    - Plotting mAP vs -log10(p-value) scatter plots
    - Saving results

    Configuration Notes:
    - By default, metadata columns are identified using the regex "^Metadata".
      You can override this by setting data.metadata_regex in your config.
    - To enable plotting, add a "plotting" section to your config with "enabled: true".

    Parameter Passing:
    The runner validates that required parameters are present but passes ALL parameters
    specified in the config to the underlying copairs functions. This means you can
    specify any additional parameters supported by the copairs functions:

    For average_precision and multilabel.average_precision:
    - Required: pos_sameby, pos_diffby, neg_sameby, neg_diffby
    - Optional: batch_size (default: 20000), distance (default: "cosine"),
      progress_bar (default: True), and others

    For mean_average_precision:
    - Required: sameby, null_size, threshold, seed
    - Optional: progress_bar (default: True), max_workers (default: CPU count + 4),
      cache_dir (default: None), and others

    Example config with optional parameters:
    ```yaml
    average_precision:
      params:
        pos_sameby: ["Metadata_gene_symbol"]
        pos_diffby: []
        neg_sameby: []
        neg_diffby: ["Metadata_cell_line"]
        batch_size: 50000  # Optional: larger batch for more memory
        distance: "euclidean"  # Optional: different distance metric
    ```

    Refer to the copairs function signatures for complete parameter details:
    - copairs.map.average_precision
    - copairs.map.multilabel.average_precision
    - copairs.map.mean_average_precision
    """

    def __init__(self, config: Union[Dict[str, Any], str, Path]):
        """Initialize runner with configuration.

        Parameters
        ----------
        config : dict, str, or Path
            Configuration dictionary or path to YAML config file
        """
        # Load config if it's a file path
        if isinstance(config, (str, Path)):
            with open(config, "r") as f:
                config = yaml.safe_load(f)

        self.config = config

    def run(self) -> pd.DataFrame:
        """Run the complete analysis pipeline.

        Returns
        -------
        pd.DataFrame
            Final analysis results
        """
        logger.info("Starting copairs analysis")

        # 1. Load data
        path = Path(self.config["data"]["path"])
        logger.info(f"Loading data from {path}")
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")

        # 2. Preprocess
        df = self.preprocess_data(df)

        # 3. Extract metadata and features
        metadata = df.filter(regex="^Metadata")
        feature_cols = [col for col in df.columns if not col.startswith("Metadata")]
        features = df[feature_cols].values
        logger.info(
            f"Extracted {metadata.shape[1]} metadata columns and {features.shape[1]} features"
        )

        # 4. Run average precision
        ap_results = self.run_average_precision(metadata, features)

        # 5. Save AP results if requested
        if self.config["output"].get("save_ap_scores", False):
            self.save_results(ap_results, suffix="ap_scores")

        # 6. Run mean average precision
        final_results = self.run_mean_average_precision(ap_results)

        # 7. Generate and save plot if enabled
        if (
            "mean_average_precision" in self.config
            and "-log10(p-value)" in final_results.columns
        ):
            self.plot_map_results(final_results)

        # 8. Save final results
        self.save_results(final_results)

        logger.info("Analysis complete")
        return final_results

    def run_average_precision(
        self, metadata: pd.DataFrame, features: np.ndarray
    ) -> pd.DataFrame:
        """Run average precision calculation.

        Parameters
        ----------
        metadata : pd.DataFrame
            Metadata dataframe
        features : np.ndarray
            Feature array

        Returns
        -------
        pd.DataFrame
            Average precision results
        """
        ap_config = self.config["average_precision"]
        params = ap_config["params"]

        # Check if multilabel
        if ap_config.get("multilabel", False):
            logger.info("Running multilabel average precision")
            results = map.multilabel.average_precision(metadata, features, **params)
        else:
            logger.info("Running average precision")
            results = map.average_precision(metadata, features, **params)

        return results

    def run_mean_average_precision(self, ap_results: pd.DataFrame) -> pd.DataFrame:
        """Run mean average precision if configured.

        Parameters
        ----------
        ap_results : pd.DataFrame
            Average precision results

        Returns
        -------
        pd.DataFrame
            Mean average precision results with p-values
        """
        if "mean_average_precision" not in self.config:
            return ap_results

        map_config = self.config["mean_average_precision"]
        params = map_config["params"]

        logger.info("Running mean average precision")
        map_results = map.mean_average_precision(ap_results, **params)

        # Add -log10(p-value) column if not present
        if "corrected_p_value" in map_results.columns:
            map_results["-log10(p-value)"] = -map_results["corrected_p_value"].apply(
                np.log10
            )

        return map_results

    def save_results(self, results: pd.DataFrame, suffix: str = ""):
        """Save results to configured output path.

        Parameters
        ----------
        results : pd.DataFrame
            Results dataframe to save
        suffix : str, optional
            Suffix to add to filename, by default ""
        """
        output_config = self.config["output"]
        output_path = Path(output_config["path"])

        # Add suffix if provided
        if suffix:
            output_path = output_path.with_name(
                output_path.stem + f"_{suffix}" + output_path.suffix
            )

        # Create directory if needed and save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix == ".parquet":
            results.to_parquet(output_path, index=False)
        else:
            results.to_csv(output_path, index=False)
        logger.info(f"Saved results to {output_path}")

    def plot_map_results(
        self,
        map_results: pd.DataFrame,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Optional[plt.Figure]:
        """Create and optionally save a scatter plot of mean average precision vs -log10(p-value).

        Parameters
        ----------
        map_results : pd.DataFrame
            Results from mean_average_precision containing 'mean_average_precision',
            'corrected_p_value', and 'below_corrected_p' columns
        save_path : str or Path, optional
            If provided, save the plot to this path. If None, uses config settings.

        Returns
        -------
        plt.Figure or None
            The matplotlib figure object if created, None if plotting is disabled
        """
        # Check if plotting is enabled
        plot_config = self.config.get("plotting", {})
        if not plot_config.get("enabled", False):
            return None

        # Get plot parameters from config
        title = plot_config.get("title")
        xlabel = plot_config.get("xlabel", "mAP")
        ylabel = plot_config.get("ylabel", "-log10(p-value)")
        annotation_prefix = plot_config.get("annotation_prefix", "Significant")
        figsize = tuple(plot_config.get("figsize", [8, 6]))
        dpi = plot_config.get("dpi", 100)

        # Set seaborn style
        sns.set_style("whitegrid", {"axes.grid": False})

        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Calculate percentage of significant results
        significant_ratio = map_results["below_corrected_p"].mean()

        # Better color scheme
        colors = map_results["below_corrected_p"].map(
            {True: "#2166ac", False: "#969696"}
        )

        # Create scatter plot with better styling
        ax.scatter(
            data=map_results,
            x="mean_average_precision",
            y="-log10(p-value)",
            c=colors,
            s=40,
            alpha=0.6,
            edgecolors="none",
        )

        # Add significance threshold line
        ax.axhline(
            -np.log10(0.05), color="#d6604d", linestyle="--", linewidth=1.5, alpha=0.8
        )

        # Add annotation without box (top left)
        ax.text(
            0.02,
            0.98,
            f"{annotation_prefix}: {100 * significant_ratio:.1f}%",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=11,
            color="#525252",
        )

        # Remove top and right spines (range frames)
        sns.despine()

        # Set x-axis limits to always show 0-1.05 range
        ax.set_xlim(0, 1.05)

        # Set y-axis limits based on the null size

        null_size = (
            self.config["mean_average_precision"].get("params", {}).get("null_size")
            if "mean_average_precision" in self.config
            else None
        )
        assert null_size  # This must exist if we are plotting mAP

        ymax = -np.log10(1 / (1 + null_size))
        ax.set_ylim(0, ymax)

        # Set labels with better formatting
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        if title:
            ax.set_title(title, fontsize=14, pad=20)

        # Customize grid
        ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)

        # Adjust layout
        plt.tight_layout()

        # Save plot if path is provided
        if save_path is None:
            save_path = plot_config.get("path")

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Get format from config or infer from extension
            plot_format = plot_config.get("format")
            if not plot_format and save_path.suffix:
                plot_format = save_path.suffix[1:]  # Remove the dot
            elif not plot_format:
                plot_format = "png"

            fig.savefig(save_path, format=plot_format, bbox_inches="tight")
            logger.info(f"Saved plot to {save_path}")

            # Close figure to free memory
            plt.close(fig)
            return None  # Return None since figure is closed

        return fig

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing steps to data.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe

        Returns
        -------
        pd.DataFrame
            Preprocessed dataframe

        Notes
        -----
        Available preprocessing steps (all parameters must be under 'params'):

        - filter: Filter rows using pandas query syntax
        - dropna: Drop rows with NaN values in specified columns
        - remove_nan_features: Remove feature columns containing NaN
        - split_multilabel: Split pipe-separated values into lists
        - filter_active: Filter based on activity CSV with below_corrected_p column
        - aggregate_replicates: Aggregate by taking median of features
        - merge_metadata: Merge external CSV metadata
        - filter_single_replicates: Remove groups with < min_replicates members
        - apply_assign_reference: Apply copairs.matching.assign_reference_index

        Example:
        ```yaml
        preprocessing:
          - type: filter
            params:
              query: "Metadata_mmoles_per_liter > 0.1"
          - type: filter_active
            params:
              activity_csv: "data/activity_map.csv"
              on: "Metadata_broad_sample"
        ```
        """
        if "preprocessing" not in self.config:
            return df

        for step in self.config["preprocessing"]:
            step_type = step["type"]
            logger.info(f"Applying preprocessing: {step_type}")

            # Get parameters from the params section
            params = step.get("params", {})
            if not params and step_type != "remove_nan_features":
                # remove_nan_features doesn't require params
                raise ValueError(
                    f"Preprocessing step '{step_type}' requires a 'params' section"
                )

            # Use getattr to call the appropriate preprocessing method
            method_name = f"_preprocess_{step_type}"
            if hasattr(self, method_name):
                try:
                    df = getattr(self, method_name)(df, params)
                except KeyError as e:
                    raise ValueError(
                        f"Missing required parameter {e} for preprocessing step '{step_type}'"
                    ) from e
            else:
                logger.warning(f"Unknown preprocessing type: {step_type}")

        return df

    def _get_feature_columns(self, df: pd.DataFrame) -> list:
        """Get non-metadata columns."""
        return [col for col in df.columns if not col.startswith("Metadata")]

    def _preprocess_filter(
        self, df: pd.DataFrame, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Filter rows using pandas query syntax."""
        df = df.query(params["query"])
        logger.info(f"After filter '{params['query']}': {len(df)} rows")
        return df

    def _preprocess_apply_assign_reference(
        self, df: pd.DataFrame, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Apply copairs.matching.assign_reference_index."""
        return assign_reference_index(df, **params)

    def _preprocess_dropna(
        self, df: pd.DataFrame, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Drop rows with NaN values."""
        columns = params.get("columns")
        df = df.dropna(subset=columns)
        logger.info(f"After dropna: {len(df)} rows")
        return df

    def _preprocess_remove_nan_features(
        self, df: pd.DataFrame, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Remove feature columns containing NaN."""
        feature_cols = self._get_feature_columns(df)
        nan_cols = df[feature_cols].isna().any()
        nan_cols = nan_cols[nan_cols].index.tolist()

        if nan_cols:
            df = df.drop(columns=nan_cols)
            logger.info(f"Removed {len(nan_cols)} features with NaN values")
        return df

    def _preprocess_split_multilabel(
        self, df: pd.DataFrame, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Split pipe-separated values into lists."""
        column = params["column"]
        separator = params.get("separator", "|")
        df[column] = df[column].str.split(separator)
        logger.info(f"Split multilabel column '{column}' by '{separator}'")
        return df

    def _preprocess_filter_active(
        self, df: pd.DataFrame, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Filter to active compounds based on below_corrected_p column."""
        activity_csv = Path(params["activity_csv"])
        on_column = params["on_column"]
        filter_column = params.get("filter_column", "below_corrected_p")

        # Load activity data
        activity_df = pd.read_csv(activity_csv)

        # Get active compounds
        active_values = activity_df[activity_df[filter_column] == True][
            on_column
        ].unique()

        df = df[df[on_column].isin(active_values)]

        logger.info(f"Filtered to {len(df)} active compounds from {activity_csv}")
        return df

    def _preprocess_aggregate_replicates(
        self, df: pd.DataFrame, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Aggregate replicates by taking median of features."""
        groupby_cols = params["groupby"]
        feature_cols = self._get_feature_columns(df)

        df = df.groupby(groupby_cols, as_index=False)[feature_cols].median()

        logger.info(f"Aggregated to {len(df)} rows by grouping on {groupby_cols}")
        return df

    def _preprocess_merge_metadata(
        self, df: pd.DataFrame, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Merge external metadata from CSV file."""
        source_path = Path(params["source"])
        on_columns = params["on"] if isinstance(params["on"], list) else [params["on"]]
        how = params.get("how", "left")

        # Load external metadata
        metadata_df = pd.read_csv(source_path)
        logger.info(f"Loaded metadata from {source_path}: {len(metadata_df)} rows")

        original_len = len(df)
        df = df.merge(metadata_df, on=on_columns, how=how)

        logger.info(
            f"Merged metadata on {on_columns} ({how} join): "
            f"{original_len} -> {len(df)} rows"
        )
        return df

    def _preprocess_filter_single_replicates(
        self, df: pd.DataFrame, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Remove groups with insufficient replicates."""
        groupby_cols = params["groupby"]
        min_replicates = params.get("min_replicates", 2)

        # Count replicates per group
        group_counts = df.groupby(groupby_cols).size()

        # Keep only groups with enough replicates
        valid_groups = group_counts[group_counts >= min_replicates].index

        # Filter to valid groups
        original_len = len(df)
        if len(groupby_cols) == 1:
            df = df[df[groupby_cols[0]].isin(valid_groups)]
        else:
            valid_df = pd.DataFrame(list(valid_groups), columns=groupby_cols)
            df = df.merge(valid_df, on=groupby_cols, how="inner")

        filtered_count = original_len - len(df)
        logger.info(
            f"Filtered {filtered_count} rows with < {min_replicates} replicates, "
            f"kept {len(df)} rows"
        )
        return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run copairs analysis")
    parser.add_argument("config", help="Path to config file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    results = CopairsRunner(args.config).run()
    print(f"Analysis complete. Results shape: {results.shape}")
