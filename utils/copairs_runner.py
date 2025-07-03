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

import json
import logging
from typing import Any, Dict, List, Union, Optional
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
    - Preprocessing steps (filtering, reference assignment)
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
            Configuration dictionary or path to YAML/JSON config file
        """
        if isinstance(config, (str, Path)):
            config = self.load_config(config)
        self.config = config
        self.validate_config()

    @staticmethod
    def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file.

        Parameters
        ----------
        config_path : str or Path
            Path to YAML or JSON configuration file

        Returns
        -------
        dict
            Configuration dictionary
        """
        config_path = Path(config_path)

        with open(config_path, "r") as f:
            if config_path.suffix in [".yaml", ".yml"]:
                return yaml.safe_load(f)
            elif config_path.suffix == ".json":
                return json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")

    def validate_config(self):
        """Validate configuration has required fields.

        Raises
        ------
        ValueError
            If required configuration fields are missing
        """
        required = ["data", "average_precision", "output"]
        for field in required:
            if field not in self.config:
                raise ValueError(f"Missing required config field: {field}")

        # Validate average_precision params
        ap_params = self.config["average_precision"].get("params", {})
        required_ap = ["pos_sameby", "pos_diffby", "neg_sameby", "neg_diffby"]
        for field in required_ap:
            if field not in ap_params:
                raise ValueError(f"Missing required average_precision param: {field}")

        # Validate mean_average_precision params if present
        if "mean_average_precision" in self.config:
            map_params = self.config["mean_average_precision"].get("params", {})
            required_map = ["sameby", "null_size", "threshold", "seed"]
            for field in required_map:
                if field not in map_params:
                    raise ValueError(
                        f"Missing required mean_average_precision param: {field}"
                    )

        # Validate plotting params if present
        if "plotting" in self.config and self.config["plotting"].get("enabled", False):
            plot_config = self.config["plotting"]
            if "mean_average_precision" not in self.config:
                logger.warning(
                    "Plotting is enabled but mean_average_precision is not configured. "
                    "No plots will be generated."
                )

    def load_data(self) -> pd.DataFrame:
        """Load data from configured path.

        Returns
        -------
        pd.DataFrame
            Loaded data

        Raises
        ------
        ValueError
            If file format is not supported
        """
        data_config = self.config["data"]
        path = Path(data_config["path"])

        logger.info(f"Loading data from {path}")

        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix in [".csv", ".gz"]:
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        return df

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
        Available preprocessing steps:

        Direct parameter steps (parameters at step level):
        - filter: Filter rows using pandas query
        - dropna: Drop rows with NaN values
        - remove_nan_features: Remove feature columns containing NaN
        - split_multilabel: Split pipe-separated values into lists
        - filter_by_external_csv: Filter based on external CSV file
        - aggregate_replicates: Aggregate by taking median of features
        - add_column_from_query: Add column from pandas eval expression (optional: fill_value)

        External function steps (parameters under 'params'):
        - apply_assign_reference: Apply copairs.matching.assign_reference_index

        The 'apply_' prefix indicates steps that call external functions
        and require parameters to be nested under 'params'.
        """
        if "preprocessing" not in self.config:
            return df

        for step in self.config["preprocessing"]:
            step_type = step["type"]
            logger.info(f"Applying preprocessing: {step_type}")

            if step_type == "filter":
                query = step["query"]
                df = df.query(query)
                logger.info(f"After filter '{query}': {len(df)} rows")

            elif step_type == "apply_assign_reference":
                params = step["params"]
                df = assign_reference_index(df, **params)

            elif step_type == "dropna":
                columns = step.get("columns")
                df = df.dropna(subset=columns)
                logger.info(f"After dropna: {len(df)} rows")

            elif step_type == "remove_nan_features":
                # Remove features with any NaN values
                feature_cols = self.get_feature_columns(df)
                nan_cols = df[feature_cols].isna().any()
                nan_cols = nan_cols[nan_cols].index.tolist()
                if nan_cols:
                    df = df.drop(columns=nan_cols)
                    logger.info(f"Removed {len(nan_cols)} features with NaN values")

            elif step_type == "split_multilabel":
                # Split pipe-separated values into lists
                column = step["column"]
                separator = step.get("separator", "|")
                df[column] = df[column].str.split(separator)
                logger.info(f"Split multilabel column '{column}' by '{separator}'")

            elif step_type == "filter_by_external_csv":
                # Filter data based on values from an external CSV file
                csv_path = Path(step["csv_path"])
                filter_column = step["filter_column"]
                csv_column = step.get("csv_column", filter_column)
                condition = step.get("condition", "below_corrected_p")

                # Load the external CSV
                external_df = pd.read_csv(csv_path)

                # Get values that meet the condition
                if condition in external_df.columns:
                    # Boolean column filter
                    valid_values = external_df[external_df[condition]][
                        csv_column
                    ].tolist()
                else:
                    # Use all values if condition column doesn't exist
                    valid_values = external_df[csv_column].tolist()

                # Filter the main dataframe
                df = df[df[filter_column].isin(valid_values)]
                logger.info(
                    f"Filtered to {len(df)} rows based on {len(valid_values)} values from {csv_path}"
                )

            elif step_type == "aggregate_replicates":
                # Aggregate replicates by taking median of features
                groupby_cols = step["groupby"]
                feature_cols = self.get_feature_columns(df)

                # Keep only groupby columns and features (matching notebook behavior)
                keep_cols = groupby_cols + feature_cols
                df = df[keep_cols]

                # Group and aggregate only feature columns
                df = df.groupby(groupby_cols, as_index=False)[feature_cols].median()

                logger.info(
                    f"Aggregated to {len(df)} rows by grouping on {groupby_cols}"
                )

            elif step_type == "add_column_from_query":
                # Add a new column based on evaluating a query expression
                query = step["query"]
                # Use provided column name or default to the query itself
                column_name = step.get("column_name", query)
                df[column_name] = df.eval(query)

                # Handle NaN values if fill_value is specified
                nan_count = df[column_name].isna().sum()
                if "fill_value" in step and nan_count > 0:
                    fill_value = step["fill_value"]
                    df[column_name] = df[column_name].fillna(fill_value)
                    logger.info(f"Filled {nan_count} NaN values with {fill_value}")

                # Log the result
                nan_info = (
                    f" ({nan_count} NaN values)"
                    if nan_count > 0 and "fill_value" not in step
                    else ""
                )
                logger.info(
                    f"Added column '{column_name}' (dtype: {df[column_name].dtype}){nan_info}"
                )

            else:
                logger.warning(f"Unknown preprocessing type: {step_type}")

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get feature columns from dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe

        Returns
        -------
        list of str
            Feature column names
        """
        data_config = self.config["data"]

        if "feature_columns" in data_config:
            # Explicit list of columns
            return data_config["feature_columns"]
        elif "feature_regex" in data_config:
            # Regex pattern for features
            return df.filter(regex=data_config["feature_regex"]).columns.tolist()
        else:
            # Default: all non-metadata columns
            metadata_regex = data_config.get("metadata_regex", "^Metadata")
            metadata_cols = df.filter(regex=metadata_regex).columns
            return [col for col in df.columns if col not in metadata_cols]

    def extract_data(self, df: pd.DataFrame) -> tuple:
        """Extract metadata and feature data.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe

        Returns
        -------
        tuple
            (metadata, features) where metadata is a DataFrame and features is a numpy array
        """
        data_config = self.config["data"]

        # Get metadata
        metadata_regex = data_config.get("metadata_regex", "^Metadata")
        metadata = df.filter(regex=metadata_regex)

        # Get features
        feature_cols = self.get_feature_columns(df)
        features = df[feature_cols].values

        logger.info(
            f"Extracted {metadata.shape[1]} metadata columns and {features.shape[1]} features"
        )

        return metadata, features

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

        # Create directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save based on file extension
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

    def run(self) -> pd.DataFrame:
        """Run the complete analysis pipeline.

        Returns
        -------
        pd.DataFrame
            Final analysis results
        """
        logger.info("Starting copairs analysis")

        # 1. Load data
        df = self.load_data()

        # 2. Preprocess
        df = self.preprocess_data(df)

        # 3. Extract metadata and features
        metadata, features = self.extract_data(df)

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


def run_copairs_analysis(
    config: Union[Dict[str, Any], str, Path], **kwargs
) -> pd.DataFrame:
    """Run copairs analysis - convenience function.

    Parameters
    ----------
    config : dict, str, or Path
        Configuration dictionary or path to config file
    **kwargs
        Additional parameters to override config values

    Returns
    -------
    pd.DataFrame
        Analysis results
    """
    # Load config if path provided
    if isinstance(config, (str, Path)):
        config = CopairsRunner.load_config(config)

    # Apply overrides
    if kwargs:
        import copy

        config = copy.deepcopy(config)
        for key, value in kwargs.items():
            if "." in key:
                # Handle nested keys like 'average_precision.params.batch_size'
                parts = key.split(".")
                current = config
                for part in parts[:-1]:
                    current = current.setdefault(part, {})
                current[parts[-1]] = value
            else:
                config[key] = value

    # Run analysis
    runner = CopairsRunner(config)
    return runner.run()


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Run copairs analysis")
    parser.add_argument("config", help="Path to config file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    results = run_copairs_analysis(args.config)
    print(f"Analysis complete. Results shape: {results.shape}")
