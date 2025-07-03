# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "pandas",
#     "numpy",
#     "copairs",
#     "pyyaml",
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
    - Saving results
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
        """Load configuration from YAML or JSON file."""
        config_path = Path(config_path)

        with open(config_path, "r") as f:
            if config_path.suffix in [".yaml", ".yml"]:
                return yaml.safe_load(f)
            elif config_path.suffix == ".json":
                return json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")

    def validate_config(self):
        """Validate configuration has required fields."""
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

    def load_data(self) -> pd.DataFrame:
        """Load data from configured path."""
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
        """Apply preprocessing steps to data."""
        if "preprocessing" not in self.config:
            return df

        for step in self.config["preprocessing"]:
            step_type = step["type"]
            logger.info(f"Applying preprocessing: {step_type}")

            if step_type == "filter":
                query = step["query"]
                df = df.query(query)
                logger.info(f"After filter '{query}': {len(df)} rows")

            elif step_type == "assign_reference":
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

            else:
                logger.warning(f"Unknown preprocessing type: {step_type}")

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get feature columns from dataframe."""
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
        """Extract metadata and feature data."""
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
        """Run average precision calculation."""
        ap_config = self.config["average_precision"]
        params = ap_config["params"].copy()

        # Check if multilabel
        if ap_config.get("multilabel", False):
            logger.info("Running multilabel average precision")
            results = map.multilabel.average_precision(metadata, features, **params)
        else:
            logger.info("Running average precision")
            results = map.average_precision(metadata, features, **params)

        return results

    def run_mean_average_precision(self, ap_results: pd.DataFrame) -> pd.DataFrame:
        """Run mean average precision if configured."""
        if "mean_average_precision" not in self.config:
            return ap_results

        map_config = self.config["mean_average_precision"]
        logger.info("Running mean average precision")

        map_results = map.mean_average_precision(ap_results, **map_config)

        # Add -log10(p-value) column if not present
        if "corrected_p_value" in map_results.columns:
            map_results["-log10(p-value)"] = -map_results["corrected_p_value"].apply(
                np.log10
            )

        return map_results

    def save_results(self, results: pd.DataFrame, suffix: str = ""):
        """Save results to configured output path."""
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

    def run(self) -> pd.DataFrame:
        """Run the complete analysis pipeline."""
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

        # 7. Save final results
        self.save_results(final_results)

        logger.info("Analysis complete")
        return final_results


def run_copairs_analysis(
    config: Union[Dict[str, Any], str, Path], **kwargs
) -> pd.DataFrame:
    """Run copairs analysis - conveniece function.

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
