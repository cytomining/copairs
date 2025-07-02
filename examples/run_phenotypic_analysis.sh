#!/bin/bash
# Example usage of the unified phenotypic analysis script

# Check if data exists, download if not
DATA_FILE="data/2016_04_01_a549_48hr_batch1_plateSQ00014812.csv"
if [ ! -f "$DATA_FILE" ]; then
    echo "Downloading example data..."
    mkdir -p data
    curl -L -o "${DATA_FILE}.gz" "https://media.githubusercontent.com/media/broadinstitute/lincs-cell-painting/da8ae6a3bc103346095d61b4ee02f08fc85a5d98/profiles/2016_04_01_a549_48hr_batch1/SQ00014812/SQ00014812_normalized_feature_select.csv.gz"
    gunzip "${DATA_FILE}.gz"
    echo "Data downloaded successfully."
fi

# Run phenotypic activity analysis
echo "Running phenotypic activity analysis..."
uv run phenotypic_analysis.py activity \
    "$DATA_FILE" \
    --output-dir results/activity \
    --plot

# Run phenotypic consistency analysis
echo "Running phenotypic consistency analysis..."
uv run phenotypic_analysis.py consistency \
    "$DATA_FILE" \
    --output-dir results/consistency \
    --activity-results results/activity/activity_map.csv \
    --aggregate-replicates \
    --plot

echo "Analysis complete! Check the results/ directory for outputs."