#!/bin/bash
# Run copairs examples using the runner

# Create data directory if it doesn't exist
mkdir -p data

# Download the data file if it doesn't exist
DATA_FILE="data/2016_04_01_a549_48hr_batch1_plateSQ00014812.csv"
if [ ! -f "$DATA_FILE" ]; then
    echo "Downloading data file..."
    COMMIT="da8ae6a3bc103346095d61b4ee02f08fc85a5d98"
    PLATE="SQ00014812"
    URL="https://media.githubusercontent.com/media/broadinstitute/lincs-cell-painting/${COMMIT}/profiles/2016_04_01_a549_48hr_batch1/${PLATE}/${PLATE}_normalized_feature_select.csv.gz"
    
    # Download and decompress
    wget -O "${DATA_FILE}.gz" "$URL"
    gunzip "${DATA_FILE}.gz"
    echo "Data file downloaded successfully!"
else
    echo "Data file already exists, skipping download."
fi

echo -e "\nRunning phenotypic activity analysis..."
uv run copairs_runner.py configs/activity_analysis.yaml --verbose

echo -e "\nRunning phenotypic consistency analysis..."
# Note: The consistency analysis requires the activity results to filter active compounds
# In the notebook, this is done by loading activity_map.csv
# For the runner, you may need to modify the consistency config or the runner
# to support filtering based on previous results
uv run copairs_runner.py configs/consistency_analysis.yaml --verbose

echo -e "\nAnalyses complete!"
echo "Results saved to:"
echo "  - data/activity_map_runner.csv"
echo "  - data/target_maps_runner.csv"