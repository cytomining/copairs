"""Keep opt-in research tests out of the ordinary production test run."""

import os

collect_ignore = (
    []
    if os.environ.get("COPAIRS_STUDY_SOURCE")
    else ["test_study.py", "test_metadata_oracle.py"]
)
