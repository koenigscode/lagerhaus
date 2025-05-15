import pandas as pd
import numpy as np
from lagerhaus.featuremanagement import FeatureStore, FeatureMetadata
from lagerhaus.featuremanagement import FeatureView
from lagerhaus.datacleaning import presets

def test_whitelist():
    df = pd.read_csv("./tests/datasets/small_dataset.csv")
    metadata = {
        "id": FeatureMetadata(description="Student ID", from_col="student_id", dtype=str),
        "age": FeatureMetadata(description="Student Age", dtype=np.float64),
        "gender": FeatureMetadata(description="Student Gender", categorical=True, dtype=str),
        "exam_score": FeatureMetadata(description="Exam Score", dtype=np.float64),
    }
    fs = FeatureStore(metadata=metadata, df=df)
    fv = FeatureView(feature_store=fs, transformers=presets.linear_regression, whitelist=["id", "age"])
    assert len(fv.get_all().columns) == 2
