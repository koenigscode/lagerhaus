import pandas as pd
import numpy as np
from lagerhaus.featuremanagement import FeatureStore, FeatureMetadata
from lagerhaus.featuremanagement import FeatureView
from lagerhaus.datacleaning.preprocessing import fill_na, std
from .helpers import load_data_and_metadata


def test_whitelist():
    df = pd.read_csv("./tests/datasets/small_dataset.csv")
    metadata = {
        "id": FeatureMetadata(description="Student ID", from_col="student_id", dtype=str),
        "age": FeatureMetadata(description="Student Age", dtype=np.float64),
        "gender": FeatureMetadata(description="Student Gender", categorical=True, dtype=str),
        "exam_score": FeatureMetadata(description="Exam Score", dtype=np.float64),
    }
    fs = FeatureStore(metadata=metadata, df=df)
    fv = FeatureView(feature_store=fs, transformers=[], whitelist=["id", "age"])
    assert len(fv.get_all().columns) == 2

def test_featurize():
    df, fs, _ = load_data_and_metadata()
    fv = FeatureView(feature_store=fs, transformers=[fill_na(), std()], whitelist=["id", "age"])
    featurized = fv.featurize(pd.DataFrame({"student_id": ['s1', 's2', 's3'], "age": [20, 25, 30]}))

    assert len(featurized.columns) == 2
    assert len(featurized) == 3
    assert "id" in featurized.columns
    assert "student_id" not in featurized.columns
    assert featurized["age"].mean() == 0
    
    
