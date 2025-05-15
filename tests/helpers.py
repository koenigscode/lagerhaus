import pandas as pd
from lagerhaus.featuremanagement import FeatureStore, FeatureView, FeatureMetadata
import numpy as np

def load_data_and_metadata():
    df = pd.read_csv("./tests/datasets/small_dataset.csv")
    metadata = {
        "id": FeatureMetadata(description="Student ID", from_col="student_id", dtype=str),
        "age": FeatureMetadata(description="Student Age", dtype=np.float64),
        "gender": FeatureMetadata(description="Student Gender", categorical=True, dtype=str),
        "exam_score": FeatureMetadata(description="Exam Score", dtype=np.float64),
    }
    fs = FeatureStore(metadata=metadata, df=df)
    fv = FeatureView(feature_store=fs)
    df = fv.get_all_raw()  # keep only metadata columns
    return df, fs, fv