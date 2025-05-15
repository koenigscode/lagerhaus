import pandas as pd
import numpy as np
from lagerhaus.featuremanagement import FeatureStore, FeatureMetadata, FeatureView
from lagerhaus.datacleaning.preprocessing import fill_na, remove_outliers, skew, std, remove_correlated_features, one_hot_encode

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
    return df, fv

def test_fill_na():
    df, fv = load_data_and_metadata()
    print(f"Missing values before filling: {df.isna().sum().sum()}")
    transformed = fill_na()(df, fv)
    assert transformed.isna().sum().sum() == 0

def test_remove_outliers():
    df, fv = load_data_and_metadata()
    transformed = remove_outliers()(df, fv)
    removed_count = len(df) - len(transformed)
    print(f"Removed {removed_count} outliers")
    assert removed_count > 0

def test_skew():
    df, fv = load_data_and_metadata()
    skew_before = df.select_dtypes(include=[np.float64]).skew().sum()
    transformed = skew()(df, fv)
    skew_after = transformed.select_dtypes(include=[np.float64]).skew().sum()
    print(f"Sum of skew reduced from {skew_before} to {skew_after}")
    assert skew_after < skew_before

def test_std():
    df, fv = load_data_and_metadata()
    std_before = df.select_dtypes(include=[np.float64]).std().mean()
    mean_before = df.select_dtypes(include=[np.float64]).mean().mean()

    transformed = std()(df, fv)

    std_after = transformed.select_dtypes(include=[np.float64]).std().mean()
    mean_after = transformed.select_dtypes(include=[np.float64]).mean().mean()

    print(f"Mean of std changed from {std_before} to {std_after}")
    print(f"Mean of mean changed from {mean_before} to {mean_after}")

    assert abs(mean_after) < abs(mean_before) # mean should be closer to 0
    assert abs(std_after - 1) < abs(std_before - 1) # std should be closer to 1

def test_remove_correlated_features():
    df, fv = load_data_and_metadata()
    # copy age column and add static value, should be detected as correlation
    df["age_copy"] = df["age"] + 0.5
    transformed = remove_correlated_features()(df, fv)
    removed_count = len(df.columns) - len(transformed.columns)
    print(f"Removed {removed_count} correlated features")
    assert removed_count > 0

def test_one_hot_encode():
    df, fv = load_data_and_metadata()
    transformed = one_hot_encode()(df, fv)
    # gender column should be dropped
    assert "gender" not in transformed.columns
    assert len(transformed.columns) > len(df.columns)
    