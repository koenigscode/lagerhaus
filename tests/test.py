import pandas as pd
from lagerhaus import stats
from lagerhaus.featuremanagement import FeatureStore, FeatureView, FeatureMetadata
from lagerhaus.datacleaning.preprocessing import *
from lagerhaus.datacleaning import presets
import numpy as np

# df = pd.read_csv("./tests/datasets/small_dataset.csv")
df = pd.read_csv("./tests/datasets/student_habits_performance.csv")

metadata = {
    "id": FeatureMetadata(description="Student ID", from_col="student_id", dtype=str),
    "age": FeatureMetadata(description="Student Age", dtype=np.float64),
    "gender": FeatureMetadata(description="Student Gender", categorical=True, dtype=str),
    "exam_score": FeatureMetadata(description="Exam Score", dtype=np.float64),
}

fs = FeatureStore(metadata=metadata, df=df)

linear_fv = FeatureView(feature_store=fs, transformers=presets.linear_regression)
decision_tree_fv = FeatureView(feature_store=fs, transformers=presets.decision_tree)

stats.init("Student Habits Performance", fs)
stats.print(fs, title="Feature Store")
stats.print(linear_fv, title="Feature View for Linear Regression")
stats.print(decision_tree_fv, title="Feature View for Decision Tree")

# stats.plot_distribution(fs.get_all(), "age", title="Age distribution in Feature Store")
# stats.plot_distribution(linear_fv.get_all(), "age", title="Age distribution in Feature View for Linear Regression")
# stats.plot_distribution(fs.get_all(), "exam_score", title="Exam Score distribution in Feature Store",)
# stats.plot_distribution(linear_fv.get_all(), "exam_score", title="Exam Score distribution in Feature View for Linear Regression",)

