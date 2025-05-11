import pandas as pd
from feature_store.core import FeatureStore, FeatureView, FeatureMetadata

df = pd.read_csv("./student_habits_performance.csv")

metadata = {
    "student_id": FeatureMetadata(description="Student ID"),
    "age": FeatureMetadata(description="Student Age"),
}

fs = FeatureStore(metadata=metadata, df=df)
linear_fv = FeatureView(feature_store=fs, model="LinearRegression")
print(fs.get_all())
print(linear_fv.get_all())
