import pandas as pd
from feature_store import std, FeatureStore, ColumnMetadataArgument
from feature_store.core import FeatureView

df = pd.read_csv("./student_habits_performance.csv")

metadata = [
    ColumnMetadataArgument(name="student_id", version=1, description="Student ID"),
    ColumnMetadataArgument(name="age", version=1, description="Student Age"),
]

fs = FeatureStore(metadata=metadata, df=df)
linear_fv = FeatureView(feature_store=fs, model="LinearRegression")
print(fs.get_all())
print(linear_fv.get_all())
