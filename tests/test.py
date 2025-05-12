import pandas as pd
from feature_store.core import FeatureStore, FeatureView, FeatureMetadata, ModelType

df = pd.read_csv("./small_dataset.csv")

metadata = {
    "student_id": FeatureMetadata(description="Student ID"),
    "age": FeatureMetadata(description="Student Age"),
}

fs = FeatureStore(metadata=metadata, df=df)
linear_fv = FeatureView(feature_store=fs, model=ModelType.LINEAR_REGRESSION)
decision_tree_fv = FeatureView(feature_store=fs, model=ModelType.DECISION_TREE)

print("Feature Store:")
print(fs.get_all())

print("Linear Regression Feature View:")
print(linear_fv.get_all())

print("Decision Tree Feature View:")
print(decision_tree_fv.get_all())
