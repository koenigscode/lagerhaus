from pydantic import BaseModel
import pandas as pd
from .preprocessing import std, fill_na
from enum import Enum

class ModelType(Enum):
    LINEAR_REGRESSION = "LinearRegression"
    DECISION_TREE = "DecisionTree"

class FeatureMetadata(BaseModel):
    description: str
    unique: bool = False


class FeatureStore:
    metadata: dict[str, FeatureMetadata]
    df: pd.DataFrame

    def __init__(self, metadata: dict[str, FeatureMetadata], df: pd.DataFrame):
        self.metadata = metadata

        missing_cols = set(self.metadata.keys()) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Column name(s) not found in DataFrame: {missing_cols}")
        
        self.df = df[self.metadata.keys()]

    def get_all(self):
        return self.df


class FeatureView:
    feature_store: FeatureStore
    model: ModelType = None

    model_transformers = {ModelType.LINEAR_REGRESSION: [fill_na, std,], ModelType.DECISION_TREE: []}

    def __init__(self, feature_store, model) -> None:
        if model not in self.model_transformers.keys() and model is not None:
            raise ValueError(f"Model '{model}' is not supported")
        
        self.feature_store = feature_store
        self.model = model
    
    def get_all(self):
        """
        Returns the feature view as a DataFrame with all the transformations applied.
        """
        df = self.feature_store.get_all()
        for transformer in self.model_transformers[self.model]:
            df = transformer(df)

        return df