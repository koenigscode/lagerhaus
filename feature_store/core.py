from pydantic import BaseModel
import pandas as pd
from .preprocessing import std


class ColumnMetadata(BaseModel):
    description: str
    unique: bool = False

class ColumnMetadataArgument(ColumnMetadata):
    name: str  
    version: int = 1


class FeatureStore:
    metadata: dict[str, dict[int, ColumnMetadata]]
    df: pd.DataFrame

    def __init__(self, metadata: list[ColumnMetadataArgument], df: pd.DataFrame):
        self.metadata = {column.name: {column.version: ColumnMetadata(description=column.description, unique=column.unique) } for column in metadata }
        

        missing_cols = set(self.metadata.keys()) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Column name(s) not found in DataFrame: {missing_cols}")
        
        self.df = df[self.metadata.keys()]

    def get_all(self):
        return self.df


class FeatureView:
    feature_store: FeatureStore
    model: str = None

    model_transformers = {"LinearRegression": [std]}

    def __init__(self, feature_store, model) -> None:
        if model not in self.model_transformers.keys() and model != None:
            raise ValueError(f"Model '{model} is not supported")
        
        self.feature_store = feature_store
        self.model = model
    
    def get_all(self):
        df = self.feature_store.get_all()
        for transformer in self.model_transformers[self.model]:
            df = transformer(df)

        return df