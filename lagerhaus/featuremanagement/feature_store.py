from .feature_metadata import FeatureMetadata
import pandas as pd

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

    def get_numerical_cols(self) -> list[str]:
        return self.df.select_dtypes(include = ['number']).columns
    def get_categorical_cols(self) -> list[str]:
        return [col for col, metadata in self.metadata.items() if metadata.categorical]