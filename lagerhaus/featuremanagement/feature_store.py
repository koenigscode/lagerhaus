from .feature_metadata import FeatureMetadata
import pandas as pd

class FeatureStore:
    metadata: dict[str, FeatureMetadata]
    df: pd.DataFrame

    def __init__(self, metadata: dict[str, FeatureMetadata], df: pd.DataFrame):
        self.metadata = metadata

        for col, met in self.metadata.items():
            if met.from_col is not None:
                df[col] = df[met.from_col]
                df.drop(met.from_col, axis=1, inplace=True)

            if met.dtype is not None:
                df[col] = df[col].astype(met.dtype)

        missing_cols = set(self.metadata.keys()) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Column name(s) not found in DataFrame: {missing_cols}")
        
        self.df = df[self.metadata.keys()]

    def get_all(self):
        return self.df
