from pydantic import BaseModel
import pandas as pd


class FeatureMetadata(BaseModel):
    description: str
    categorical: bool = False


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
    """
    View into a FeatureStore allowing transformations and whitelists.
    
    Parameters:
        feature_store: The FeatureStore to use.
        whitelist: A list of column names to include in the view.
                   If empty, include all columns from the FeatureStore.
        transformers: A list of functions to apply to the features.
    """
    feature_store: FeatureStore
    whitelist: list[str] = None
    transformers = []

    def __init__(self, feature_store: FeatureStore, whitelist: list[str] = None, transformers: [] = []) -> None:
        self.whitelist = whitelist
        self.feature_store = feature_store
        self.transformers = transformers
    
    def get_all(self):
        """
        Returns the feature view as a DataFrame with all the transformations applied.
        """
        df = self.feature_store.get_all()
        if self.whitelist is not None:
            df = df[self.whitelist]

        for transformer in self.transformers:
            df = transformer(df, feature_store=self.feature_store)

        return df