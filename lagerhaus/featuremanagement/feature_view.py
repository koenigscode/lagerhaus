from lagerhaus.featuremanagement import FeatureStore

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
        df = self.get_all_raw()

        for transformer in self.transformers:
            print("before")
            print(df.max())
            df = transformer(df, feature_view=self)
            print("after")
            print(df.max())

        return df
    
    def get_all_raw(self):
        """
        Returns the DataFrame without applying transformations, but
        applies the whitelist
        """
        df = self.feature_store.get_all()
        if self.whitelist is not None:
            df = df[self.whitelist]
        return df
        
    def get_numerical_cols(self) -> list[str]:
        return self.get_all_raw().select_dtypes(include = ['number']).columns

    def get_categorical_cols(self) -> list[str]:
        df = [col for col, metadata in self.feature_store.metadata.items() if metadata.categorical]

        if self.whitelist is not None:
            df = [col for col in df if col in self.whitelist]
        return df
