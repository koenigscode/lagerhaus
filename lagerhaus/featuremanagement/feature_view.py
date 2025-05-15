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
        df = self.feature_store.get_all()
        if self.whitelist is not None:
            df = df[self.whitelist]

        for transformer in self.transformers:
            df = transformer(df, feature_store=self.feature_store)

        return df