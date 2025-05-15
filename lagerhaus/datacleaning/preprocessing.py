from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import IsolationForest
from .helpers import apply_transformation
from lagerhaus.featuremanagement import FeatureStore

def std():
    def transform(df: pd.DataFrame, feature_store: FeatureStore):
        numerical_columns = feature_store.get_numerical_cols()
        scaler = StandardScaler()
        return apply_transformation(df, numerical_columns, scaler.fit_transform)
    return transform

# TODO: fill other values than numbers too?
def fill_na(strategy="mean"):
    def transform(df: pd.DataFrame, feature_store: FeatureStore):
        df = df.copy()
        numerical_columns = feature_store.get_numerical_cols()

        if strategy == "mean":
            num_df = numerical_columns.fillna(numerical_columns.mean())
        elif strategy == "median":
            num_df = numerical_columns.fillna(numerical_columns.median())
        elif strategy == "mode":
            num_df = numerical_columns.fillna(numerical_columns.mode())
        else:
            raise ValueError(f"Invalid strategy: {strategy}")
        df.loc[:, numerical_columns.columns] = num_df
        return df
    return transform

def one_hot_encode():
    def transform(df: pd.DataFrame, feature_store: FeatureStore):
        df = df.copy()
        columns = feature_store.get_categorical_cols()
        encoder = OneHotEncoder()
        transformed = encoder.fit_transform(df[columns])
        encoded_df = pd.DataFrame(transformed.toarray(), columns=encoder.get_feature_names_out(), index=df.index)
        df = df.drop(columns, axis=1)
        df = pd.concat([df, encoded_df], axis=1)
        return df
    return transform

def skew():
    def transform(df: pd.DataFrame, feature_store: FeatureStore):
        numerical_columns = feature_store.get_numerical_cols()
        pt = PowerTransformer()
        return apply_transformation(df, numerical_columns, pt.fit_transform)
    return transform

def remove_correlated_features():
    def transform(df: pd.DataFrame, feature_store: FeatureStore):
        numerical_columns = feature_store.get_numerical_cols()
        corr_matrix = df[numerical_columns].corr().abs()

        # gets rid of duplicates, since the matrix has every correlation twice
        upper = corr_matrix.where(
            pd.np.triu(pd.np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        threshold = 0.9
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        return df.drop(columns=to_drop)
    return transform

def remove_outliers(contamination=0.05):
    def transform(df: pd.DataFrame, feature_store: FeatureStore):
        df = df.copy()
        numerical_cols = feature_store.get_numerical_cols() 
        
        iso_forest = IsolationForest(contamination=contamination)
        preds = iso_forest.fit_predict(df[numerical_cols])
        return df.loc[preds == 1]
    return transform
    