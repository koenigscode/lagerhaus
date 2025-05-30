from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import IsolationForest
from .helpers import apply_transformation
from lagerhaus.featuremanagement import FeatureStore, FeatureView
import numpy as np

def std():
    def transform(df: pd.DataFrame, feature_view: FeatureView):
        numerical_columns = df[feature_view.get_numerical_cols()]
        if(numerical_columns.empty):
            return df

        scaler = StandardScaler()
        return apply_transformation(df, numerical_columns, scaler.fit_transform)
    return transform

# TODO: fill other values than numbers too?
def fill_na(strategy="mean"):
    def transform(df: pd.DataFrame, feature_view: FeatureView):
        df = df.copy()
        numerical_columns = df[feature_view.get_numerical_cols()]
        if(numerical_columns.empty):
            return df

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
    def transform(df: pd.DataFrame, feature_view: FeatureView):
        df = df.copy()
        categorical_columns = df[feature_view.get_categorical_cols()]
        if(categorical_columns.empty):
            return df

        encoder = OneHotEncoder()
        transformed = encoder.fit_transform(categorical_columns)
        encoded_df = pd.DataFrame(transformed.toarray(), columns=encoder.get_feature_names_out(), index=df.index)
        df = df.drop(categorical_columns.columns, axis=1)
        df = pd.concat([df, encoded_df], axis=1)
        return df
    return transform

def skew():
    def transform(df: pd.DataFrame, feature_view: FeatureView):
        numerical_columns = df[feature_view.get_numerical_cols()]
        if(numerical_columns.empty):
            return df

        pt = PowerTransformer(standardize=False) # use std() if needed
        return apply_transformation(df, numerical_columns, pt.fit_transform)
    return transform

def remove_correlated_features():
    def transform(df: pd.DataFrame, feature_view: FeatureView):
        numerical_columns = df[feature_view.get_numerical_cols()]
        if(numerical_columns.empty):
            return df

        corr_matrix = numerical_columns.corr().abs()

        # gets rid of duplicates, since the matrix has every correlation twice
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        threshold = 0.9
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        return df.drop(columns=to_drop)
    return transform

def remove_outliers(contamination=0.05):
    def transform(df: pd.DataFrame, feature_view: FeatureView):
        df = df.copy()
        numerical_cols = df[feature_view.get_numerical_cols()]
        if(numerical_cols.empty):
            return df
        
        iso_forest = IsolationForest(contamination=contamination)
        preds = iso_forest.fit_predict(numerical_cols)
        return df.loc[preds == 1]
    return transform
    

def drop_columns(columns: list[str], reset_index=True):
    def transform(df: pd.DataFrame, feature_view: FeatureView):
        df = df.copy()
        df = df.drop(columns=columns)
        if reset_index:
            df = df.reset_index(drop=True)
        return df
    return transform 