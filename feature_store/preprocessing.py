from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from .core import FeatureStore
from sklearn.preprocessing import PowerTransformer

def apply_transformation(df: pd.DataFrame, columns: pd.DataFrame, transform_fn) -> pd.DataFrame:
    """
    Helper function for reducing duplicate code that merges transformed
    dataframes back together.
    """
    transformed = transform_fn(columns)
    transformed_df = pd.DataFrame(transformed, columns=columns.columns, index=df.index)
    df = df.copy()
    df.loc[:, columns.columns] = transformed_df
    return df


def std(df: pd.DataFrame, feature_store: FeatureStore):
    numerical_columns = feature_store.get_numerical_cols()
    scaler = StandardScaler()
    return apply_transformation(df, numerical_columns, scaler.fit_transform)

# TODO: fill other values than numbers too?
def fill_na(df: pd.DataFrame, feature_store: FeatureStore, strategy="mean"):
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

def one_hot_encode(df: pd.DataFrame, feature_store: FeatureStore):
    df = df.copy()
    columns = feature_store.get_categorical_cols()
    encoder = OneHotEncoder()
    transformed = encoder.fit_transform(df[columns])
    encoded_df = pd.DataFrame(transformed.toarray(), columns=encoder.get_feature_names_out(), index=df.index)
    df = df.drop(columns, axis=1)
    df = pd.concat([df, encoded_df], axis=1)
    return df

def skew(df: pd.DataFrame, feature_store: FeatureStore):
    numerical_columns = feature_store.get_numerical_cols()
    pt = PowerTransformer()
    return apply_transformation(df, numerical_columns, pt.fit_transform)