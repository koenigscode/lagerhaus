from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from .core import FeatureStore

def _get_numerical(df):
    return df.select_dtypes(include = ['number'])

def std(feature_store: FeatureStore):
    df = feature_store.get_all().copy()
    numerical_columns = _get_numerical(df)
    scaler = StandardScaler()
    scaler.fit(numerical_columns)
    scaled_array = scaler.fit_transform(numerical_columns)
    numerical_columns_df = pd.DataFrame(scaled_array, columns=numerical_columns.columns, index=numerical_columns.index)
    df.loc[:, numerical_columns.columns] = numerical_columns_df
    return df

# TODO: fill other values than numbers too?
def fill_na(feature_store: FeatureStore, strategy="mean"):
    df = feature_store.get_all().copy()
    numerical_columns = _get_numerical(df)

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

def one_hot_encode(feature_store: FeatureStore):
    columns = [col for col, metadata in feature_store.metadata.items() if metadata.categorical]
    df = feature_store.get_all().copy()
    encoder = OneHotEncoder()
    transformed = encoder.fit_transform(df[columns])
    encoded_df = pd.DataFrame(transformed.toarray(), columns=encoder.get_feature_names_out(), index=df.index)
    df = df.drop(columns, axis=1)
    df = pd.concat([df, encoded_df], axis=1)
    return df