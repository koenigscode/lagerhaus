import pandas as pd

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
