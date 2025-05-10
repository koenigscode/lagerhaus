from sklearn.preprocessing import StandardScaler
import pandas as pd

def std(df):
    numerical_columns = df.select_dtypes(include = ['number'])
    scaler = StandardScaler()
    scaler.fit(numerical_columns)
    scaled_array = scaler.fit_transform(numerical_columns)
    numerical_columns_df = pd.DataFrame(scaled_array, columns=numerical_columns.columns, index=numerical_columns.index)
    df.update(numerical_columns_df)
    return df
