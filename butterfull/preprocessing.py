from butterfree.transform.transformations import CustomTransform
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pyspark.sql import SparkSession

def std(df, parent_feature=None):
    # If df is a Spark DataFrame, convert to pandas
    if not isinstance(df, pd.DataFrame):
        df = df.toPandas()
    
    numeric_cols = df.select_dtypes(include='number').columns
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[numeric_cols])

    df_scaled = pd.DataFrame(scaled_values, columns=numeric_cols, index=df.index)
    non_numeric = df.drop(columns=numeric_cols)

    spark = SparkSession.builder.getOrCreate()
    return spark.createDataFrame(pd.concat([non_numeric, df_scaled], axis=1))

NaiveBayes = CustomTransform(transformer= std, )