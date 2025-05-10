from butterfree.transform.transformations import CustomTransform
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, mean as _mean, stddev as _stddev
from pyspark.sql.types import NumericType, DoubleType

def _std_spark(df: DataFrame, parent_feature=None, id_column=None) -> DataFrame:
    # 1) discover all numeric columns in the schema
    numeric_cols = [
        f.name
        for f in df.schema.fields
        if isinstance(f.dataType, NumericType)
    ]
    # 2) drop the id column if it showed up in numeric
    if id_column in numeric_cols:
        numeric_cols.remove(id_column)

    if not numeric_cols:
        return df

    # 3) compute mean and stddev for each numeric column in one pass
    agg_exprs = []
    for c in numeric_cols:
        agg_exprs.append(_mean(c).alias(f"{c}__mean"))
        agg_exprs.append(_stddev(c).alias(f"{c}__stddev"))

    stats = df.agg(*agg_exprs).collect()[0]

    # 4) for each numeric column, subtract mean and divide by stddev
    out = df
    for c in numeric_cols:
        m = stats[f"{c}__mean"]
        s = stats[f"{c}__stddev"] or 1.0
        out = out.withColumn(
            c,
            ((col(c) - m) / s).cast(DoubleType())
        )

    return out

# wrap as a Butterfree CustomTransform
std = lambda id_column: CustomTransform(transformer=_std_spark, id_column=id_column)

def _fillna_mean(df: DataFrame, parent_feature=None, id_column=None) -> DataFrame:
    # 1) find all numeric columns
    numeric_cols = [
        f.name
        for f in df.schema.fields
        if isinstance(f.dataType, NumericType)
    ]
    # 2) drop id column if numeric
    if id_column in numeric_cols:
        numeric_cols.remove(id_column)

    if not numeric_cols:
        return df

    # 3) compute means
    agg_exprs = [_mean(c).alias(c) for c in numeric_cols]
    means = df.agg(*agg_exprs).collect()[0].asDict()

    # 4) fill nulls with corresponding mean
    return df.fillna(means)

# wrap as a Butterfree CustomTransform
fillna_mean = lambda id_column: CustomTransform(transformer=_fillna_mean, id_column=id_column)