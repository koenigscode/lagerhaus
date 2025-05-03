from butterfree.extract import Source
from butterfree.extract.readers import FileReader
from butterfree.transform.features import Feature, KeyFeature, TimestampFeature
from butterfree.transform import FeatureSet
from butterfree.transform.transformations.h3_transform import H3HashTransform
from butterfree.clients import SparkClient
from butterfree.constants import DataType

spark_client = SparkClient()

file_reader = FileReader(
                id="data1",
                path="./data.csv",
                format="csv",
                format_options={"header": "true"}
            )

source = Source(readers=[file_reader], query="select * from data1")
source_df = source.construct(spark_client)
print("Source")
source_df.show()

keys = [
    KeyFeature(
        name="id",
        description="Unique identificator",
        dtype=DataType.BIGINT,
    )
]

ts_feature = TimestampFeature(from_ms=True)

features = [
    Feature(
        name="col1",
        description="First column",
        dtype=DataType.STRING,
    ),
    Feature(
        name="col2",
        description="Second column",
        dtype=DataType.STRING,
    ),
]

feature_set = FeatureSet(
    name="testing",
    entity="testentity",  
    description="test description",
    keys=keys,
    timestamp=ts_feature,
    features=features,
)

feature_set_df = feature_set.construct(source_df, spark_client)
print(feature_set_df.toPandas())