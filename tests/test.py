from butterfree.extract import Source
from butterfree.extract.readers import FileReader
from butterfree.transform.features import Feature, KeyFeature, TimestampFeature
from butterfree.transform import FeatureSet
from butterfree.clients import SparkClient
from butterfree.constants import DataType
from butterfull import std, fillna_mean
from pyspark.sql.functions import current_timestamp
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

spark_client = SparkClient()

schema = StructType([
  StructField("student_id", StringType(), nullable=False),
  StructField("age", IntegerType(), nullable=True),
  StructField("gender", StringType(), nullable=True),
  StructField("study_hours_per_day", DoubleType(), nullable=True),
])

file_reader = FileReader(
                id="student_habits",
                path="./student_habits_performance.csv",
                format="csv",
                format_options={"header": "true" },
                schema=schema
            )

source = Source(readers=[file_reader], query="select * from student_habits")
source_df = source.construct(spark_client)
source_df = source_df.withColumn("timestamp", current_timestamp())
print("Source")
source_df.show()

keys = [
    KeyFeature(
        name="student_id",
        description="Unique student identificator",
        dtype=DataType.STRING
    )
]

ts_feature = TimestampFeature()

features = [
    Feature(
        name="age",
        description="Age of the student",
        dtype=DataType.INTEGER
    ),
    Feature(
        name="gender",
        description="Gender of the student",
        dtype=DataType.STRING
    ),
    Feature(
        name="study_hours_per_day",
        description="Study hours per day of the student",
        dtype=DataType.DOUBLE,
        transformation=[std(id_column="student_id"), fillna_mean(id_column="student_id")]
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
feature_set_df.orderBy("student_id").show()
feature_set_df.printSchema()
