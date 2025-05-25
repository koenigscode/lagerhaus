import pandas as pd
from lagerhaus.featuremanagement import FeatureStore, FeatureView, FeatureMetadata
from lagerhaus.datacleaning.preprocessing import *
from lagerhaus.datacleaning import presets
from sklearn.linear_model import LogisticRegression


def datetime_transformer(df: pd.DataFrame, feature_view: FeatureView) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    return df


metadata = {
    "timestamp": FeatureMetadata(description="Transaction Timestamp"),
    "amount": FeatureMetadata(description="Transaction Amount", dtype=np.float64),
    "merchant_category": FeatureMetadata(
        description="Merchant Category", categorical=True, dtype=str
    ),
    "geo_anomaly_score": FeatureMetadata(
        description="Geo Anomaly Score", dtype=np.float64
    ),
    "payment_channel": FeatureMetadata(
        description="Payment Channel", categorical=True, dtype=str
    ),
    "is_fraud": FeatureMetadata(description="Is Fraud", dtype=bool),
}

df = pd.read_csv("./tests/datasets/financial_fraud_excerpt.csv")
fs = FeatureStore(metadata=metadata, df=df)
fv = FeatureView(
    feature_store=fs,
    transformers=[
        datetime_transformer,
        drop_columns(["timestamp"]),
        *presets.logistic_regression,
    ],
)

X_train, X_test, y_train, y_test = fv.get_train_test_split(y="is_fraud")

model = LogisticRegression()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))
