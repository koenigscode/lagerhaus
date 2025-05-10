from butterfree.transform.transformations import CustomTransform
import pandas as pd
from sklearn.preprocessing import StandardScaler

def std(df, parent_feature):
    scaler = StandardScaler()
    df = scaler.fit_transform(df)
    df = pd.DataFrame(df, columns=df.columns)
    return df

NaiveBayes = CustomTransform(transformer= std, )