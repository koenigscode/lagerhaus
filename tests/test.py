import pandas as pd
from feature_store import std

df = pd.read_csv("./student_habits_performance.csv")
print(df.dtypes)
df = std(df=df)
print(df)