import streamlit as st
import altair as alt
from lagerhaus.featuremanagement import FeatureStore, FeatureView

def init(title, feature_store: FeatureStore):
    with st.container(border=True):
        st.title(title)
        st.write("Feature Store Stats:")
        st.write(feature_store.get_all().describe())


def print(obj, title=None):
    """
    Wrapper around streamlit.write.
    Also accepts FeatureStore or FeatureView
    """
    if isinstance(obj, FeatureStore) or isinstance(obj, FeatureView):
        obj = obj.get_all()

    if title is not None:
        st.write(title)
    st.write(obj)

def plot_distribution(df, column, maxbins=10, title=None):
    st.write(title)

    hist = alt.Chart(df).mark_bar().encode(
        alt.X(column, bin=alt.Bin(maxbins=maxbins), title=column),
        alt.Y("count()", title="Frequency")
    ).properties(
        title=f"{column} Distribution"
    )

    st.altair_chart(hist, use_container_width=True)
