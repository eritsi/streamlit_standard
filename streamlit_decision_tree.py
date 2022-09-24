import streamlit as st
import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.tree import plot_tree
from sklearn import tree

# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806


def filedownload(df):
    csv = df.to_csv(index=False, encoding='utf-8_sig')
    # strings <-> bytes conversions
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="clustering.csv">Download CSV File</a>'
    return href


st.title('分析木のお試しサイト')
st.markdown("""
quick decision tree site.
""")
#---------------------------------#
# About
expander_bar = st.expander("About")
expander_bar.markdown("""
Pythonでの分析木作成の体験をしていただけます。
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader(
    "Upload your input CSV file", type=["csv"])

st.sidebar.markdown("""
想定データフォーマット
|  カラム名1  |  カラム名2  |  ...  |  カラム名N  |
| ---- | ---- | ---- | ---- |
|  データ1  |  データ2  |  ...  |  データN  |
""")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = None


def my_desicion_tree(dfx, dfy):
    fig, ax = plt.subplots()
    model = tree.DecisionTreeClassifier(max_depth=2, random_state=1)
    model.fit(dfx, dfy)
    plot_tree(model, feature_names=dfx.columns, class_names=True, filled=True)
    return fig


if uploaded_file is not None:
    st.subheader('分析木')
    st.write('読み込んだデータの冒頭と末尾を表示')
    st.write(df.head())
    st.write(df.tail())

    st.write('サイドバーで、分析したい列名を選んでください')
    # Sidebar - Column selection for learning scope
    sorted_cols = sorted(df.columns)
    selected_x_col = st.sidebar.multiselect(
        '説明変数のカラムを選んでください(複数可).',
        sorted_cols,
        sorted_cols)
    selected_y_col = st.sidebar.selectbox(
        '目的変数のカラムを選んでください.',
        sorted_cols,
        index=0)

    # test_x = df[['sex','type']]
    # test_y = df['purchase']
    test_x = df[selected_x_col]
    test_y = df[selected_y_col]
    test_x = pd.get_dummies(test_x, drop_first=True)
    test_y = pd.get_dummies(test_y, drop_first=True)

    f = my_desicion_tree(test_x, test_y)
    st.pyplot(f)
