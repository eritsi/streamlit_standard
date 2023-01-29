import streamlit as st
import pandas as pd
import numpy as np
import base64
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import japanize_matplotlib
from statsmodels.tsa.seasonal import seasonal_decompose


# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df):
    csv = df.to_csv(index=False, encoding='utf-8_sig')
    # strings <-> bytes conversions
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="clustering.csv">Download CSV File</a>'
    return href


st.title('EDAお試しサイト')
st.markdown("""
quick EDA site.
""")
#---------------------------------#
# About
expander_bar = st.expander("About")
expander_bar.markdown("""
講義で使ったEDAのノートブックをstreamlitで再度実装しています。
Pythonを書かない一般ユーザーにもデータハンドリングを経験していただけます。
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

if uploaded_file is not None:
    st.subheader('1. データフレームの特徴')
    st.write('読み込んだデータの冒頭と末尾を表示')
    st.write(df.head())
    st.write(df.tail())
    st.write('読み込んだデータの行数、列数')
    st.write(df.shape)
    st.write('読み込んだデータの基本統計量を表示')
    st.write(df.describe().T)

    st.write('読み込んだデータに欠損(黄色)があるか表示')
    f, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis")
    st.pyplot(f)

    st.write('読み込んだデータの相関関係を表示')
    corr = df.corr()
    sns.heatmap(corr,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
    st.pyplot(f)

    st.subheader('2. 一変数の分析')
    st.write('サイドバーで、分析したい列名を選んでください')
    st.sidebar.subheader('2.')

    # Sidebar - Column selection for learning scope
    sorted_cols = sorted(df.columns)
    selected_time_col = st.sidebar.selectbox(
        '時刻のカラムを選んでください.',
        sorted_cols,
        index=3)
    selected_col = st.sidebar.selectbox(
        '一変数の分析を行いたいカラムを選んでください.',
        sorted_cols,
        index=3)

    df_single = pd.Series(df[selected_col], dtype='int')
    df_single.index = pd.to_datetime(df[selected_time_col])
    st.write(df_single.T)

    st.write('選択した1変数データの時系列プロット')
    f2, ax2 = plt.subplots(figsize=(10, 5))
    df_single.plot()
    plt.ylabel(selected_col)
    st.pyplot(f2)

    st.write('選択した1変数データのヒストグラム')
    selected_binsize = st.slider(
        'ヒストグラムのビンサイズ',
        5,
        25,
        10)
    f3, ax3 = plt.subplots(figsize=(10, 5))
    plt.hist(df_single, bins=selected_binsize)
    plt.xlabel(selected_col, fontsize=18)
    plt.ylabel("frequency", fontsize=18)
    plt.tick_params(labelsize=18)
    st.pyplot(f3)

    st.write('選択した1変数データのコレログラム')
    f4, ax4 = plt.subplots(2, 1, figsize=(12, 8))
    fig = sm.graphics.tsa.plot_acf(
        df_single,
        lags=24,
        ax=ax4[0],
        color="darkgoldenrod")
    fig = sm.graphics.tsa.plot_pacf(
        df_single, lags=24, ax=ax4[1], color="darkgoldenrod")
    plt.show()
    st.pyplot(f4)

    st.subheader('3. 二変数の分析')
    st.write('サイドバーで、分析したい列名をもう一つ選んでください')
    st.sidebar.subheader('3.')

    # Sidebar - Column selection for learning scope
    selected_col2 = st.sidebar.selectbox(
        '二つ目の変数のカラムを選んでください.',
        sorted_cols,
        index=3)

    selected_cols = [selected_col] + [selected_col2]
    df2 = df[[selected_time_col] + [selected_col] + [selected_col2]]
    df2.index = pd.to_datetime(df2[selected_time_col])
    st.write(df2.head())

    st.write('選択した2変数データの箱ひげ図、分布')
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 5))
    axes[0, 0].set_title('Box-{}'.format(selected_cols[0]))
    axes[0, 1].set_title('DistributionPlot-{}'.format(selected_cols[0]))
    axes[1, 0].set_title('Box-{}'.format(selected_cols[1]))
    axes[1, 1].set_title('DistributionPlot-{}'.format(selected_cols[1]))

    sns.boxplot(df2[selected_cols[0]], orient='v', ax=axes[0, 0])
    sns.distplot(df2[selected_cols[0]], ax=axes[0, 1])
    sns.boxplot(df2[selected_cols[1]], orient='v', ax=axes[1, 0])
    sns.distplot(df2[selected_cols[1]], ax=axes[1, 1])

    fig.tight_layout()
    st.pyplot(fig)

    st.write('選択した2変数データの時系列')
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(111)
    ax1.plot(df2.index, df2[selected_cols[0]], 'C0')
    ax2 = ax1.twinx()
    ax2.plot(df2.index, df2[selected_cols[1]], 'C1')

    ax1.set_xlabel(selected_time_col)
    ax1.set_ylabel(selected_col, color='C0')
    ax2.set_ylabel(selected_col2, color='C1')
    st.pyplot(fig)
