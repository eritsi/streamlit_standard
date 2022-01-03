import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt

st.title('demand model creator')
st.markdown("""
This app creates ML models. (For the moment : L-GBM)
""")
#---------------------------------#
# About
expander_bar = st.expander("About")
expander_bar.markdown("""
* Input : csv table (from BQ table : future implement)
* **ATTENTION:**
一旦、csvから読み込む
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader(
    "Upload your input CSV file", type=["csv"])
st.sidebar.markdown("""
想定データフォーマット

|  id  |  T1  |  T2  |  count  |  F1  |  F2  |  F3  |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|  JAN等  |  Year等  |  Month[1-12], Week[1-53], Day[1-366]等  |  目的変数  |  特徴量1  |  特徴量2  |  特徴量3  |

""")

df = None
df2 = None

def conventional_features(_df):
    # よく使われる特徴量を計算
    _df = _df.sort_values(_df.columns[0:3].to_list()).reset_index().drop("index", axis = 1)
    _df["min"]=_df.groupby([_df.columns[0]]).shift(1).rolling(selected_rolling_window)[_df.columns[3]].min().reset_index()[_df.columns[3]]
    _df["max"]=_df.groupby([_df.columns[0]]).shift(1).rolling(selected_rolling_window)[_df.columns[3]].max().reset_index()[_df.columns[3]]
    _df["std"]=_df.groupby([_df.columns[0]]).shift(1).rolling(selected_rolling_window)[_df.columns[3]].std().reset_index()[_df.columns[3]]
    _df["lag1"]=_df.groupby([_df.columns[0]]).shift(1).rolling(1)[_df.columns[3]].sum().reset_index()[_df.columns[3]]
    _df["cumsum1_2"]=_df.groupby([_df.columns[0]]).shift(1).rolling(2)[_df.columns[3]].sum().reset_index()[_df.columns[3]]
    _df["cumsum1_3"]=_df.groupby([_df.columns[0]]).shift(1).rolling(3)[_df.columns[3]].sum().reset_index()[_df.columns[3]]
    _df["cumsum1_4"]=_df.groupby([_df.columns[0]]).shift(1).rolling(4)[_df.columns[3]].sum().reset_index()[_df.columns[3]]

    return _df

if uploaded_file is not None:
    st.subheader('Display csv Inputs')
    df = pd.read_csv(uploaded_file)
    st.write(df.head(15))

else:
    st.write('Select Inputs first...')

if df is not None:

    # Sidebar - Categorical Feature selection
    sorted_categoricals = sorted(df.columns)
    selected_categoricals = st.sidebar.multiselect(
        'Select Categorical Features',
        sorted_categoricals,
        set(df.columns[0:3]))

    # Sidebar - Clustering Column selection
    selected_clustering = st.sidebar.selectbox(
        'Select Clustering Column',
        sorted_categoricals,
        index=4)

    # Side bar - rolling period
    selected_rolling_window = st.sidebar.slider(
        'Set a Rolling Windows', 
        0, 
        20, 
        6)

    if st.sidebar.button('Create More Features'):
        # よく使われる特徴量を計算
        df = conventional_features(df)

        # カテゴリカル変数へ変更
        for i in selected_categoricals:
            # st.sidebar.write(i)
            df[i] = df[i].astype('category')
        # カテゴリカル変数になると統計量が計算されなくなる
        # st.write(df.describe())

        st.subheader('Display Inputs for ML Model')
        st.write(df.head(40))

# # Sidebar - 目的変数以外のカラムで、モデル生成に使う特徴量を選ぶ
# # sorted_features = df.columns
# sorted_features = set(df.columns) - set(df.columns[3:4])
# selected_features = st.sidebar.multiselect(
#     'Select Features for ML Model',
#     sorted_features,
#     sorted_features)


# # Sidebar - Cluster selection
# sorted_clusters = sorted(df[selected_clustering].unique())

# selected_clusters = st.sidebar.multiselect(
#     'Select Clusters for ML Model',
#     sorted_clusters,
#     sorted_clusters)    

# # カテゴリカル化が済んでいることが先へ進む条件
# if st.sidebar.button('Create Model'):
#     df_modeling = df[df[cluster_col].isin(sorted_clusters)]
#     # Sidebar - Feature selection
#     st.sidebar.write(df_modeling)
# else:
#     st.write('Then, Create Features...')



