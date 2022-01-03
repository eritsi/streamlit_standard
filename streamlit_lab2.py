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

@st.cache
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

    # Side bar - rolling period
    selected_rolling_window = st.sidebar.slider(
        'Set a Rolling Windows', 
        0, 
        20, 
        6)

    if st.sidebar.button('Create More Features'):
        # よく使われる特徴量を計算
        df2 = conventional_features(df)

        # カテゴリカル変数へ変更
        for i in selected_categoricals:
            # st.sidebar.write(i)
            df2[i] = df2[i].astype('category')
        # カテゴリカル変数になると統計量が計算されなくなる
        # st.write(df.describe())

        st.subheader('Display Inputs for ML Model')
        st.write(df2.head(40))

if df2 is not None:

    sorted_features = sorted(df2.columns)
    selected_features = st.sidebar.multiselect(
        'Select Features for ML Model',
        sorted_features,
        set(sorted_features) - set(df2.columns[0]))

    # Sidebar - Cluster Column selection
    cluster_col = st.sidebar.selectbox(
        'Select Cluster Column',
        sorted_features
    )

    # Sidebar - Cluster selection
    sorted_clusters = sorted(df[cluster_col].unique())
    selected_fclusters = st.sidebar.multiselect(
        'Select Clusters for ML Model',
        sorted_clusters,
        sorted_clusters)    

    # カテゴリカル化が済んでいることが先へ進む条件
    if st.sidebar.button('Create Model'):
        # Sidebar - Feature selection
        st.sidebar.write('WWIP...')
    else:
        st.write('Then, Create Features...')





# if st.sidebar.button('Create Model'):
#     # オプションに沿ってモデルを作成
#     st.sidebar.write('WIP...')


# if df2 is not None:

#     # Sidebar - Feature selection
#     sorted_features = sorted(df2.columns)
#     selected_features = st.sidebar.multiselect(
#         'Select Features for ML Model',
#         sorted_features,
#         set(sorted_features) - set(df2.columns[3]))

#     # Sidebar - Cluster Column selection
#     cluster_col = st.sidebar.selectbox(
#         'Select Cluster Column',
#         sorted_features
#     )

#     # Sidebar - Cluster selection
#     sorted_clusters = sorted(df2[cluster_col].unique())
#     selected_fclusters = st.sidebar.multiselect(
#         'Select Clusters for ML Model',
#         sorted_clusters,
#         sorted_clusters)
    
#     if st.sidebar.button('Create Model'):
#         # オプションに沿ってモデルを作成
#         st.sidebar.write('WIP...')


# else:
#     df2 = pd.DataFrame()

