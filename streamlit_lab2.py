import streamlit as st
import pandas as pd
from PIL import Image

def app():
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
    - Featureカラム数は任意  
    - カテゴリカル情報を読込  

    |  id  |  T1  |  T2  |  count  |  F1  |  F2  |  F3  |
    | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
    |  JAN等  |  Year等  |  Month[1-12], Week[1-53], Day[1-366]等  |  目的変数  |  特徴量1  |  特徴量2  |  特徴量3  |

    """)

    image = Image.open('st_app2.png')
    st.image(image, width=500)

    df = None
    if 'ML_df' not in st.session_state:
        st.session_state['ML_df'] = pd.DataFrame()
    if 'classification_col' not in st.session_state:
        st.session_state['classification_col'] = None

    def conventional_features(_df):
        # よく使われる特徴量を計算
        _df = _df.sort_values(
            _df.columns[0:3].to_list()).reset_index().drop("index", axis=1)
        _df["min"] = _df.groupby([_df.columns[0]]).shift(1).rolling(
            selected_rolling_window)[_df.columns[3]].min().reset_index()[_df.columns[3]]
        _df["max"] = _df.groupby([_df.columns[0]]).shift(1).rolling(
            selected_rolling_window)[_df.columns[3]].max().reset_index()[_df.columns[3]]
        _df["std"] = _df.groupby([_df.columns[0]]).shift(1).rolling(
            selected_rolling_window)[_df.columns[3]].std().reset_index()[_df.columns[3]]
        _df["lag1"] = _df.groupby([_df.columns[0]]).shift(1).rolling(
            1)[_df.columns[3]].sum().reset_index()[_df.columns[3]]
        _df["cumsum1_2"] = _df.groupby([_df.columns[0]]).shift(1).rolling(
            2)[_df.columns[3]].sum().reset_index()[_df.columns[3]]
        _df["cumsum1_3"] = _df.groupby([_df.columns[0]]).shift(1).rolling(
            3)[_df.columns[3]].sum().reset_index()[_df.columns[3]]
        _df["cumsum1_4"] = _df.groupby([_df.columns[0]]).shift(1).rolling(
            4)[_df.columns[3]].sum().reset_index()[_df.columns[3]]

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

        # Sidebar - Column selection for learning scope
        selected_classification_col = st.sidebar.selectbox(
            'Select Column to limit the learning target.',
            sorted_categoricals,
            index=3)

        # Side bar - rolling period
        selected_rolling_window = st.sidebar.slider(
            'Set a Rolling Windows for Min/Max/Std calc',
            0,
            20,
            6)

        if st.button('Create More Features'):
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
            st.write('DataFrame is ready. Please go to next app(3. 学習).')
            st.session_state['ML_df'] = df
            st.session_state['classification_col'] = selected_classification_col
            st.session_state['df_time'] = df.iloc[:,[1,2]].drop_duplicates()
