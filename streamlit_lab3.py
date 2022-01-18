import streamlit as st
import pandas as pd
import pickle
import lightgbm as lgb
import matplotlib.pyplot as plt
from PIL import Image
import copy

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

    image = Image.open('st_app3.png')
    st.image(image, width=500)

    # # Collects user input features into dataframe
    # uploaded_file = st.sidebar.file_uploader(
    #     "Upload your input CSV file", type=["csv"])

    if 'ML_df' not in st.session_state:
        st.write("Please go back to 2. データ入力..")
    elif 'classification_col' not in st.session_state:
        st.write("Using all clustering for model creation")
    else:
        df = st.session_state['ML_df']
        df_time = st.session_state['df_time']
        selected_classification_col = st.session_state['classification_col']

        # Sidebar - 目的変数以外のカラムで、モデル生成に使う特徴量を選ぶ
        # 全選択からドロップしていく方法だとエラーが出る。おそらくsession_state関係
        # sorted_features = sorted(df.columns)
        # sorted_features = sorted(list(set(df.columns) - set(df.columns[3:4]))) # - set("shifted_count")
        sorted_features = df.columns.to_list()
        st.write(sorted_features.pop(3))
        # try: sorted_features.remove('shifted_count')
        selected_features = st.sidebar.multiselect(
            'Select Features for ML Model',
            sorted_features,
            sorted_features)

        # Sidebar - Cluster selection
        sorted_clusters = sorted(df[selected_classification_col].unique())

        selected_clusters = st.sidebar.multiselect(
            'Select Clusters for ML Model',
            sorted_clusters,
            sorted_clusters)

        # Sidebar - Model Selector
        st.sidebar.subheader("ML Model and Prediction Period")
        MODELS = {
            "Light GBM": 0,
            "SARIMA": 1,
            "状態空間モデル": 2,
            "Prophet": 3
        }
        mdl = st.sidebar.radio("Select ML Models", MODELS)
        from_period = st.sidebar.slider('From when (incl.)', 1, 100, 1)
        to_period = st.sidebar.slider('To when (incl.)', 1, 100, 4)

        # Sidebar - Optuna
        st.sidebar.subheader("Tuning Hyper Parameters")
        num_leaves = st.sidebar.slider(
            'Number of Leaves (default: 31)', 500, 1500, 731)
        max_depth = st.sidebar.slider('Max Depth (default:-1)', -1, 128, 102)
        min_child_samples = st.sidebar.slider(
            'Min Child Samples (deafult: 20)', 1, 150, 100)  # default

        def set_params():
            params = {
                'num_leaves': num_leaves,
                'max_depth': max_depth,
                'boosting_type': 'gbdt',
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample_for_bin': 200000,
                'objective': None,
                'class_weight': None,
                'min_split_gain': 0.0,
                'min_child_weight': 0.001,
                'min_child_samples': min_child_samples,
                'subsample': 1.0,
                'subsample_freq': 0,
                'colsample_bytree': 1.0,
                'reg_alpha': 0.0,
                'reg_lambda': 0.0,
                'random_state': 0,
                'n_jobs': -1,
                'importance_type': 'split'
            }
            return params
        def shift_df(df, n_shift):
            _df = df.copy()
            _df["shifted_count"] = _df.groupby([_df.columns[0]]).shift(-n_shift).rolling(
            1)[_df.columns[3]].sum().reset_index()[_df.columns[3]]
            # sorted_features.pop(-1)
            # try: sorted_features.remove('shifted_count')
            return _df

        # カテゴリカル化が済んでいることが先へ進む条件
        if st.button('Create Model'):
            if mdl == 'Light GBM':
                # 図の通り、　
                # ①countをto_period分シフトする
                # ②to_period分のデータを捨てる：すなわち df_time.tail(to_period) を落としてinner join
                # ③捨てた部分を推論用に次のセッションへ送る
                
                # ①シフトする
                df_modeling = shift_df(df, to_period)
                # ②学習期間のdfを切り出す
                df_train = pd.merge(df_modeling, df_time[:-to_period], how='inner')
                # 学習対象を選ばれしクラスタのみにする
                df_train = df_train[df_train[selected_classification_col].isin(
                    selected_clusters)]
                # ③推論期間のdfを切り出す。推論対象は選択外のクラスタに対しても実施する可能性を残す。
                df_inference = pd.merge(df_modeling, df_time.tail(to_period), how='inner')[selected_features]
                
                st.write('X_train')
                X_train, y_train = df_train[selected_features], df_train["shifted_count"]
                st.write(X_train.columns.to_list())
                
                params = set_params()
                model = lgb.LGBMRegressor(**params)
                model.fit(X_train, y_train)
                st.session_state['model'] = model
                with open('test.pickle', mode='wb') as f:  # with構文でファイルパスとバイナリ書き込みモードを設定
                    dic = {'model': model,
                            'X': df_inference,
                            'from': from_period,
                            'to': to_period,
                            'categories': selected_clusters
                            }
                    pickle.dump(dic, f)
                st.write('pickle has been created')

                st.session_state['categories'] = selected_clusters
                st.session_state['X_inference'] = df_inference
                st.session_state['from_period'] = from_period
                st.session_state['to_period'] = to_period
            else:
                st.write("Not implemented yet...")
        else:
            st.write('Select Feature Columns and Learning Scope.')
            df_modeling = shift_df(df, to_period)
            df_inference = pd.merge(df_modeling, df_time.tail(to_period), how='inner')[selected_features]
            st.session_state['categories'] = selected_clusters
            st.session_state['X_inference'] = df_inference
            st.session_state['from_period'] = from_period
            st.session_state['to_period'] = to_period
