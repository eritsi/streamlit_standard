import streamlit as st
import pandas as pd
import pickle
import lightgbm as lgb
import matplotlib.pyplot as plt


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

    # # Collects user input features into dataframe
    # uploaded_file = st.sidebar.file_uploader(
    #     "Upload your input CSV file", type=["csv"])

    if 'ML_df' not in st.session_state:
        st.write("Please go back to 2. データ入力..")
    elif 'learning_col' not in st.session_state:
        st.write("Using all clustering for model creation")
    else:
        df = st.session_state['ML_df']
        selected_learning_col = st.session_state['learning_col']

        # Sidebar - 目的変数以外のカラムで、モデル生成に使う特徴量を選ぶ
        # 全選択からドロップしていく方法だとエラーが出る。おそらくsession_state関係
        # sorted_features = sorted(df.columns)
        sorted_features = sorted(list(set(df.columns) - set(df.columns[3:4])))
        selected_features = st.sidebar.multiselect(
            'Select Features for ML Model',
            sorted_features)

        # Sidebar - Cluster selection
        sorted_clusters = sorted(df[selected_learning_col].unique())

        selected_clusters = st.sidebar.multiselect(
            'Select Clusters for ML Model',
            sorted_clusters,
            sorted_clusters)

        # Sidebar - Model Selector
        MODELS = {
            "Light GBM": 0,
            "SARIMA": 1,
            "状態空間モデル": 2,
            "Prophet": 3
        }
        mdl = st.sidebar.radio("Select ML Models", MODELS)

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

        # カテゴリカル化が済んでいることが先へ進む条件
        if st.button('Create Model'):
            if mdl == 'Light GBM':
                df_modeling = df[df[selected_learning_col].isin(
                    selected_clusters)]
                X_train, y_train = df_modeling[selected_features], df_modeling.iloc[:, 3]
                st.write(X_train)
                st.session_state['X_train'] = X_train

                params = set_params()
                model = lgb.LGBMRegressor(**params)
                model.fit(X_train, y_train)
                st.session_state['model'] = model
                with open('test.pickle', mode='wb') as f:  # with構文でファイルパスとバイナリ書き込みモードを設定
                    pickle.dump(model, f)
                st.write('pickle has been created')

                st.session_state['categories'] = selected_clusters
            else:
                st.write("Not implemented yet...")
        else:
            st.write('Select Feature Columns and Learning Scope.')
            df_modeling = df[df[selected_learning_col].isin(selected_clusters)]
            X_train, y_train = df_modeling[selected_features], df_modeling.iloc[:, 3]
            st.session_state['X_train'] = X_train
            st.session_state['categories'] = selected_clusters
