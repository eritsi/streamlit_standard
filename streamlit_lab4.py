import streamlit as st
import pandas as pd
import shap
import pickle
import lightgbm as lgb
import matplotlib.pyplot as plt
from util_ml import pivot_df_for_dengram


def app():
    st.title('demand visualizer')
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

    if 'model' not in st.session_state:
        pickle_file = st.sidebar.file_uploader(
            "Upload your input pickle file", type=["pickle"])
        if pickle_file:
            pckl = pickle.load(pickle_file)
            model = pckl['model']
            st.write(model)
            X_inference = pckl['X']
            selected_clusters = pckl['categories']
            st.write('Feature columns in order...')
            st.write(X_inference.columns.to_list())
            st.write('Clusters used for training')
            st.write(selected_clusters)
    elif 'X_inference' not in st.session_state:
        st.write("Go back to 3.")
    else:
        X_inference = st.session_state['X_inference']
        model = st.session_state['model']
        selected_clusters = st.session_state['categories']
    uploaded_file = st.sidebar.file_uploader(
        "Upload your y_true CSV file", type=["csv"])
    if uploaded_file is not None:
        st.subheader('Display csv Inputs')
        df_true = pd.read_csv(uploaded_file)

    if st.button('SHAP'):
        st.set_option('deprecation.showPyplotGlobalUse', False)

        # Explaining the model's predictions using SHAP values
        # https://github.com/slundberg/shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_inference)
        

        st.header('Feature Importance')
        plt.title('Feature importance based on SHAP values')
        shap.summary_plot(shap_values, X_inference)
        st.pyplot(bbox_inches='tight')
        st.write('---')

        plt.title('Feature importance based on SHAP values (Bar)')
        shap.summary_plot(shap_values, X_inference, plot_type="bar")
        st.pyplot(bbox_inches='tight')
    if st.button('inference'):
        df_inf = model.predict(X_inference)
        # 93行目まで一旦、とりあえず
        def inc_year(x):
            if x['client_year'] ==2020:
                return 2021
        def inc_week(x):
            if x['client_week_num'] == 50:
                return 1
            elif x['client_week_num'] == 51:
                return 2
            elif x['client_week_num'] == 52:
                return 3
            elif x['client_week_num'] == 53:
                return 4
        def id_pred(x):
            return str(int(x['product_code'])) + '_pred'         
        X_inference['client_year'] = X_inference.apply(inc_year, axis=1)
        X_inference['client_week_num'] = X_inference.apply(inc_week, axis=1)
        X_inference['product_code'] = X_inference.apply(id_pred, axis=1)
        df_inf = pd.concat([X_inference.iloc[:, 0:3], pd.DataFrame(df_inf, columns=['sales']), X_inference[['cluster']]], axis=1)
        st.session_state['df_inf'] = df_inf
        st.write(df_inf)
        

    if st.button('Graph'):
        ML_df = st.session_state['ML_df']
        df_inf = st.session_state['df_inf']
        # クラスタ毎の描写
        # 推論部分のy_trueを含んだdfと、クラスタ番号リストをjoinする
        ML_df = pd.merge(df_true, ML_df[[ML_df.columns[0], 'cluster']].drop_duplicates())
        ML_df.iloc[:, 0] = ML_df.iloc[:, 0].astype('str')
        ML_df = pd.concat([df_inf, ML_df])
        st.write(ML_df)
        # 推論結果をid_predとしてconcat
        # 学習期間のラストを得る
        # st.write(ML_df.iloc[:, 1:3].drop_duplicates())
        for c in sorted(selected_clusters):
            pivot_df = pivot_df_for_dengram(ML_df[ML_df['cluster'] == c].iloc[:, 0:4])
            # st.write(pivot_df.T)

            fig = plt.figure(figsize=(15, 10 / 2))
            ax = fig.add_subplot(
                1, 1, 1, title="cluster = {}".format(c))
            pivot_df.T.plot(figsize=(10, 5), ax=ax)
            st.pyplot(fig)

            # Remaining To Do : clusterを全部App2の選択肢へ差し替え
            # shifted_countを選べないように。
            # NaNを最初に落とせば、全Feature入れておいても大丈夫そう
            # pickleから入る場合のデバッグ
            # App4とりあえず、を可変に差し替え
            # App1のクラスタリングを引き継ぐやり方
            # id選んで一個ずつy_true, y_predを表示させる
