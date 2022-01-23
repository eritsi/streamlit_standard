import streamlit as st
import pandas as pd
import shap
import pickle
import lightgbm as lgb
import matplotlib.pyplot as plt
from util_ml import pivot_df_for_dengram, filedownload
from util_ml import datasetLoader

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

    df_true = None
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
        selected_classification_col = st.session_state['classification_col']
        from_period = st.session_state['from_period']
        to_period = st.session_state['to_period']
    uploaded_file = st.sidebar.file_uploader(
        "Upload your y_true CSV file", type=["csv"])
    if uploaded_file is not None:
        df_true = pd.read_csv(uploaded_file)

    # Access to GCP
    st.sidebar.subheader('... Or get data by SQL')
    SQL_input = "SELECT * \n FROM {DATASET.TABLE} \n ORDER BY {T1, T2}\n"

    SQL_input = st.sidebar.text_area("SQL input", SQL_input, height=150)
    dataset_loader = datasetLoader()

    if st.sidebar.button('Send SQL'):
        df_true = dataset_loader.load(SQL_input)
        st.session_state['df_true'] = df_true

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
        if 'df_true' in st.session_state:
            df_true = st.session_state['df_true']
        if df_true is None:
            st.write('Please upload y_true data for calendrical process.')
        else:
            df_inf = model.predict(X_inference)
            def inc_df(df, current_t1, current_t2, shift_by):
                _df = df.iloc[:,[1,2]] \
                        .drop_duplicates() \
                        .sort_values([df.columns[1],df.columns[2]]) \
                        .reset_index(drop=True)
                shift_index = _df[(_df.iloc[:,0]==current_t1) & (_df.iloc[:,1]==current_t2)].index \
                                + shift_by
                shifted_t1=_df.iloc[shift_index.to_list()[0],0]
                shifted_t2=_df.iloc[shift_index.to_list()[0],1]
                return shifted_t1, shifted_t2
            def inference_time_convert_df(df, df_true, _from, _to): # fromは現状使ってない。5-8週モデルで確認
                time_conv = df.iloc[:, [1,2]].drop_duplicates().sort_values([df.columns[1],df.columns[2]]).reset_index(drop=True)
                time_conv['temp_t1'] = None
                time_conv['temp_t2'] = None
                for i in time_conv.index:
                    shifted_t1, shifted_t2 = inc_df(df_true, time_conv.iloc[i, 0], time_conv.iloc[i, 1], to_period)
                    time_conv.at[time_conv.index[i],'temp_t1'] = shifted_t1
                    time_conv.at[time_conv.index[i],'temp_t2'] = shifted_t2
                return time_conv
            def id_pred(x):
                return str(int(x[0])) + '_pred'         

            conv_df = inference_time_convert_df(X_inference, df_true, from_period, to_period)
            conv_df = pd.merge(X_inference.iloc[:, 0:3], conv_df)
            conv_df.drop(conv_df.columns[[1, 2]], axis=1, inplace=True) 
            conv_df.rename(columns={'temp_t1': X_inference.columns[1],'temp_t2': X_inference.columns[2] }, inplace=True)
            conv_df[X_inference.columns[0]] = conv_df.apply(id_pred, axis=1)
            # st.write(conv_df)
            df_inf = pd.concat([conv_df, pd.DataFrame(df_inf, columns=[df_true.columns[3]]), X_inference[[selected_classification_col]]], axis=1)
            st.session_state['df_inf'] = df_inf
            st.write(df_inf)
            
            # 129行目までは、「とりあえず」時代の名残
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
            # shift_t1, shift_t2 = inc_df(df_true, X_inference['client_year'], X_inference['client_week_num'], 1)
            # X_inference['client_year'] = X_inference.apply(inc_year, axis=1)
            # X_inference['client_week_num'] = X_inference.apply(inc_week, axis=1)
            # X_inference['product_code'] = X_inference.apply(id_pred, axis=1)
            # df_inf = pd.concat([X_inference.iloc[:, 0:3], pd.DataFrame(df_inf, columns=['sales']), X_inference[[selected_classification_col]]], axis=1)
            
        

    if st.button('Graph'):
        if 'df_true' in st.session_state:
            df_true = st.session_state['df_true']
        
        input_df = st.session_state['input_df']
        df_inf = st.session_state['df_inf']
        # クラスタ毎の描写
        # 推論部分のy_trueを含んだdfと、クラスタ番号リストをjoinする
        input_df = pd.merge(df_true, input_df[[input_df.columns[0], selected_classification_col]].drop_duplicates())
        input_df.iloc[:, 0] = input_df.iloc[:, 0].astype('str')
        input_df = pd.concat([df_inf, input_df])
        st.write(input_df)
        st.markdown(filedownload(input_df), unsafe_allow_html=True)
        # 推論結果をid_predとしてconcat
        # 学習期間のラストを得る
        # st.write(input_df.iloc[:, 1:3].drop_duplicates())
        st.subheader.write('Plots for learned clusters: ')
        for c in sorted(selected_clusters):
            pivot_df = pivot_df_for_dengram(input_df[input_df[selected_classification_col] == c].iloc[:, 0:4])
            # st.write(pivot_df.T)

            fig = plt.figure(figsize=(15, 10 / 2))
            ax = fig.add_subplot(
                1, 1, 1, title="cluster = {}".format(c))
            pivot_df.T.plot(figsize=(10, 5), ax=ax)
            st.pyplot(fig)
        
        st.subheader.write('Plots for NOT learned clusters: ')
        for c in sorted(set(input_df['cluster'].drop_duplicates()) - set(selected_clusters)):
            pivot_df = pivot_df_for_dengram(input_df[input_df[selected_classification_col] == c].iloc[:, 0:4])
            # st.write(pivot_df.T)

            fig = plt.figure(figsize=(15, 10 / 2))
            ax = fig.add_subplot(
                1, 1, 1, title="cluster = {}".format(c))
            pivot_df.T.plot(figsize=(10, 5), ax=ax)
            st.pyplot(fig)


            # Remaining To Do : 
            # テスト -------------
            # 1-1, 5-8の時の挙動確認
            # pickleから入る場合のデバッグ
            # 実装 ---------------
            # 他モデルの実装
            # MAPE計算追加
            # id選んで一個ずつy_true, y_predを表示させる
            
            # important ids の対応
