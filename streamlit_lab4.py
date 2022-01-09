import streamlit as st
import pandas as pd
import shap
import pickle
import lightgbm as lgb
import matplotlib.pyplot as plt


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

    if st.button('Graph'):
        # 推論してy_trueとグラフ描画
        for c in selected_clusters:
            ML_df = st.session_state['ML_df']
            st.write(c)
            st.write(ML_df[ML_df['cluster'] == c].iloc[:, 0:4])
