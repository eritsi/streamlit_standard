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
            model = pickle.load(pickle_file)
            st.write(model)
            X_train = st.session_state['X_train']
            selected_clusters = st.session_state['categories']
    elif 'X_train' not in st.session_state:
        st.write("Go back to 3.")
    else:
        X_train = st.session_state['X_train']
        model = st.session_state['model']
        selected_clusters = st.session_state['categories']
    
    if st.button('SHAP'):
        st.set_option('deprecation.showPyplotGlobalUse', False)

        # Explaining the model's predictions using SHAP values
        # https://github.com/slundberg/shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)

        st.header('Feature Importance')
        plt.title('Feature importance based on SHAP values')
        shap.summary_plot(shap_values, X_train)
        st.pyplot(bbox_inches='tight')
        st.write('---')

        plt.title('Feature importance based on SHAP values (Bar)')
        shap.summary_plot(shap_values, X_train, plot_type="bar")
        st.pyplot(bbox_inches='tight')

    if st.button('Graph'):
        # 推論してy_trueとグラフ描画
        for c in selected_clusters:
            ML_df = st.session_state['ML_df']
            st.write(c)
            st.write(ML_df[ML_df['cluster']==c].iloc[:, 0:4])
    
