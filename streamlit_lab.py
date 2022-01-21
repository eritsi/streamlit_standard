import streamlit as st
import pandas as pd
import numpy as np
import base64
from util_ml import get_dengram, add_one_item_in_dendrogram, plot_line_or_band, pivot_df_for_dengram
# from util_ml import datasetLoader

# Download clustering result
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806


def filedownload(df):
    csv = df.to_csv(index=False, encoding='utf-8_sig')
    # strings <-> bytes conversions
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="clustering.csv">Download CSV File</a>'
    return href


def app():
    st.title('demand categolizer')
    st.markdown("""
    This app retrieves clustering based on time-history data.
    """)
    #---------------------------------#
    # About
    expander_bar = st.expander("About")
    expander_bar.markdown("""
    * Input : csv table (from BQ table : future implement)
    * **ATTENTION:**
    一旦、id x T1 x T2 は抜け漏れなく、countにNullはない想定（今後充実させる方向で検討）
    """)

    # Collects user input features into dataframe
    uploaded_file = st.sidebar.file_uploader(
        "Upload your input CSV file", type=["csv"])

    st.sidebar.markdown("""
    想定データフォーマット

    |  id  |  T1  |  T2  |  count  |
    | ---- | ---- | ---- | ---- |
    |  JAN等  |  Year等  |  Month[1-12], Week[1-53], Day[1-366]等  |  目的変数  |

    """)

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = None
        df_clustering_input = None
    
    # # Access to GCP
    # st.sidebar.subheader('... Or get data by SQL')
    # SQL_input = "SELECT * \n FROM {DATASET.TABLE} \n ORDER BY {T1, T2}\n"

    # SQL_input = st.sidebar.text_area("SQL input", SQL_input, height=150)
    # dataset_loader = datasetLoader()

    # if st.sidebar.button('Send SQL'):
    #     df = dataset_loader.load(SQL_input)

    # Displays full user input dataframe
    st.subheader('1a. User Input')
    if (uploaded_file is not None) | (df is not None):
        st.write('Data Dimension: {} items and data '.format(len(df.iloc[:, 0].unique())) +
                 'from {}.{}'.format(min(df.iloc[:, 1]), min(df[(df.iloc[:, 1] == min(df.iloc[:, 1]))].iloc[:, 2])) +
                 ' to {}.{}.'.format(max(df.iloc[:, 1]), max(df[(df.iloc[:, 1] == max(df.iloc[:, 1]))].iloc[:, 2])) +
                 ' ({} ticks)'.format(max(df.groupby(df.columns[0]).size())))
        st.write('Number of the items which have full-ticks time-history data: ' +
                 str(sum(df.groupby(df.columns[0]).size() == max(df.groupby(df.columns[0]).size()))))
        st.write(df.head(10))
    else:
        st.write('Awaiting CSV file to be uploaded.')

    # Displays only the longest time-history items
    st.subheader('1b. Clustering Input (Only full length items)')
    if (uploaded_file is not None) | (df is not None):
        df_clustering_input = (df.iloc[:, 0].value_counts() == max(
            df.groupby(df.columns[0]).size()))
        df_clustering_input = df[(df.iloc[:, 0]).isin(
            df_clustering_input[df_clustering_input].index)]
        df_clustering_input = df_clustering_input.reset_index(drop=True)
        st.write('Data Dimension: {} items.'.format(
            len(df_clustering_input.iloc[:, 0].unique())))
        st.write(df_clustering_input)
    else:
        st.write(
            'Awaiting CSV file to be uploaded.')

    # Dendrogram parameters
    st.subheader('1c. Dendrogram Output')
    selected_threshold = st.sidebar.slider(
        'Dendrogram threshold', 0.0, 1.0, 0.17)

    if (uploaded_file is not None) | (df is not None):
        pivot_df = pivot_df_for_dengram(df_clustering_input)
        cluster_dict, fig = get_dengram(pivot_df, selected_threshold)
        st.pyplot(fig)

        df_long_tf = pd.DataFrame(cluster_dict, index=['cluster', ]).T
        df_long_tf.reset_index(inplace=True)
        df_long_tf.rename(columns={'index': 'product_code'}, inplace=True)

        st.write('Number of Clusters : {}'.format(
            len(df_long_tf.cluster.unique())))
    else:
        st.write(
            'Awaiting CSV file to be uploaded.')

    # Plot by Dendrogram cluster
    if st.button('Plot by dendrogram cluster'):
        #---------------------------------#
        # Page layout (continued)
        # Divide page to 3 columns (col1 = sidebar, col2 and col3 = page contents)
        col1, col2 = st.columns((0.97, 1))

        st.subheader('Cluster Plot')
        # Time-History Plot by clusters (Normalized)
        normalized_pivot_df = ((pivot_df.T - pivot_df.T.min()) /
                               (pivot_df.T.max() - pivot_df.T.min())).T
        for c in set(cluster_dict.values()):
            with col1:
                fig = plot_line_or_band(normalized_pivot_df, cluster_dict, c)
                col1.pyplot(fig)
            with col2:
                fig2 = plot_line_or_band(pivot_df, cluster_dict, c)
                col2.pyplot(fig2)

    # Clustering for other items with short time-history
    if st.button('clustering for shorter TH items'):
        st.subheader('Clustering result for shorter TH items')

        df_short_tf = pd.DataFrame()
        all_items = pd.unique(df[df.columns[0]])
        long_items = pd.unique(
            df_clustering_input[df_clustering_input.columns[0]])

        # launch dendrogram for all the items with shorter time-history, one by
        # one
        for item_code in list(set(all_items) - set(long_items)):
            one_item = pivot_df_for_dengram(df[df[df.columns[0]] == item_code])
            pivot_df_plus_one = pd.concat([pivot_df, one_item], axis=0)

            recommended_original_cluster, original_clusters, original_cluster_colleague_counts = add_one_item_in_dendrogram(
                pivot_df_plus_one, selected_threshold, item_code, cluster_dict)
            # record the result
            df_short_tf = pd.concat([df_short_tf,
                                     pd.DataFrame([item_code,
                                                   recommended_original_cluster,
                                                   original_clusters,
                                                   original_cluster_colleague_counts]).T],
                                    axis=0)

        df_short_tf.rename(
            columns={
                0: 'product_code',
                1: 'cluster',
                2: 'candidate_clusters',
                3: 'colleague_counts'},
            inplace=True)
        st.write(df_short_tf)

        st.subheader('Clustering result for all items')
        df_cluster = pd.concat([df_short_tf, df_long_tf])
        st.write(df_cluster)
        st.markdown(filedownload(df_cluster), unsafe_allow_html=True)
