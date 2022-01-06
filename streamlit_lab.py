import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


def app():
    st.title('demand forcaster')
    st.markdown("""
    This app retrieves simple regression model.
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

    # Displays the user input features
    st.subheader('1a. User Input')
    if uploaded_file is not None:
        st.write('Data Dimension: ' +
                 str(len(df.iloc[:, 0].unique())) +
                 ' items and data from ' +
                 str(min(df.iloc[:, 1])) +
                 '. ' +
                 str(min(df[(df.iloc[:, 1] == min(df.iloc[:, 1]))].iloc[:, 2])) +
                 ' to ' +
                 str(max(df.iloc[:, 1])) +
                 '. ' +
                 str(max(df[(df.iloc[:, 1] == max(df.iloc[:, 1]))].iloc[:, 2])) +
                 '. It means ' +
                 str(max(df.groupby(df.columns[0]).size())) +
                 ' ticks and ' +
                 str(sum(df.groupby(df.columns[0]).size() == max(df.groupby(df.columns[0]).size()))) +
                 ' items have full-ticks time-history data.')
        st.write(df)
    else:
        st.write(
            'Awaiting CSV file to be uploaded.')

    # Select items with the longest time-history
    st.subheader('1b. Clustering Input (Only full length items)')

    if uploaded_file is not None:
        df_clustering_input = (df.iloc[:, 0].value_counts() == max(
            df.groupby(df.columns[0]).size()))
        df_clustering_input = df[(df.iloc[:, 0]).isin(
            df_clustering_input[df_clustering_input].index)]
        df_clustering_input = df_clustering_input.reset_index(drop=True)
        st.write('Data Dimension: ' +
                 str(len(df_clustering_input.iloc[:, 0].unique())) +
                 ' items.'
                 )
        st.write(df_clustering_input)
    else:
        st.write(
            'Awaiting CSV file to be uploaded.')

    # Dendrogram
    st.subheader('1c. Dendrogram Output')
    selected_threshold = st.sidebar.slider(
        'Dendrogram threshold', 0.0, 1.0, 0.17)

    # dropna()はDynamic　Warpingの実装を見据えるとイマイチなので直したい
    # 関数の中で正規化してるが、表示のために関数外でも正規化することになるので無駄っぽい
    # method='ward'固定は嫌だがとりあえず。

    def get_dengram(_df, _threshold, _plot=True):
        _normalized_df = ((_df.dropna().T - _df.dropna().T.min()) /
                          (_df.dropna().T.max() - _df.dropna().T.min())).T
        clustered = linkage(_normalized_df, method='ward', metric='euclidean')

        if _plot:
            fig = plt.figure(figsize=(15, 10 / 2))
            ax = fig.add_subplot(1, 1, 1, title="dendrogram")
            plt.xticks(rotation=90)  # JANコードは長くて見づらいので回転させる

        dendrogram(clustered, color_threshold=_threshold *
                   max(clustered[:, 2]), labels=_df.index)

        t = _threshold * max(clustered[:, 2])
        c = fcluster(clustered, t, criterion='distance')
        if _plot:
            st.pyplot(fig)

        return dict(zip(list(_df.dropna().index), list(c)))

    if uploaded_file is not None:
        pivot_df = df_clustering_input.pivot_table(
            index=df.columns[0],
            columns=[
                df.columns[1],
                df.columns[2]],
            values=df.columns[3])
        cluster_dict = get_dengram(pivot_df, selected_threshold)

        df_cluster = pd.DataFrame(cluster_dict, index=['cluster', ]).T
        df_cluster.reset_index(inplace=True)
        df_cluster.rename(columns={'index': 'product_code'}, inplace=True)

        st.write('Number of Clusters : ' +
                 str(max(df_cluster.cluster))
                 )
        # Plot by Dendrogram cluster
        normalized_pivot_df = ((pivot_df.T - pivot_df.T.min()) /
                               (pivot_df.T.max() - pivot_df.T.min())).T

        def display_by_cluster(d, l, a): return [
            a.append(k) for k, v in d.items() if v == l]
    else:
        st.write(
            'Awaiting CSV file to be uploaded.')

    def plot_line_or_band(_df, _cluster):
        a = []
        fig2 = plt.figure(figsize=(15, 10 / 2))
        ax2 = fig2.add_subplot(
            1, 1, 1, title="dendrogram cluster = {}".format(_cluster))
        display_by_cluster(cluster_dict, _cluster, a)
        #plt.subplot(int(a),2, 1)
        _df.loc[(a), :].T.plot(figsize=(10, 5), ax=ax2)
        st.pyplot(fig2)

    if st.button('Plot by dendrogram cluster'):
        st.subheader('Cluster Plot')

        for i in set(cluster_dict.values()):
            plot_line_or_band(normalized_pivot_df, i)

    # Clustering for other items with short time-history

    def _add_one_item_in_dendrogram(_item_code, _original_cluster_dict):
        selected_one = df[df[df.columns[0]] == _item_code].pivot_table(
            index=df.columns[0],
            columns=[
                df.columns[1],
                df.columns[2]],
            values=df.columns[3])

        pivot_df_plus_one = pd.concat([pivot_df, selected_one], axis=0)
        # fillna(0)して長さを合わせる　 vs. DTW vs. 今年の値をリピートさせる
        cluster_dict_plus_one = get_dengram(
            pivot_df_plus_one.fillna(0), selected_threshold, False)
        # これは新しいデンドロでのクラスタ番号
        plus_one_cluster_id = cluster_dict_plus_one[_item_code]
        # 新しいデンドロで同じクラスタに入った他のJANを取得
        colleague_keys = [
            k for k,
            v in cluster_dict_plus_one.items() if v == plus_one_cluster_id]
        colleague_keys.remove(_item_code)
        # 同僚itemが前のデンドロでどこのクラスタにいたかを取得
        colleague_original_clusters = [
            _original_cluster_dict[colleague_keys[i]] for i in range(0, len(colleague_keys))]
        original_clusters, original_cluster_colleague_counts = np.unique(
            colleague_original_clusters, return_counts=True)
        # 複数のクラスタが得られた場合、より多くの同僚itemがいたクラスタを取得
        recommended_original_cluster = original_clusters[[i for i, v in enumerate(
            original_cluster_colleague_counts) if v == max(original_cluster_colleague_counts)][0]]

        print('for item:{} newly entered cluster is {}'.format(
            _item_code, recommended_original_cluster))

        return recommended_original_cluster, original_clusters, original_cluster_colleague_counts

    if st.button('clustering for shorter TH items'):
        st.subheader('Cluster shorts')
        df_short_tf = pd.DataFrame()
        for i, item_code in enumerate(list(set(pd.unique(df[df.columns[0]])) - set(
                pd.unique(df_clustering_input[df_clustering_input.columns[0]])))):
            recommended_original_cluster, original_clusters, original_cluster_colleague_counts = _add_one_item_in_dendrogram(
                item_code, cluster_dict)
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
