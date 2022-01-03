# wip https://qiita.com/niship2/items/f0c825c6f0d291583b27
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

class _SessionState:

    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()

    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False

        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)


def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")

    return session_info.session


def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state

def main():
    state = _get_state()
    pages = {
        "クラスタリング":clustering_page,
        "データ加工":input_page,
        "学習":ml_page,
        "データ可視化":out_page
    }

    #st.sidebar.title(":floppy_disk: Page states")
    page = st.sidebar.radio("ページ選択", tuple(pages.keys()))

    # Display the selected page with the session state
    pages[page](state)

    # Mandatory to avoid rollbacks with widgets, must be called at the end of your app
    state.sync()

def clustering_page(state):
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
        df = pd.DataFrame()

    # Displays the user input features
    st.subheader('User Input')
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

    if uploaded_file is not None:
        st.write(df)
    else:
        st.write(
            'Awaiting CSV file to be uploaded.')
        st.write(df)

    # Select items with the longest time-history
    st.subheader('Clustering Input')

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
    st.subheader('Dendrogram Output')
    selected_threshold = st.sidebar.slider('Dendrogram threshold', 0.0, 1.0, 0.17)

    # dropna()はDynamic　Warpingの実装を見据えるとイマイチなので直したい
    # 関数の中で正規化してるが、表示のために関数外でも正規化することになるので無駄っぽい
    # method='ward'固定は嫌だがとりあえず。


    def get_dengram(_df, _threshold, _plot=True):
        _normalized_df = ((_df.dropna().T - _df.dropna().T.min()) /
                        (_df.dropna().T.max() - _df.dropna().T.min())).T
        clustered = linkage(_normalized_df, method='ward', metric='euclidean')

        if _plot==True:
            fig = plt.figure(figsize=(15, 10 / 2))
            ax = fig.add_subplot(1, 1, 1, title="dendrogram")
            plt.xticks(rotation=90)  # JANコードは長くて見づらいので回転させる

        dendrogram(clustered, color_threshold=_threshold *
                max(clustered[:, 2]), labels=_df.index)

        t = _threshold * max(clustered[:, 2])
        c = fcluster(clustered, t, criterion='distance')
        if _plot==True:
            st.pyplot(fig)

        return dict(zip(list(_df.dropna().index), list(c)))


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

    st.write('Cluster Dimension: ' +
            str(max(df_cluster.cluster))
            )
    # Plot by Dendrogram cluster
    normalized_pivot_df = ((pivot_df.T - pivot_df.T.min()) / (pivot_df.T.max() - pivot_df.T.min())).T
    display_by_cluster = lambda d,l,a:[a.append(k) for k,v in d.items() if v==l]


    def plot_line_or_band(_df, _cluster):
        a=[]
        fig2 = plt.figure(figsize=(15, 10 / 2))
        ax2 = fig2.add_subplot(1, 1, 1, title="dendrogram")
        display_by_cluster(cluster_dict, _cluster, a)
        #plt.subplot(int(a),2, 1)
        _df.loc[(a),:].T.plot(figsize=(10, 5))
        st.pyplot(fig2)

    if st.button('Plot by dendrogram cluster'):
        st.subheader('Cluster Plot')
        col2, col3 = st.columns((1, 1))

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

    #受け渡したい変数をstate.~~で入れて、
    # state.dataframe1 = dataframe1
    # state.var1 = var1

def input_page(state):

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

        # Sidebar - Clustering Column selection
        selected_clustering = st.sidebar.selectbox(
            'Select Clustering Column',
            sorted_categoricals,
            index=4)

        # Side bar - rolling period
        selected_rolling_window = st.sidebar.slider(
            'Set a Rolling Windows', 
            0, 
            20, 
            6)

        if st.sidebar.button('Create More Features'):
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

    # Sidebar - 目的変数以外のカラムで、モデル生成に使う特徴量を選ぶ
    # sorted_features = df.columns
    sorted_features = set(df.columns) - set(df.columns[3:4])
    selected_features = st.sidebar.multiselect(
        'Select Features for ML Model',
        sorted_features,
        sorted_features)

    #受け渡したい変数をstate.~~で入れて、
    state.df1 = df
    state.var1 = selected_features
    state.var2 = selected_clustering

def ml_page(state):
    df = state.dataframe1
    selected_features = state.var1
    selected_clustering = state.var2
    st.write(df)
    st.write(selected_features)