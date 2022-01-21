import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

plt.rcParams.update({'figure.max_open_warning': 0})

def pivot_df_for_dengram(df):
    """ データフレームをピボットする（デンドログラム用）
        Parameters
        ----------
        df : 1列目　: アイテムを特定するid
             2列目　: T1 時系列の年など
             3列目　: T2 時系列の月[1-12]/週[1-53]/日[1-366]など。
             4列目　: yデータ
        Returns
        -------
        pivot_df : T1,T2のマルチインデックスで、列ごとにid(インデックス)が並ぶ。
    """
    pivot_df = df.pivot_table(
        index=df.columns[0],
        columns=[
            df.columns[1],
            df.columns[2]],
        values=df.columns[3])
    return pivot_df


def get_dengram(pivot_df, threshold, plot=True):
    """ データフレームを正規化してデンドログラムにかけ、結果とグラフのfigを返す
        Parameters
        ----------
        pivot_df : データフレーム。時系列の長さが揃っている前提。
              T1,T2のマルチインデックスで、列ごとにid(インデックス)が並ぶ。
        threshold : デンドログラムのカテゴリ分け閾値
        plot : figを生成する・しないを切り替える

        Returns
        -------
        dictionaly : デンドログラムの結果（辞書型）
        fig : デンドログラムのトーナメント図を描いたfig

        Examples
        --------
        >>> from util_ml import get_dengram
        >>> cluster_dict, fig = get_dengram(pivot_df, selected_threshold)
        >>> st.pyplot(fig)

        Remaining to do:
        --------
        # dropna()はDynamic　Warpingの実装を見据えるとイマイチなので直したい
        # 関数の中で正規化してるが、グラフ表示のために関数外でも正規化することになるので無駄かも
        # method='ward'固定は嫌だがとりあえず。
    """
    normalized_df = ((pivot_df.dropna().T - pivot_df.dropna().T.min()) /
                     (pivot_df.dropna().T.max() - pivot_df.dropna().T.min())).T
    clustered = linkage(normalized_df, method='ward', metric='euclidean')

    if plot:
        fig = plt.figure(figsize=(15, 10 / 2))
        ax = fig.add_subplot(1, 1, 1, title="dendrogram")
        plt.xticks(rotation=90)  # JANコードは長くて見づらいので回転させる
    else:
        fig = None

    dendrogram(clustered, color_threshold=threshold *
               max(clustered[:, 2]), labels=pivot_df.index)

    t = threshold * max(clustered[:, 2])
    c = fcluster(clustered, t, criterion='distance')

    return dict(zip(list(pivot_df.dropna().index), list(c))), fig


def add_one_item_in_dendrogram(
        pivot_df_plus_one,
        threshold,
        item_code,
        original_cluster_dict):
    """ デンドログラムにitemを1つ追加し、追加前のクラスタにわりつける
        Parameters
        ----------
        pivot_df_plus_one : データフレーム。pivot_dfにidを1列だけ追加したもの
        threshold : デンドログラムのカテゴリ分け閾値
        item_code : 追加したid
        original_cluster_dict : 元のクラスタ結果

        Returns
        -------
        recommended_original_cluster : 割り付けられた、元のクラスタ番号
        original_clusters : 新しいクラスタ結果で同グループになった他のidが所属していた元のクラスタ番号
        original_cluster_colleague_counts : 上記他のidがそれぞれ何個所属していたか

        Remaining to do:
        --------
        fillna(0)して長さを合わせる　 vs. DTW vs. 今年の値をリピートさせる
    """
    # fillna(0)して長さを合わせる　 vs. DTW vs. 今年の値をリピートさせる
    cluster_dict_plus_one, fig = get_dengram(
        pivot_df_plus_one.fillna(0), threshold, False)
    # これは新しいデンドロでのクラスタ番号
    plus_one_cluster_id = cluster_dict_plus_one[item_code]
    # 新しいデンドロで同じクラスタに入った他のJANを取得
    colleague_keys = [
        k for k,
        v in cluster_dict_plus_one.items() if v == plus_one_cluster_id]
    colleague_keys.remove(item_code)
    # 同僚itemが前のデンドロでどこのクラスタにいたかを取得
    colleague_original_clusters = [
        original_cluster_dict[colleague_keys[i]] for i in range(0, len(colleague_keys))]
    original_clusters, original_cluster_colleague_counts = np.unique(
        colleague_original_clusters, return_counts=True)
    # 複数のクラスタが得られた場合、より多くの同僚itemがいたクラスタを取得
    recommended_original_cluster = original_clusters[[i for i, v in enumerate(
        original_cluster_colleague_counts) if v == max(original_cluster_colleague_counts)][0]]

    print('for item:{} newly entered cluster is {}'.format(
        item_code, recommended_original_cluster))

    return recommended_original_cluster, original_clusters, original_cluster_colleague_counts


def plot_line_or_band(pivot_df, cluster_dict, cluster):
    """ データフレームを正規化してデンドログラムにかけ、結果とグラフのfigを返す
        Parameters
        ----------
        pivot_df : データフレーム。
              T1,T2のマルチインデックスで、列ごとにid(インデックス)が並ぶ。
        cluster_dict : デンドログラムの結果
        cluster : 描画したいクラスタ番号

        Returns
        -------
        fig : 指定のクラスタに所属するidの時系列プロットのfig
    """
    a = []

    display_by_cluster = lambda d,l,a:[a.append(k) for k,v in d.items() if v==l]

    fig = plt.figure(figsize=(15, 10 / 2))
    ax = fig.add_subplot(
        1, 1, 1, title="dendrogram cluster = {}".format(cluster))
    display_by_cluster(cluster_dict, cluster, a)
    pivot_df.loc[(a), :].T.plot(figsize=(10, 5), ax=ax)
    return fig

# import os
# from google.cloud import bigquery
# from google.cloud import bigquery_storage_v1beta1

# '''
# BQデータテーブル読み込み書き込み処理プログラム
# '''


# class datasetLoader(object):

#     # GCPの設定を行う
#     def __init__(self):
#         self.project = os.environ["GCLOUD_PROJECT"]
#         self.bqclient = bigquery.Client(
#             project=self.project, location="asia-northeast1")
#         self.bqstorageclient = bigquery_storage_v1beta1.BigQueryStorageClient()
#         return

#     # SQLを実行し、データフレームへ入れる
#     def __sqlToDataframe(self, in_Lines):

#         # read bq table through bqstorage_client
#         df = (
#             self.bqclient.query(in_Lines)
#             .result()
#             .to_dataframe(
#                 bqstorage_client=self.bqstorageclient
#             )
#         )
#         return df
        
#     # データ読み込み処理
#     def load(self, lines):
#         """SQLを実行し、dfに入れる
#         Parameters
#         ----------
#         lines : SQL文
#         Returns
#         -------
#         df : pandasのdataframeが返る
#         Examples
#         --------
#         >>> import datasetLoader
#         >>> dataset_loader = datasetLoader()
#         >>> sql = '''
#             SELECT
#               *
#             FROM
#               `bigquery-public-data.baseball.schedules`
#           '''
#         >>> df = dataset_loader.load( sql )
#         """
#         whole_dataset = []
#         whole_dataset = self.__sqlToDataframe(lines)

#         return whole_dataset