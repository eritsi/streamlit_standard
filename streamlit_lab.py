import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

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
|  JAN等  |  Year等  |  Month,Week等  |  目的変数  |  

""")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.DataFrame()

# Displays the user input features
st.subheader('User Input')
st.write('Data Dimension: ' +
         str(len(df.iloc[:,0].unique())) +
         ' items and data from ' +
         str(min(df.iloc[:,1]))  +
         '. ' +
         str(min(df[(df.iloc[:,1]==min(df.iloc[:,1]))].iloc[:,2])) +
         ' to ' +
         str(max(df.iloc[:,1]))  +
         '. ' +
         str(max(df[(df.iloc[:,1]==max(df.iloc[:,1]))].iloc[:,2])) +
         '. It means ' +
         str(max(df.groupby(df.columns[0]).size())) +
         ' ticks and ' +
         str(sum(df.groupby(df.columns[0]).size()==max(df.groupby(df.columns[0]).size()))) +
         ' items have full-ticks time-history data.'
)

if uploaded_file is not None:
    st.write(df)
else:
    st.write(
        'Awaiting CSV file to be uploaded.')
    st.write(df)

# Select items with the longest time-history
st.subheader('Clustering Input')

if uploaded_file is not None:
    df_clustering_input=(df.iloc[:,0].value_counts()==max(df.groupby(df.columns[0]).size()))
    df_clustering_input = df[(df.iloc[:,0]).isin(df_clustering_input[df_clustering_input==True].index)]
    df_clustering_input = df_clustering_input.reset_index(drop=True)
    st.write('Data Dimension: ' +
         str(len(df_clustering_input.iloc[:,0].unique())) +
         ' items.'
    )
    st.write(df_clustering_input)
else:
    st.write(
        'Awaiting CSV file to be uploaded.')      

# Dendrogram
st.subheader('Dendrogram Output')
selected_threshold = st.sidebar.slider('Dendrogram threshold', 0.0, 1.0, 0.17)

## dropna()はDynamic　Warpingの実装を見据えるとイマイチなので直したい
## 関数の中で正規化してるが、表示のために関数外でも正規化することになるので無駄っぽい
## method='ward'固定は嫌だがとりあえず。

def get_dengram(_df, _threshold):
    _normalized_df = ((_df.dropna().T - _df.dropna().T.min()) / (_df.dropna().T.max() - _df.dropna().T.min())).T
    clustered = linkage(_normalized_df,method='ward',metric='euclidean')
    
    fig = plt.figure(figsize=(15, 10/2))
    ax = fig.add_subplot(1, 1, 1, title="dendrogram")
    dendrogram(clustered, color_threshold=_threshold*max(clustered[:,2]), labels=_df.index)
    plt.xticks(rotation=90) # JANコードは長くて見づらいので回転させる
    
    t = _threshold*max(clustered[:,2])
    c = fcluster(clustered, t, criterion='distance')
    
    return dict(zip(list(_df.dropna().index),list(c))), st.pyplot(fig)

pivot_df = df_clustering_input.pivot_table(index=df.columns[0], columns=[df.columns[1], df.columns[2]], values=df.columns[3])
cluster_dict = get_dengram(pivot_df, selected_threshold)
