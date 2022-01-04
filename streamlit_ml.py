import streamlit_lab
import streamlit_lab2
import streamlit_lab3

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

st.set_page_config(layout="wide")

PAGES = {
    "1. クラスタリング": streamlit_lab,
    "2. データ入力と加工": streamlit_lab2,
    "3. 学習": streamlit_lab3,
    # "データ可視化":out_page
}
st.sidebar.title('Navigation')
st.sidebar.markdown("""

Some scripts are fixed due to errors by version updates.
""")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
