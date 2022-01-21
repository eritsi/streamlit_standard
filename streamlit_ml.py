import streamlit_lab
import streamlit_lab2
import streamlit_lab3
import streamlit_lab4
import streamlit_lab5

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


st.set_page_config(layout="wide")

PAGES = {
    "App1. クラスタリング": streamlit_lab,
    "App2. データ入力と加工": streamlit_lab2,
    "App3. 学習": streamlit_lab3,
    "App4. 推論と可視化": streamlit_lab4,
    "App5. モデルの比較": streamlit_lab5
}
st.sidebar.title('Navigation')
st.sidebar.markdown("""

""")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
