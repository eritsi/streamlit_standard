# app.py

import streamlit_yfinance
import streamlit_sp500
import streamlit_football_app
import streamlit_basketball_app
import streamlit_dna
import streamlit_crypto
import streamlit_iris
import streamlit_boston
import streamlit_penguin
import streamlit_jpx_wiki
import streamlit as st

st.set_page_config(layout="wide")

PAGES = {
    "App1: GOOGL": streamlit_yfinance,
    "App2 SP500": streamlit_sp500,
    "App3 Football": streamlit_football_app,
    "App4 Basketball": streamlit_basketball_app,
    "App5 DNA": streamlit_dna,
    "App6 Crypto": streamlit_crypto,
    "App7 iris": streamlit_iris,
    "App8 Boston": streamlit_boston,
    "App9 Penguin": streamlit_penguin,
    "App10 JPX": streamlit_jpx_wiki
}
st.sidebar.title('Navigation')
st.sidebar.markdown("""
Applications are from [DataProfessor](https://github.com/dataprofessor/streamlit_freecodecamp)
Lectures are here : [Youtube](https://www.youtube.com/watch?v=JwSS70SZdyM)

Some scripts are fixed due to errors by version updates.
""")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
