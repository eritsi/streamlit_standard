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

from streamlit_google_oauth import google_oauth2_required

st.set_page_config(layout="wide")

# ref. https://zenn.dev/yag_ays/articles/ac982910770010
# streamlit実行前に環境変数に設定が必要
# cloud run で実現するには、Dockerfileに都度記載することを考える
# localまたはgitpodでは実行前に以下3行を実施。先二つは「OAuth 2.0 クライアント ID」から確認可能
# export GOOGLE_CLIENT_ID="464291079551-xxx.apps.googleusercontent.com"
# export GOOGLE_CLIENT_SECRET="GOCSPX--yyy"
# export REDIRECT_URI="https://8501-eritsi-streamlitstandar-zzz.ws-us67.gitpod.io/"
@google_oauth2_required
def main():
    user_id = st.session_state.user_id
    user_email = st.session_state.user_email 
    st.sidebar.write(f"You're logged in {user_id}, {user_email}")

    PAGES = {
        "App1: GOOGL": streamlit_yfinance,
        "App2: SP500": streamlit_sp500,
        "App3: Football": streamlit_football_app,
        "App4: Basketball": streamlit_basketball_app,
        "App5: DNA": streamlit_dna,
        "App6: Crypto": streamlit_crypto,
        "App7: iris": streamlit_iris,
        "App8: Boston": streamlit_boston,
        "App9: Penguin": streamlit_penguin,
        "App10: JPX": streamlit_jpx_wiki
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

main()