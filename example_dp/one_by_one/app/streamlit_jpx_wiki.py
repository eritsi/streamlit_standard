import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yfinance as yf

plt.rcParams['font.family'] = 'IPAexGothic'

st.title('JPX App')

st.markdown("""
This app retrieves the list of the **JPX** (from Wikipedia) and its corresponding **stock closing price** (year-to-date)!
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn
* **Data source:** [at_wiki](https://w.atwiki.jp/sysd/pages/1643.html).
""")

st.sidebar.header('User Input Features')

# Web scraping of S&P 500 data
#


@st.cache
def load_data():
    pages = range(1557, 1644)
    df = pd.DataFrame()

    for page in pages:
        url = "https://w.atwiki.jp/sysd/pages/" + str(page) + ".html"
        html = pd.read_html(url, encoding='utf-8')
        downloaded_table = html[0]
        # 1000番台と2000番台以降でテーブル構造(列数・順序)が異なる部分への対応
        if downloaded_table.shape[1] == 6:
            downloaded_table.drop(
                columns=downloaded_table.columns[4],
                inplace=True)
            downloaded_table = downloaded_table.rename(columns={5: 4})
        # 廃業や東証以外の銘柄を無視する
        df = pd.concat([df, downloaded_table], axis=0)
        df = df[df[0].isin(['東証１部', '東証２部', '東証マザ'])]
    df = df.reset_index(
        drop=True).rename(
        columns={
            0: 'JPX_sector',
            1: 'code',
            2: 'company',
            4: 'explanation'})
    df.drop(columns=df.columns[3], inplace=True)
    # 証券コードが規定する業種区分(だいたい・・)
    # 　https://ja.wikipedia.org/wiki/%E8%A8%BC%E5%88%B8%E3%82%B3%E3%83%BC%E3%83%89

    def f_category(x):
        c = int(x.strip('（').strip('）'))
        if c < 1400:
            CAT = '水産・農業'
        elif c < 1500:
            CAT = '住居'
        elif c < 1600:
            CAT = '鉱業'
        elif c < 1700:
            CAT = '鉱業（石油/ガス開発）'
        elif c < 2000:
            CAT = '建設'
        elif c < 3000:
            CAT = '食品'
        elif c < 4000:
            CAT = '繊維・紙'
        elif c < 5000:
            CAT = '化学・薬品'
        elif c < 6000:
            CAT = '資源・素材'
        elif c < 7000:
            CAT = '機械・電機'
        elif c < 8000:
            CAT = '自動車・輸送機'
        elif c < 9000:
            CAT = '金融・商業・不動産'
        else:
            CAT = '運輸・通信・電気・ガス・サービス'
        return CAT
    df['category'] = df['code'].map(f_category)
    # yfinanceが要求するtickersにフォーマットを合わせる
    def f_tickers(x): return '{}.T'.format(x.strip('（').strip('）'))
    df['code'] = df['code'].map(f_tickers)

    return df


df = load_data()
sector = df.groupby('JPX_sector')

# Sidebar - Sector selection
sorted_sector_unique = sorted(df['JPX_sector'].unique())
selected_sector = st.sidebar.multiselect(
    'JPX_sector',
    sorted_sector_unique,
    sorted_sector_unique)

# Sidebar - Category selection
sorted_category_unique = sorted(df['category'].unique())
selected_category = st.sidebar.multiselect(
    'category',
    sorted_category_unique,
    sorted_category_unique)

# Filtering data
df_selected_sector = df[(df['JPX_sector'].isin(selected_sector)) & (df['category'].isin(selected_category))]

st.header('Display Companies in Selected Sector')
st.write('Data Dimension: ' +
         str(df_selected_sector.shape[0]) +
         ' rows and ' +
         str(df_selected_sector.shape[1]) +
         ' columns.')
st.dataframe(df_selected_sector)


# Download JPX data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df):
    csv = df.to_csv(index=False)
    # strings <-> bytes conversions
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="SP500.csv">Download CSV File</a>'
    return href


st.markdown(filedownload(df_selected_sector), unsafe_allow_html=True)

# https://pypi.org/project/yfinance/

data = yf.download(
    tickers=list(df_selected_sector[:10].code),
    period="ytd",
    interval="1d",
    group_by='ticker',
    auto_adjust=True,
    prepost=True,
    threads=True,
    proxy=None
)

# # Plot Closing Price of Query Symbol


def price_plot(code, company):
    df = pd.DataFrame(data[code].Close)
    df['Date'] = df.index
    f, ax = plt.subplots(figsize=(7, 5))
    plt.fill_between(df.Date, df.Close, color='skyblue', alpha=0.3)
    plt.plot(df.Date, df.Close, color='skyblue', alpha=0.8)
    plt.xticks(rotation=90)
    plt.title('{}/{}'.format(code, company), fontweight='bold')
    plt.xlabel('Date', fontweight='bold')
    plt.ylabel('Closing Price', fontweight='bold')
    return st.pyplot(f)


num_company = st.sidebar.slider('Number of Companies', 1, 10)

if st.button('Show Plots'):
    st.header('Stock Closing Price')
    for i, j in zip(
        list(
            df_selected_sector.code)[
            :num_company], list(
                df_selected_sector.company)[
                    :num_company]):
        price_plot(i, j)
