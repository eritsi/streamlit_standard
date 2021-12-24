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
This app retrieves the list of the **JPX** nd its corresponding **stock closing price** (year-to-date)!
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn
* **Data source:** [at_JPX_official](https://www.jpx.co.jp/markets/statistics-equities/misc/01.html).
""")

st.sidebar.header('User Input Features')

# Read Downloaded JPX data
#


@st.cache
def load_data():
    df = pd.read_excel('data_j.xls', dtype=str)
    df.drop(columns=df.columns[0], inplace=True)
    df = df.reset_index(
        drop=True).rename(
        columns={
            'コード': 'code',
            '銘柄名': 'company',
            '市場・商品区分': 'JPX_sector',
            '33業種コード': 'sector33_code',
            '33業種区分': 'sector33',
            '17業種コード': 'sector17_code',
            '17業種区分': 'sector17',
            '規模コード': 'index_code',
            '規模区分': 'index'})
    # yfinanceが要求するtickersにフォーマットを合わせる
    def f_tickers(x): return '{}.T'.format(x)
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

# Sidebar - sector33 selection
sorted_sector33_unique = sorted(df['sector33'].unique())
selected_sector33 = st.sidebar.multiselect(
    'sector33',
    sorted_sector33_unique,
    sorted_sector33_unique)

# Sidebar - sector17 selection
sorted_sector17_unique = sorted(df['sector17'].unique())
selected_sector17 = st.sidebar.multiselect(
    'sector17',
    sorted_sector17_unique,
    sorted_sector17_unique)

# Sidebar - index selection
sorted_index_unique = sorted(df['index'].unique())
selected_index = st.sidebar.multiselect(
    'index',
    sorted_index_unique,
    ['TOPIX Core30'])

# Filtering data
df_selected_sector = df[(df['JPX_sector'].isin(selected_sector))
                        & (df['sector33'].isin(selected_sector33))
                        & (df['sector17'].isin(selected_sector17))
                        & (df['index'].isin(selected_index))]

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
    csv = df.to_csv(index=False, encoding='utf-8_sig')
    # strings <-> bytes conversions
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="JPX.csv">Download CSV File</a>'
    return href


st.markdown(filedownload(df_selected_sector), unsafe_allow_html=True)

# https://pypi.org/project/yfinance/

data = yf.download(
    tickers=list(df_selected_sector[:30].code),
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


num_company = st.sidebar.slider('Number of Companies', 1, 30)

if st.button('Show Plots'):
    st.header('Stock Closing Price')
    for i, j in zip(list(df_selected_sector.code)[:num_company],
                    list(df_selected_sector.company)[:num_company]):
        price_plot(i, j)
