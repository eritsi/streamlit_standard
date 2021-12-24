# This app is for educational purpose only. Insights gained is not
# financial advice. Use at your own risk!
import streamlit as st
from PIL import Image
import pandas as pd
import base64
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests
import json
import time
#---------------------------------#
# New feature (make sure to upgrade your streamlit library)
# pip install --upgrade streamlit

#---------------------------------#
# Page layout
# Page expands to full width
st.set_page_config(layout="wide")
#---------------------------------#
# Title

image = Image.open('crypto-logo.jpg')

st.image(image, width=500)

st.title('Crypto Price App')
st.markdown("""
This app retrieves cryptocurrency prices for the top 100 cryptocurrency from the **CoinMarketCap**!
""")
#---------------------------------#
# About
expander_bar = st.expander("About")
expander_bar.markdown("""
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn, BeautifulSoup, requests, json, time
* **Data source:** [CoinMarketCap](http://coinmarketcap.com).
* **Credit:** Web scraper adapted from the Medium article *[Web Scraping Crypto Prices With Python](https://towardsdatascience.com/web-scraping-crypto-prices-with-python-41072ea5b5bf)* written by [Bryan Feng](https://medium.com/@bryanf).
""")


#---------------------------------#
# Page layout (continued)
# Divide page to 3 columns (col1 = sidebar, col2 and col3 = page contents)
col1 = st.sidebar
col2, col3 = st.columns((2, 1))

#---------------------------------#
# Sidebar + Main panel
col1.header('Input Options')

# Sidebar - Currency price unit
currency_price_unit = col1.selectbox(
    'Select currency for price', ('USD', 'BTC', 'ETH'))

# Web scraping of CoinMarketCap data


@st.cache
def load_data():
    cmc = requests.get('https://coinmarketcap.com')
    soup = BeautifulSoup(cmc.content, 'html.parser')

    data = soup.find('script', id='__NEXT_DATA__', type='application/json')
    coins = {}
    coin_data = json.loads(data.contents[0])
    listings = coin_data['props']['initialState']['cryptocurrency']['listingLatest']['data']
    listings[0]['keysArr'].append('unknown')
    df = pd.DataFrame(listings[2:], columns=listings[0]['keysArr'])
    df = df.filter(items = ['slug',
                            'symbol', 
                            'quote.' + currency_price_unit + '.price',
                            'quote.' + currency_price_unit + '.percentChange1h',
                            'quote.' + currency_price_unit + '.percentChange24h',
                            'quote.' + currency_price_unit + '.percentChange7d',
                            'quote.' + currency_price_unit + '.marketCap',
                            'quote.' + currency_price_unit + '.volume24h']).rename(columns={
                            'slug':'coin_name',
                            'symbol':'coin_symbol', 
                            'quote.' + currency_price_unit + '.price':'market_cap',
                            'quote.' + currency_price_unit + '.percentChange1h':'percent_change_1h',
                            'quote.' + currency_price_unit + '.percentChange24h':'percent_change_24h',
                            'quote.' + currency_price_unit + '.percentChange7d':'percent_change_7d',
                            'quote.' + currency_price_unit + '.marketCap':'price',
                            'quote.' + currency_price_unit + '.volume24h':'volume_24h'})
    return df


df = load_data()

# Sidebar - Cryptocurrency selections
sorted_coin = sorted(df['coin_symbol'])
selected_coin = col1.multiselect('Cryptocurrency', sorted_coin, sorted_coin)

# Filtering data
df_selected_coin = df[(df['coin_symbol'].isin(selected_coin))]

# Sidebar - Number of coins to display
num_coin = col1.slider('Display Top N Coins', 1, 100, 100)
df_coins = df_selected_coin[:num_coin]

# Sidebar - Percent change timeframe
percent_timeframe = col1.selectbox('Percent change time frame',
                                   ['7d', '24h', '1h'])
percent_dict = {
    "7d": 'percent_change_7d',
    "24h": 'percent_change_24h',
    "1h": 'percent_change_1h'}
selected_percent_timeframe = percent_dict[percent_timeframe]

# Sidebar - Sorting values
sort_values = col1.selectbox('Sort values?', ['Yes', 'No'])

col2.subheader('Price Data of Selected Cryptocurrency')
col2.write('Data Dimension: ' +
           str(df_selected_coin.shape[0]) +
           ' rows and ' +
           str(df_selected_coin.shape[1]) +
           ' columns.')

col2.dataframe(df_coins)

# Download CSV data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806


def filedownload(df):
    csv = df.to_csv(index=False)
    # strings <-> bytes conversions
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="crypto.csv">Download CSV File</a>'
    return href


col2.markdown(filedownload(df_selected_coin), unsafe_allow_html=True)

#---------------------------------#
# Preparing data for Bar plot of % Price change
col2.subheader('Table of % Price Change')
df_change = pd.concat([df_coins.coin_symbol,
                       df_coins.percent_change_1h,
                       df_coins.percent_change_24h,
                       df_coins.percent_change_7d],
                      axis=1)
df_change = df_change.set_index('coin_symbol')
df_change['positive_percent_change_1h'] = df_change['percent_change_1h'] > 0
df_change['positive_percent_change_24h'] = df_change['percent_change_24h'] > 0
df_change['positive_percent_change_7d'] = df_change['percent_change_7d'] > 0
col2.dataframe(df_change)

# Conditional creation of Bar plot (time frame)
col3.subheader('Bar plot of % Price Change')

if percent_timeframe == '7d':
    if sort_values == 'Yes':
        df_change = df_change.sort_values(by=['percent_change_7d'])
    col3.write('*7 days period*')
    plt.figure(figsize=(5, 25))
    plt.subplots_adjust(top=1, bottom=0)
    df_change['percent_change_7d'].plot(
        kind='barh', color=df_change.positive_percent_change_7d.map({True: 'g', False: 'r'}))
    col3.pyplot(plt)
elif percent_timeframe == '24h':
    if sort_values == 'Yes':
        df_change = df_change.sort_values(by=['percent_change_24h'])
    col3.write('*24 hour period*')
    plt.figure(figsize=(5, 25))
    plt.subplots_adjust(top=1, bottom=0)
    df_change['percent_change_24h'].plot(
        kind='barh', color=df_change.positive_percent_change_24h.map({True: 'g', False: 'r'}))
    col3.pyplot(plt)
else:
    if sort_values == 'Yes':
        df_change = df_change.sort_values(by=['percent_change_1h'])
    col3.write('*1 hour period*')
    plt.figure(figsize=(5, 25))
    plt.subplots_adjust(top=1, bottom=0)
    df_change['percent_change_1h'].plot(
        kind='barh', color=df_change.positive_percent_change_1h.map({True: 'g', False: 'r'}))
