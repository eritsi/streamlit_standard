import plotly.express as px
import pandas as pd

# Dummy data for USD, EUR, and JPY
df = pd.read_csv('dummy.csv')
df = df[df['Currency'] == 'USD']


def add_MA(df, window=3):
    df['Moving_Average'] = df['Price'].rolling(window).mean()


def add_MACD(df):
    # Calculate the 26-day exponential moving average of the Price column
    ema_26 = df['Price'].ewm(span=26).mean()

    # Calculate the 12-day exponential moving average of the Price column
    ema_12 = df['Price'].ewm(span=12).mean()

    # Calculate the MACD line by subtracting the 26-day EMA from the 12-day EMA
    df['MACD_Line'] = ema_12 - ema_26

    # Calculate the 9-day EMA of the MACD line, which is the signal line
    df['Signal_Line'] = df['MACD_Line'].ewm(span=9).mean()

    # Calculate the histogram by subtracting the signal line from the MACD line
    df['MACD_Histogram'] = df['MACD_Line'] - df['Signal_Line']

    # Reset the index of the dataframe to ensure the MACD data is correctly
    # aligned with the original data
    df = df.reset_index(drop=True)
    return df


df = add_MA(df, 3)
df = add_MACD(df)
df.head()

# Create the plot
fig = px.line(df, x='Date', y='Price', color='Currency')

# Show the plot
fig.show()
