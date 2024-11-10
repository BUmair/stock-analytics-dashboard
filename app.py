import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from textblob import TextBlob
import ta  # Technical Analysis library

# Install required packages:
# pip install yfinance textblob ta pandas-ta

# Page config
st.set_page_config(layout="wide", page_title="Advanced Stock Analytics by Umair Bhalli")

# Custom CSS with black footer
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .creator-header {
        color: #0066cc;
        font-size: 1.2em;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #000000;
        padding: 10px;
        text-align: center;
        border-top: 1px solid #333;
        color: white;
        z-index: 999;
    }
    .footer a {
        color: #00a3ff;
        text-decoration: none;
        margin: 0 10px;
    }
    .footer a:hover {
        color: #0077ff;
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)

# Add creator header
st.markdown('<p class="creator-header">Developed by Umair Bhalli</p>', unsafe_allow_html=True)

# Add this near the top, after the creator header
st.sidebar.header("Stock Selection")
selected_ticker = st.sidebar.selectbox(
    "Choose a Stock",
    options=["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"],
    index=0
)

# Add these filters in the sidebar after the stock selection
st.sidebar.header("Filters")

# Date Range Filter
st.sidebar.subheader("Time Period")
time_period = st.sidebar.selectbox(
    "Select Time Period",
    options=[
        "1h", "2h", "4h", "1d",           # Intraday
        "1wk", "2wk",                      # Weekly
        "1mo", "2mo", "3mo",              # Monthly
        "6mo", "1y", "2y", "5y", "max"    # Longer periods
    ],
    index=3  # Default to 1d
)

# Add this after the time period selector
if time_period in ["1h", "2h", "4h"]:
    st.sidebar.warning("⚠️ Intraday data might be delayed by 15-20 minutes for free API users.")

# Technical Indicator Filter
st.sidebar.subheader("Technical Indicators")
selected_indicators = st.sidebar.multiselect(
    "Select Technical Indicators",
    options=["RSI", "MACD", "Bollinger Bands", "EMA", "SMA", "ATR"],
    default=["RSI", "MACD"]
)

# Volume Filter
show_volume = st.sidebar.checkbox("Show Volume", value=True)

# Candlestick/Line Toggle
chart_type = st.sidebar.radio(
    "Chart Type",
    options=["Candlestick", "Line"],
    index=0
)

# Add this function after the imports
def get_interval(period):
    interval_map = {
        "1h": "1m",    # 1-minute intervals for 1 hour
        "2h": "1m",    # 1-minute intervals for 2 hours
        "4h": "1m",    # 1-minute intervals for 4 hours
        "1d": "1m",    # 1-minute intervals for 1 day
        "1wk": "5m",   # 5-minute intervals for 1 week
        "2wk": "15m",  # 15-minute intervals for 2 weeks
        "1mo": "1h",   # 1-hour intervals for 1 month
        "2mo": "1h",   # 1-hour intervals for 2 months
        "3mo": "1d",   # Daily intervals for 3 months and longer
        "6mo": "1d",
        "1y": "1d",
        "2y": "1d",
        "5y": "1d",
        "max": "1d"
    }
    return interval_map.get(period, "1d")

# Update the load_stock_data function
@st.cache_data(ttl=300)
def load_stock_data(ticker="AAPL", period="1d"):
    stock = yf.Ticker(ticker)
    
    if period in ["1h", "2h", "4h"]:
        df = stock.history(period="1d", interval="1m")
        hours = int(period[0])
        df = df.last(f'{hours}H')
    else:
        interval = get_interval(period)
        df = stock.history(period=period, interval=interval)
    
    info = stock.info
    news = stock.news
    return df, info, news

# Replace the existing data loading with this
df_stock, company_info, news = load_stock_data(selected_ticker, time_period)

# Update the competitors dictionary to be more dynamic
@st.cache_data(ttl=3600)
def get_competitors(ticker):
    competitors_map = {
        'AAPL': {'MSFT': 'Microsoft', 'GOOGL': 'Google', 'SAMSUNG.KS': 'Samsung'},
        'MSFT': {'AAPL': 'Apple', 'GOOGL': 'Google', 'ORCL': 'Oracle'},
        'GOOGL': {'MSFT': 'Microsoft', 'META': 'Meta', 'AMZN': 'Amazon'},
        'AMZN': {'GOOGL': 'Google', 'MSFT': 'Microsoft', 'WMT': 'Walmart'},
        'META': {'GOOGL': 'Google', 'SNAP': 'Snap', 'PINS': 'Pinterest'},
        'NVDA': {'AMD': 'AMD', 'INTC': 'Intel', 'TSM': 'TSMC'},
        'TSLA': {'F': 'Ford', 'GM': 'General Motors', 'TM': 'Toyota'}
    }
    return competitors_map.get(ticker, {'AAPL': 'Apple', 'MSFT': 'Microsoft', 'GOOGL': 'Google'})

# Add this function after get_competitors function
@st.cache_data(ttl=3600)
def load_competitors_data():
    comp_data = {}
    comp_info = {}
    
    # Get data for each competitor
    for comp_ticker in competitors.keys():
        try:
            stock = yf.Ticker(comp_ticker)
            # Use the same time period as the main stock
            comp_data[comp_ticker] = stock.history(period=time_period)
            comp_info[comp_ticker] = stock.info
        except Exception as e:
            st.warning(f"Could not load data for {comp_ticker}: {str(e)}")
            continue
    
    return comp_data, comp_info

# Then use it to load competitor data
competitors = get_competitors(selected_ticker)
comp_data, comp_info = load_competitors_data()

def calculate_technical_indicators(df):
    # Existing indicators
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['BB_upper'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
    df['BB_lower'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
    df['BB_middle'] = ta.volatility.BollingerBands(df['Close']).bollinger_mavg()
    
    # New indicators
    df['EMA_20'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
    df['SMA_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    return df

# Calculate technical indicators
df_stock = calculate_technical_indicators(df_stock)

# Main title and description
st.title(f"🍎 {company_info.get('longName', 'Apple Inc.')} Advanced Analytics Dashboard")

# Tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "Technical Analysis", "Competitor Analysis", 
    "News & Sentiment", "Financial Metrics"
])

with tab1:
    # Overview Section
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = company_info.get('currentPrice', 0)
        previous_price = df_stock['Close'].iloc[-2]
        price_change = ((current_price - previous_price) / previous_price) * 100
        st.metric("Current Price", f"${current_price:,.2f}", f"{price_change:+.2f}%")
    
    with col2:
        market_cap = company_info.get('marketCap', 0)
        st.metric("Market Cap", f"${market_cap/1e9:,.2f}B")
    
    with col3:
        volume = df_stock['Volume'].iloc[-1]
        avg_volume = df_stock['Volume'].mean()
        volume_change = ((volume - avg_volume) / avg_volume) * 100
        st.metric("Volume", f"{volume:,.0f}", f"{volume_change:+.2f}%")
    
    with col4:
        pe_ratio = company_info.get('forwardPE', 0)
        st.metric("Forward P/E", f"{pe_ratio:.2f}")

    # Price Chart with filters
    fig_price = go.Figure()
    
    if chart_type == "Candlestick":
        fig_price.add_trace(go.Candlestick(
            x=df_stock.index,
            open=df_stock['Open'],
            high=df_stock['High'],
            low=df_stock['Low'],
            close=df_stock['Close'],
            name='OHLC'
        ))
    else:
        fig_price.add_trace(go.Scatter(
            x=df_stock.index,
            y=df_stock['Close'],
            name='Price',
            line=dict(color='blue')
        ))
    
    # Add volume if selected
    if show_volume:
        fig_price.add_trace(go.Bar(
            x=df_stock.index,
            y=df_stock['Volume'],
            name='Volume',
            yaxis='y2',
            opacity=0.3
        ))
        
        # Update layout for volume
        fig_price.update_layout(
            yaxis2=dict(
                title="Volume",
                overlaying="y",
                side="right",
                showgrid=False
            )
        )

    # Get the interval for the chart title
    current_interval = get_interval(time_period)

    # Update the chart title
    fig_price.update_layout(
        title=f'{selected_ticker} Stock Price ({time_period} - {current_interval})',
        xaxis_title='Date',
        yaxis_title='Price',
        height=600
    )
    st.plotly_chart(fig_price, use_container_width=True)

with tab2:
    # Technical Analysis Section
    st.header("Technical Analysis")
    
    # Only show selected indicators
    if "RSI" in selected_indicators:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df_stock.index, y=df_stock['RSI'], name='RSI'))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.update_layout(title='Relative Strength Index (RSI)')
        st.plotly_chart(fig_rsi, use_container_width=True)
    
    if "MACD" in selected_indicators:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df_stock.index, y=df_stock['MACD'], name='MACD'))
        fig_macd.update_layout(title='Moving Average Convergence Divergence (MACD)')
        st.plotly_chart(fig_macd, use_container_width=True)
    
    if "Bollinger Bands" in selected_indicators:
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=df_stock.index, y=df_stock['Close'], name='Price'))
        fig_bb.add_trace(go.Scatter(x=df_stock.index, y=df_stock['BB_upper'], name='Upper Band'))
        fig_bb.add_trace(go.Scatter(x=df_stock.index, y=df_stock['BB_lower'], name='Lower Band'))
        fig_bb.update_layout(title='Bollinger Bands')
        st.plotly_chart(fig_bb, use_container_width=True)
    
    if "EMA" in selected_indicators or "SMA" in selected_indicators:
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=df_stock.index, y=df_stock['Close'], name='Price'))
        if "EMA" in selected_indicators:
            fig_ma.add_trace(go.Scatter(x=df_stock.index, y=df_stock['EMA_20'], name='EMA 20'))
        if "SMA" in selected_indicators:
            fig_ma.add_trace(go.Scatter(x=df_stock.index, y=df_stock['SMA_50'], name='SMA 50'))
        fig_ma.update_layout(title='Moving Averages')
        st.plotly_chart(fig_ma, use_container_width=True)
    
    if "ATR" in selected_indicators:
        fig_atr = go.Figure()
        fig_atr.add_trace(go.Scatter(x=df_stock.index, y=df_stock['ATR'], name='ATR'))
        fig_atr.update_layout(title='Average True Range (ATR)')
        st.plotly_chart(fig_atr, use_container_width=True)

with tab3:
    # Competitor Analysis Section
    st.header("Competitor Analysis")
    
    # Normalize prices for comparison
    comp_prices = pd.DataFrame()
    for ticker, name in competitors.items():
        if ticker in comp_data:
            prices = comp_data[ticker]['Close']
            comp_prices[name] = prices / prices.iloc[0] * 100

    fig_comp = px.line(comp_prices, title='Price Performance Comparison (Normalized)')
    st.plotly_chart(fig_comp, use_container_width=True)
    
    # Market Cap Comparison
    market_caps = {}
    for ticker in competitors:
        try:
            stock = yf.Ticker(ticker)
            market_caps[competitors[ticker]] = stock.info.get('marketCap', 0) / 1e9
        except:
            continue
    
    fig_market_cap = px.bar(
        x=list(market_caps.keys()),
        y=list(market_caps.values()),
        title='Market Capitalization Comparison (Billion $)'
    )
    st.plotly_chart(fig_market_cap, use_container_width=True)

with tab4:
    # News & Sentiment Analysis Section
    st.header("News & Sentiment Analysis")
    
    # Sentiment Analysis of News
    sentiments = []
    for article in news:
        if 'title' in article:
            blob = TextBlob(article['title'])
            sentiments.append(blob.sentiment.polarity)
    
    if sentiments:
        avg_sentiment = np.mean(sentiments)
        st.metric(
            "Average News Sentiment", 
            f"{avg_sentiment:.2f}",
            "Positive" if avg_sentiment > 0 else "Negative"
        )
    
    # Display News with Sentiment
    for i, article in enumerate(news[:5]):
        if 'title' in article:
            sentiment = sentiments[i] if i < len(sentiments) else 0
            sentiment_color = "green" if sentiment > 0 else "red" if sentiment < 0 else "gray"
            
            st.markdown(f"""
            <div style="padding: 10px; border-left: 5px solid {sentiment_color};">
                <h4>{article.get('title', 'No Title')}</h4>
                <p>{article.get('description', 'No description available')}</p>
                <a href="{article.get('link', '#')}">Read More</a>
            </div>
            """, unsafe_allow_html=True)

with tab5:
    # Financial Metrics Section
    st.header("Financial Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        metrics = {
            'Revenue': company_info.get('totalRevenue', 0) / 1e9,
            'Gross Profit': company_info.get('grossProfits', 0) / 1e9,
            'Net Income': company_info.get('netIncomeToCommon', 0) / 1e9,
            'Operating Cash Flow': company_info.get('operatingCashflow', 0) / 1e9,
        }
        
        fig_metrics = px.bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            title='Key Financial Metrics (Billion $)'
        )
        st.plotly_chart(fig_metrics, use_container_width=True)
    
    with col2:
        ratios = {
            'P/E Ratio': company_info.get('forwardPE', 0),
            'Price/Book': company_info.get('priceToBook', 0),
            'Debt/Equity': company_info.get('debtToEquity', 0),
            'Current Ratio': company_info.get('currentRatio', 0),
        }
        
        fig_ratios = px.bar(
            x=list(ratios.keys()),
            y=list(ratios.values()),
            title='Financial Ratios'
        )
        st.plotly_chart(fig_ratios, use_container_width=True)

# Download Section
st.header("Export Data")
col1, col2 = st.columns(2)

with col1:
    csv_stock = df_stock.to_csv().encode('utf-8')
    st.download_button(
        "Download Stock Data (CSV)",
        csv_stock,
        "stock_data.csv",
        "text/csv",
        key='download-stock'
    )

with col2:
    # Prepare competitor data for download
    comp_df = pd.DataFrame()
    for ticker, name in competitors.items():
        if ticker in comp_data:
            comp_df[name] = comp_data[ticker]['Close']
    
    csv_comp = comp_df.to_csv().encode('utf-8')
    st.download_button(
        "Download Competitor Data (CSV)",
        csv_comp,
        "competitor_data.csv",
        "text/csv",
        key='download-comp'
    )

# Updated footer with white text and hover effects
st.markdown("""
    <div class="footer">
        <p>© 2024 | Designed and Developed by Umair Bhalli | 
        <a href="https://linkedin.com/in/umair-bhalli" target="_blank">LinkedIn</a> | 
        <a href="https://github.com/ubbhalli" target="_blank">GitHub</a>
        </p>
    </div>
""", unsafe_allow_html=True)