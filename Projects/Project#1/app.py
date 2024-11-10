import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import numpy as np

# Page config
st.set_page_config(
    page_title="Sales Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Sales Analytics Dashboard")

# Create sample data
def generate_sample_data():
    data = {
        'Date': pd.date_range(start='2023-01-01', end='2024-03-15', freq='D'),
        'Product': ['Laptop', 'Phone', 'Tablet', 'Watch'] * 109,
        'Category': ['Electronics', 'Electronics', 'Electronics', 'Accessories'] * 109,
        'Sales': [round(abs(np.random.normal(1000, 300)), 2) for _ in range(436)],
        'Units': [int(abs(np.random.normal(50, 20))) for _ in range(436)]
    }
    return pd.DataFrame(data)

# Load data
@st.cache_data
def load_data():
    df = generate_sample_data()
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")

# Date range filter
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(df['Date'].min(), df['Date'].max()),
    min_value=df['Date'].min(),
    max_value=df['Date'].max()
)

# Product filter
products = st.sidebar.multiselect(
    "Select Products",
    options=df['Product'].unique(),
    default=df['Product'].unique()
)

# Filter data
filtered_df = df[
    (df['Date'].dt.date >= date_range[0]) &
    (df['Date'].dt.date <= date_range[1]) &
    (df['Product'].isin(products))
]

# KPI metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_sales = filtered_df['Sales'].sum()
    st.metric("Total Sales", f"${total_sales:,.2f}")

with col2:
    total_units = filtered_df['Units'].sum()
    st.metric("Total Units", f"{total_units:,}")

with col3:
    avg_sale = total_sales / total_units
    st.metric("Average Price per Unit", f"${avg_sale:.2f}")

with col4:
    daily_avg = total_sales / len(filtered_df['Date'].unique())
    st.metric("Daily Average Sales", f"${daily_avg:,.2f}")

# Charts
col1, col2 = st.columns(2)

with col1:
    # Sales trend
    sales_trend = filtered_df.groupby('Date')['Sales'].sum().reset_index()
    fig_trend = px.line(
        sales_trend,
        x='Date',
        y='Sales',
        title='Daily Sales Trend'
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with col2:
    # Product breakdown
    product_sales = filtered_df.groupby('Product')['Sales'].sum().reset_index()
    fig_pie = px.pie(
        product_sales,
        values='Sales',
        names='Product',
        title='Sales by Product'
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# Detailed data
st.subheader("Detailed Data")
st.dataframe(
    filtered_df.sort_values('Date', ascending=False),
    use_container_width=True,
    hide_index=True
)

# Download button
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    "Download Data",
    csv,
    "sales_data.csv",
    "text/csv",
    key='download-csv'
) 