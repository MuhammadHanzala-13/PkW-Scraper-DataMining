import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- Page Configurations Setup ---
st.set_page_config(page_title="PakWheels Data Dashboard", layout="wide", page_icon="🚗")

# Custom CSS for Beautiful UI
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stMetric {background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.05);}
    h1, h2, h3 {color: #1f2937;}
    </style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_data():
    file_path = "data/pakwheels_cars_processed.csv"
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    return df

df = load_data()

if df is None:
    st.error("🚨 Data file not found! Please run `pakwheels_data_engineering.py` to generate the processed data.")
    st.stop()

# --- Dashboard Header ---
st.title("🚗 Exploratory Data Analysis (EDA) Dashboard")
st.markdown("Automated Analysis & Visualizations for the Artificial Neural Network (ANN) Dataset")
st.markdown("---")

# --- Key Performance Indicators (KPIs) ---
st.header("1. Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Total Vehicles Processed", value=f"{len(df):,}")
with col2:
    st.metric(label="Avg. Price (PKR)", value=f"{int(df['price'].mean()):,}")
with col3:
    st.metric(label="Average Mileage (km)", value=f"{int(df['mileage_km'].mean()):,}")
with col4:
    st.metric(label="Average Car Age", value=f"{round(df['car_age'].mean(), 1)} Years")

st.markdown("---")

# --- Visualizations ---
col_charts1, col_charts2 = st.columns(2)

with col_charts1:
    st.subheader("Distribution of Car Prices (Outliers Removed)")
    fig_price = px.histogram(df, x='price', nbins=50, title="Price Range Frequency", 
                             color_discrete_sequence=['#3b82f6'])
    fig_price.update_xaxes(title="Price (PKR)")
    st.plotly_chart(fig_price, use_container_width=True)

with col_charts2:
    st.subheader("Price vs. Manufacturing Year")
    fig_year = px.scatter(df, x="year", y="price", color="fuel_type", title="Correlation: Year & Price",
                          opacity=0.6, hover_data=["title", "price"])
    st.plotly_chart(fig_year, use_container_width=True)

st.markdown("---")

col_charts3, col_charts4 = st.columns(2)

with col_charts3:
    st.subheader("Top Brands in Dataset")
    if 'brand_clean' in df.columns:
        brand_counts = df['brand_clean'].value_counts().reset_index()
        brand_counts.columns = ['Brand', 'Count']
        fig_brands = px.bar(brand_counts, x='Brand', y='Count', title="Market Share",
                            color='Count', color_continuous_scale="Blues")
        st.plotly_chart(fig_brands, use_container_width=True)
    else:
        st.warning("Brand column missing for this plot.")

with col_charts4:
    st.subheader("Transmission Type Distribution")
    if 'transmission' in df.columns:
        fig_trans = px.pie(df, names='transmission', title="Automatic vs Manual", hole=0.4, 
                           color_discrete_sequence=px.colors.sequential.Teal)
        st.plotly_chart(fig_trans, use_container_width=True)
    else:
        st.warning("Transmission column missing.")

st.markdown("---")

# --- Mathematical Correlation Matrix (Crucial for ANN) ---
st.header("2. Feature Correlation (Mathematical Analysis)")
st.markdown("This heatmap determines which numerical features have the highest impact on **Price**, allowing optimal ANN feature selection.")

# Select only numerical columns for the heatmap
numerical_df = df.select_dtypes(include=['float64', 'int64', 'int32'])
if not numerical_df.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    correlation_matrix = numerical_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, linewidths=0.5)
    st.pyplot(fig)
else:
    st.error("No numerical columns found for correlation.")

st.markdown("---")

# --- Raw Data Inspector ---
st.header("3. Processed Data Inspector")
st.markdown("Explore the raw matrix before it is passed to the Neural Network.")
st.dataframe(df.head(100), use_container_width=True)
