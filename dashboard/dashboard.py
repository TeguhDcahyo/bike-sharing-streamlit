import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from datetime import datetime

sns.set(style='dark')

# Load datasets
day_df = pd.read_csv("Day.csv")
hour_df = pd.read_csv("Hour.csv")

# Data preprocessing
day_df.rename(columns={
    'dteday': 'dateday',
    'yr': 'year',
    'mnth': 'month',
    'weathersit': 'weather',
    'cnt': 'count'
}, inplace=True)

day_df['month'] = day_df['month'].map({
    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
})
day_df['season'] = day_df['season'].map({
    1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'
})
day_df['weather'] = day_df['weather'].map({
    1: 'Clear/Partly Cloudy',
    2: 'Misty/Cloudy',
    3: 'Light Snow/Rain',
    4: 'Severe Weather'
})

# Convert 'dateday' to datetime
day_df['dateday'] = pd.to_datetime(day_df['dateday'])
day_df['season'] = day_df['season'].astype('category')
day_df['year'] = day_df['year'].astype('category')
day_df['month'] = day_df['month'].astype('category')
day_df['weather'] = day_df['weather'].astype('category')

# RFM Analysis
latest_date = day_df['dateday'].max()
day_df['days_since_last_rental'] = (latest_date - day_df['dateday']).dt.days

# Frequency (rentals per month)
day_df['month_year'] = day_df['dateday'].dt.to_period('M')
frequency_df = day_df.groupby('month_year')['count'].count().reset_index()

# Monetary (total rentals per day)
monetary_df = day_df.groupby('dateday')['count'].sum().reset_index()

# Combine RFM data
rfm_df = day_df[['dateday', 'days_since_last_rental', 'count']].copy()
rfm_df = rfm_df.groupby('dateday').agg({
    'days_since_last_rental': 'min',
    'count': ['sum', 'count']
}).reset_index()

rfm_df.columns = ['dateday', 'recency', 'monetary', 'frequency']

# Streamlit Dashboard
st.title("Bike Sharing Data Analysis Dashboard")

# Filter for date range
min_date = day_df['dateday'].min().date()
max_date = day_df['dateday'].max().date()

start_date, end_date = st.sidebar.date_input(
    "Select Date Range",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

filtered_df = day_df[(day_df['dateday'] >= str(start_date)) & (day_df['dateday'] <= str(end_date))]

# Menampilkan statistik dasar
st.header("Basic Statistics")
st.write(filtered_df.describe())

# Total Rental berdasarkan musim
st.subheader("Total Rentals by Season")
season_usage = filtered_df.groupby('season')['count'].sum().reset_index()

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x='season', y='count', data=season_usage, palette='coolwarm', ax=ax)
ax.set_title('Total Bike Rentals by Season')
ax.set_xlabel('Season')
ax.set_ylabel('Total Rentals')
st.pyplot(fig)

# Rental berdasarkan kondisi cuaca
st.subheader("Rentals by Weather Condition")
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x='weather', y='count', data=filtered_df, palette='viridis', ax=ax)
ax.set_title('Box Plot: Weather vs Bike Rentals')
ax.set_xlabel('Weather Condition')
ax.set_ylabel('Total Rentals')
st.pyplot(fig)

month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

day_df['month'] = pd.Categorical(day_df['month'], categories=month_order, ordered=True)

# Mengelompokkan dan menghitung rata-rata penggunaan sepeda per bulan untuk setiap tahun
monthly_usage = day_df.groupby(['year', 'month'])['count'].mean().unstack()

fig, ax = plt.subplots(figsize=(10, 6))
monthly_usage.T.plot(kind='line', marker='o', linestyle='-', ax=ax)
ax.set_title('Average Monthly Bike Rentals (2011 vs 2012)')
ax.set_xlabel('Month')
ax.set_ylabel('Average Rentals')
ax.set_xticks(range(12))
ax.set_xticklabels(monthly_usage.columns, rotation=45)
ax.legend(title='Year')
st.pyplot(fig)

# Visualisasi RFM Analisis
st.subheader("RFM Analysis: Recency vs Monetary vs Frequency")

fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(rfm_df['recency'], rfm_df['monetary'], c=rfm_df['frequency'], cmap='viridis', s=100, alpha=0.7)
plt.colorbar(scatter, ax=ax, label='Frequency')
ax.set_title('RFM Analysis: Recency vs Monetary vs Frequency')
ax.set_xlabel('Recency (Days Since Last Rental)')
ax.set_ylabel('Monetary (Total Rentals)')
st.pyplot(fig)

st.caption('By: Teguh Kukuh Dwi Cahyo')
st.caption('Data sourced from bike rental dataset')
