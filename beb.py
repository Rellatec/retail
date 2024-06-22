import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# Load the data
retail_data = pd.read_csv('new_retail_data.csv')

# Navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", [
    'Overview', 
    'Sales by Product Category', 
    'Sales by Country', 
    'Sales by Product Brand', 
    'Customer Feedback Distribution', 
    'Sales by Payment Method', 
    'Order Status Distribution',
    'Sales Forecasting ',
    'Total Purchases Over Time'
])

# Overview
if options == 'Overview':
    st.title("Retail Data Analysis Overview")
    st.write("""
    This analysis focuses on retail data and utilizes a Python-based dashboard to provide insights into various aspects of retail performance. It covers areas such as sales by product category and country, sales by product brand, customer feedback distribution, sales by payment method, order status distribution, sales forecasting using ARIMA modeling, and trends in total purchases over time.
    """)

# Sales by Product Category
elif options == 'Sales by Product Category':
    st.title("Total Sales by Product Category")
    sales_by_category = retail_data.groupby('Product_Category')['Total_Amount'].sum().reset_index()
    fig1 = px.pie(sales_by_category, values='Total_Amount', names='Product_Category', 
                  title='Total Sales by Product Category')
    st.plotly_chart(fig1)

# Sales by Country
elif options == 'Sales by Country':
    st.title("Total Sales by Country")
    sales_by_country = retail_data.groupby('Country')['Total_Amount'].sum().reset_index()
    fig2 = px.scatter_geo(sales_by_country, locations='Country', locationmode='country names', 
                          size='Total_Amount', color='Total_Amount',
                          hover_name='Country', projection='natural earth',
                          title='Total Sales by Country', labels={'Total_Amount': 'Total Sales Amount'},
                          color_continuous_scale=px.colors.sequential.Plasma)
    fig2.update_layout(legend_title_text='Country')
    st.plotly_chart(fig2)

# Sales by Product Brand
elif options == 'Sales by Product Brand':
    st.title("Total Sales by Product Brand ")
    sales_by_brand = retail_data.groupby('Product_Brand')['Total_Amount'].sum().reset_index()
    top_6_brands = sales_by_brand.nlargest(6, 'Total_Amount')
    fig3 = px.bar(top_6_brands, x='Product_Brand', y='Total_Amount', 
                  title='Total Sales by Product Brand (Top 6)', 
                  labels={'Total_Amount': 'Total Sales Amount', 'Product_Brand': 'Product Brand'},
                  color='Product_Brand', color_discrete_map={
                      'Nike': 'blue', 'Adidas': 'green', 'Puma': 'red', 
                      'Reebok': 'orange', 'Under Armour': 'purple', 'Asics': 'brown'
                  })
    st.plotly_chart(fig3)

# Customer Feedback Distribution
elif options == 'Customer Feedback Distribution':
    st.title("Customer Feedback Distribution")
    feedback_distribution = retail_data['Feedback'].value_counts().reset_index()
    feedback_distribution.columns = ['Feedback', 'Count']
    fig4 = px.pie(feedback_distribution, values='Count', names='Feedback', title='Customer Feedback Distribution')
    st.plotly_chart(fig4)

# Sales by Payment Method
elif options == 'Sales by Payment Method':
    st.title("Total Sales by Payment Method")
    sales_by_payment_method = retail_data.groupby('Payment_Method')['Total_Amount'].sum().reset_index()
    fig5 = px.bar(sales_by_payment_method, x='Payment_Method', y='Total_Amount', 
                  title='Total Sales by Payment Method', labels={'Total_Amount': 'Total Sales Amount', 'Payment_Method': 'Payment Method'},
                  color='Payment_Method', color_discrete_map={
                      'Credit Card': 'blue', 'Debit Card': 'green', 'Cash': 'red', 
                      'Online Payment': 'orange', 'Bank Transfer': 'purple'
                  })
    st.plotly_chart(fig5)

# Order Status Distribution
elif options == 'Order Status Distribution':
    st.title("Order Status Distribution")
    order_status_distribution = retail_data['Order_Status'].value_counts().reset_index()
    order_status_distribution.columns = ['Order_Status', 'Count']
    fig6 = px.histogram(order_status_distribution, x='Order_Status', y='Count', 
                        title='Order Status Distribution', labels={'Order_Status': 'Order Status', 'Count': 'Count'},
                        color='Order_Status', color_discrete_sequence=px.colors.qualitative.Set1)
    st.plotly_chart(fig6)

# Sales Forecasting (ARIMA)
elif options == 'Sales Forecasting (ARIMA)':
    st.title("Sales Forecasting (ARIMA)")
    date_column = 'Date'
    retail_data[date_column] = pd.to_datetime(retail_data[date_column], errors='coerce')
    retail_data = retail_data.set_index(date_column)
    monthly_sales = retail_data['Total_Amount'].resample('M').sum()

    # Fit ARIMA model
    model = ARIMA(monthly_sales, order=(5, 1, 0))
    model_fit = model.fit()

    # Forecast
    forecast = model_fit.forecast(steps=12)
    forecast_index = pd.date_range(start=monthly_sales.index[-1], periods=12, freq='M')
    forecast_series = pd.Series(forecast, index=forecast_index)

    fig8 = px.line(monthly_sales, title='Sales Forecasting (ARIMA)')
    fig8.add_scatter(x=forecast_series.index, y=forecast_series, mode='lines', name='Forecast')
    st.plotly_chart(fig8)

# Total Purchases Over Time
elif options == 'Total Purchases Over Time':
    st.title("Total Purchases Over Time by Month")
    date_column = 'Date'
    retail_data[date_column] = pd.to_datetime(retail_data[date_column], errors='coerce')
    retail_data = retail_data.set_index(date_column)
    monthly_purchases = retail_data['Total_Purchases'].resample('M').sum().reset_index()
    fig9 = px.line(monthly_purchases, x='Date', y='Total_Purchases', title='Total Purchases Over Time by Month', 
                   labels={'Date': 'Date', 'Total_Purchases': 'Total Purchases'})
    st.plotly_chart(fig9)
