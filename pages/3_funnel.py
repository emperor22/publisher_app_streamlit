import streamlit as st
import pandas as pd
import numpy as np
from funnel_func import funnel_viz, compare_funnel, bar_line_cluster, df_prepro, df_trend, funnel_data, bar_line_stacked


file_path = 'data/amz_biz_rpt_dec_24.parquet'

df = df_prepro(file_path, 'data/metadata_test.parquet')

st.title('Sales & Traffic Funnel V1')

filter_mode = st.radio('Select filter mode', ['ISBN13', 'Title', 'ASIN'])

filter_col_dct = {'ASIN': 'parent_asin', 'ISBN13': 'isbn13', 'Title': 'title_x'}
filter_col = filter_col_dct[filter_mode]

options = df[filter_col].unique().tolist()
selected_option = st.selectbox(f'Select {filter_mode}:', options, key='funnel')

date_range = df[df[filter_col] == selected_option]['date'].sort_values(ascending=False).dt.strftime('%B-%Y').unique().tolist()
date_selected = st.selectbox('Select Date Range:', date_range)

df_selected = df[(df[filter_col]==selected_option) & (df.month_year == date_selected)]
df_funnel = funnel_data(df_selected)

st.write(funnel_viz(df_funnel))


avg, conversion = st.columns(2)

with avg:
    container = st.container(border=True)
    container.write(f'Avg Units Sold per Order: {df_selected.avg_unit_per_order.values[0]}')

with conversion:
    container = st.container(border=True)
    container.write(f'Sales Conversion Rate: {df_selected.sales_conversion_rate.values[0]}%')

st.write(f'***Orders from other sellers are estimated based on Ayvax’s Buybox% of {int(df_selected.buy_box_percentage.values[0])}%***')

agree = st.checkbox("Compare with Other ASIN")

if agree:
    # options.remove(selected_option1)
    
    col1, col2 = st.columns(2)

    with col1:
        container = st.container(border=True)
        options_col1 = df[filter_col].unique().tolist()
        comp_selected_option1 = container.selectbox(f'Select {filter_mode} to Compare:', options_col1, key='asin1')
        
        date_range = df[df[filter_col] == comp_selected_option1]['date'].dt.strftime('%B-%Y').unique().tolist()
        comp_date_selected1 = container.selectbox('Select Date Range:', date_range, key='date1')
        
        df_selected_col1 = df[(df[filter_col]==comp_selected_option1) & (df.month_year == comp_date_selected1)]
        df_funnel_comp1 = funnel_data(df_selected_col1)

        avg, conversion = st.columns(2)
        with avg:
            container = st.container(border=True)
            container.write(f'Avg Units Sold per Order: {df_selected_col1.avg_unit_per_order.values[0]}')
        with conversion:
            container = st.container(border=True)
            container.write(f'Sales Conversion Rate: {df_selected_col1.sales_conversion_rate.values[0]}%')

    with col2:
        container = st.container(border=True)
        options_col2 = df[filter_col].unique().tolist()
        comp_selected_option2 = container.selectbox(f'Select {filter_mode} to Compare:', options_col2, key='asin2')
        
        date_range = df[df[filter_col] == comp_selected_option2]['date'].sort_values(ascending=False).dt.strftime('%B-%Y').unique().tolist()
        comp_date_selected2 = container.selectbox('Select Date Range:', date_range, key='date2')
        
        df_selected_col2 = df[(df[filter_col]==comp_selected_option2) & (df.month_year == comp_date_selected2)]
        
        avg, conversion = st.columns(2)
        with avg:
            container = st.container(border=True)
            container.write(f'Avg Units Sold per Order: {df_selected_col2.avg_unit_per_order.values[0]}')
        with conversion:
            container = st.container(border=True)
            container.write(f'Sales Conversion Rate: {df_selected_col2.sales_conversion_rate.values[0]}%')
        df_funnel_comp2 = funnel_data(df_selected_col2)
        
    st.write(compare_funnel(df_funnel_comp1, df_funnel_comp2, comp_selected_option1, comp_selected_option2))
    
    col3, col4 = st.columns(2)
    
    with col3:
        container = st.container(border=True)
        container.write(f'**Orders from other sellers are estimated based on Ayvax’s Buybox% of {int(df_selected_col1.buy_box_percentage.values[0])}%**')
        
    with col4:
        container = st.container(border=True)
        container.write(f'**Orders from other sellers are estimated based on Ayvax’s Buybox% of {int(df_selected_col2.buy_box_percentage.values[0])}%**')

st.title('Sales Conversion Trend V1')

trend_asin = st.selectbox(f'Select {filter_mode}:', options, key='trend2')
vis_df = df[df[filter_col] == trend_asin]
date_range = vis_df.month_year.unique().tolist()
st.write('Select Date Range')

col5, col6 = st.columns(2)

with col5:
    container = st.container()
    start_date = container.selectbox('Start Date:', date_range, key='start-1')

with col6:
    container = st.container()
    end_range = date_range[date_range.index(start_date):]
    end_date = container.selectbox('End Date:', end_range, key='end-1')

start = vis_df[vis_df.month_year==start_date]['date'].min()
end = vis_df[vis_df.month_year==end_date]['date'].max()
vis_df = vis_df[(vis_df.date >= start) & (vis_df.date <= end)]

#st.subheader("Stacked", divider=True)
#st.write(bar_line_stacked(vis_df))

#st.subheader("Cluster", divider=True)
st.write(bar_line_cluster(vis_df))


