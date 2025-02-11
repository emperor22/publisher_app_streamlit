import streamlit as st
import pandas as pd
import numpy as np
from utils import merge_df_and_get_pubs, avg_viz, bar_trend_vis, df_filter, avg_used_book_score



df_us, pubs_us = merge_df_and_get_pubs('data/hist_data_test.parquet', 'data/metadata_test.parquet')
df_uk, pubs_uk = merge_df_and_get_pubs('data/hist_data_test_uk.parquet', 'data/metadata_test_uk.parquet')
# df = pd.read_csv('data_last_6_months.csv')
# df['date'] = pd.to_datetime(df['date'])

tab1, tab2= st.tabs(["tab 1", "tab 2"])



ratio_data = {'ratio': ['100%-149%', '150%-199%', '200%-249%', '250%-299%', '300%-349% ', '>350%'],
        'score': [10, 8, 6, 4, 2, 0]}
pr_df = pd.DataFrame(ratio_data)

release_date_gap_data = {'gap': ['0-3', '3-4', '4-5', '5-6', '>6 '],
        'score': [10, 7.5, 5, 2.5, 0]}
rdg_df = pd.DataFrame(release_date_gap_data)

with tab1:
    tab1_check = st.checkbox('Show aggregate Used Score')
    if tab1_check:
        channel_flt = st.radio('Select Channel', ['US', 'UK'])
        df = df_us if channel_flt == 'US' else df_uk
        options = pubs_us if channel_flt == 'US' else pubs_uk
        date_range = df.sort_values('date').month_year.unique().tolist()
        st.write('Select Date Range')

        col3, col4 = st.columns(2)

        with col3:
            container = st.container()
            start_date = container.selectbox('Start Date:', date_range, key='start-1')

        with col4:
            container = st.container()
            end_range = date_range[date_range.index(start_date):]
            end_date = container.selectbox('End Date:', end_range, key='end-1')


        st.header("Total Used Score Across All Books by Publisher")

        pub1 = st.radio(
            "Select Publisher",
            options,
            index=0,
            key='tab1'
        )
        container = st.container()
        df_tab1 = df_filter(df, start_date, end_date, pub1)
        st.write(f'Date Range: {df_tab1.date.dt.date.min()} - {df_tab1.date.dt.date.max()}')
        df_tab1 = avg_used_book_score(df_tab1)
        # df_tab_1 = df_tab_1[['asin', 'publishing_date', 'list_price', 'buybox_new_price','lowest_used_price', 'used_book_score']]
        avg_score = df_tab1['Avg Used Book Score'].mean()
        
        container.write(avg_viz(avg_score))

        st.subheader(f"Top 10 Highest Score by {pub1}", divider=True)
        st.write(df_tab1.sort_values('Avg Used Book Score', ascending=False).head(10).reset_index(drop=True))

        st.subheader(f"Top 10 Lowest Score by {pub1}", divider=True)
        st.write(df_tab1.sort_values('Avg Used Book Score').head(10).reset_index(drop=True))

        col1, col2 = st.columns(2)

        with col1:
            container = st.container(border=True, height=330)
            container.write('New/Used Price Ratio Scoring:')
            container.table(pr_df)

        with col2:
            container = st.container(border=True, height=330)
            container.write('Release Date Gap Scoring (in Years)')
            container.table(rdg_df)


with tab2:
    tab2_check = st.checkbox('Show Used Score and Used Score Trend per Title')
    if tab2_check:
        channel_flt_2 = st.radio('Select Channel', ['US', 'UK'], key='channel2')
        df = df_us if channel_flt_2 == 'US' else df_uk
        date_range = df.sort_values('date').month_year.unique().tolist()
        st.write('Select Date Range')

        col5, col6 = st.columns(2)

        with col5:
            container = st.container()
            start_date = container.selectbox('Start Date:', date_range, key='start-2')

        with col6:
            container = st.container()
            end_range = date_range[date_range.index(start_date):]
            end_date = container.selectbox('End Date:', end_range, key='end-2')
        
        st.header("Total Used Score per Title")
        options = pubs_us if channel_flt_2 == 'US' else pubs_uk
        pub2 = st.radio(
                    "Select Publisher",
                    options,
                    index=0,
                    key='tab2'
                )

        df_tab2_TUSpT = df_filter(df, start_date, end_date, pub2)
        st.write(f'Date Range: {df_tab2_TUSpT.date.dt.date.min()} - {df_tab2_TUSpT.date.dt.date.max()}')
        asin_list = df_tab2_TUSpT.asin.unique().tolist()
        
        asins = st.multiselect(
                    "Select Asin",
                    asin_list
                )

        df_tab2_TUSpT = df_tab2_TUSpT[df_tab2_TUSpT['asin'].isin(asins)]
        df_tab2_TUSpT = avg_used_book_score(df_tab2_TUSpT)
        
        st.write('Score of Books Selected: ')
        st.write(df_tab2_TUSpT)

        col1, col2 = st.columns(2)

        with col1:
            container = st.container(border=True, height=330)
            container.write('New/Used Price Ratio Scoring:')
            container.table(pr_df)

        with col2:
            container = st.container(border=True, height=330)
            container.write('Release Date Gap Scoring (in Years)')
            container.table(rdg_df)


        st.header("Trend Analysis")
        date_range = df.sort_values('date').month_year.unique().tolist()
        st.write('Select Date Range')

        col3, col4 = st.columns(2)

        with col3:
            container = st.container()
            start_date = container.selectbox('Start Date:', date_range)

        with col4:
            container = st.container()
            end_range = date_range[date_range.index(start_date):]
            end_date = container.selectbox('End Date:', end_range)

        pub3 = st.radio(
                    "Select Publisher",
                    options,
                    index=0,
                    key='tab2-trend'
                )

        df_tab2_TA = df_filter(df, start_date, end_date, pub3)
        
        asins = df_tab2_TA.asin.unique().tolist()
        asin = st.selectbox('Select ASIN:', asins)
        
        st.write(bar_trend_vis(df_tab2_TA, asin))
        
