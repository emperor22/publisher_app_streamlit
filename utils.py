import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

def release_date_gap_score(days):
    years = days / 365
    if years >= 0 and years <= 3:
        return 10
    elif years > 3 and years <= 4:
        return 7.5
    elif years > 4 and years <= 5:
        return 5
    elif years > 5 and years<= 6:
        return 2.5
    else:
        return 0

def ratio_score(ratio):
    if ratio >= 100 and ratio <= 149:
        return 10
    elif ratio >= 150 and ratio <= 199:
        return 8
    elif ratio >= 200 and ratio <= 249:
        return 6
    elif ratio >= 250 and ratio <= 299:
        return 4
    elif ratio >= 300 and ratio <= 349:
        return 2
    else:
        return 0

@st.cache_data
def merge_df_and_get_pubs(hist_file, metadata_file):
    df1 = pd.read_parquet(hist_file)
    df2 = pd.read_parquet(metadata_file)
    
    df1 = df1.dropna(subset=['buybox_new_price', 'lowest_used_price'])
    df1 = df1[df1.buybox_new_price > 0]
    df1['date'] = pd.to_datetime(df1['date'])
    df1 = df1.dropna(subset=['buybox_new_price', 'lowest_used_price'])
    df1['month_year'] = df1['date'].dt.strftime('%B-%Y')
    
    df = df1.merge(df2, on='asin', how='left')
    df = df[df.publishing_date.notna()]
    df = df.dropna(subset=['asin', 'publishing_date', 'list_price', 'buybox_new_price', 'lowest_used_price'])
    pubs = df.publisher.unique().tolist()
    
    return df, pubs


def avg_used_book_score(df):

    df = df.groupby('asin').agg({'publishing_date' : 'first', 
                                  'list_price': 'mean', 
                                  'buybox_new_price': 'mean', 
                                  'lowest_used_price': 'mean'}).round(2).reset_index()
    
    df['release_date_gap'] = (pd.to_datetime('today') - df['publishing_date']).dt.days
    df['ratio'] = df.buybox_new_price / df.lowest_used_price * 100
    # df = df[df.ratio >= 100]
    
    df['used_book_score'] = df.apply(lambda x: np.round((ratio_score(x['ratio']) * 0.25) + (release_date_gap_score(x['release_date_gap']) * 0.75), 2), axis=1)

    df = df[['asin', 'publishing_date', 'list_price', 'buybox_new_price', 'lowest_used_price', 'used_book_score']]
    
    df.rename(columns={'asin': 'ASIN', 
                         'publishing_date': 'Publishing Date', 
                         'list_price': 'Avg List Price', 
                         'buybox_new_price': 'Avg New Price', 
                         'lowest_used_price': 'Avg Used Price', 
                         'used_book_score': 'Avg Used Book Score'}, inplace=True)
    
    df['Publishing Date'] = pd.to_datetime(df['Publishing Date'])
    df['Publishing Date'] = df['Publishing Date'].apply(lambda x: x.date())
    
    return df
    
def df_filter(df, start_date, end_date, pub):
    
    start = df[df.month_year==start_date]['date'].min()
    end = df[df.month_year==end_date]['date'].max()
    df = df[(df.date >= start) & (df.date <= end) & (df.publisher==pub)]
    
    return df

def avg_viz(score):
    score = round(score, 2)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value = score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Average Used Book Score",
               'font': {'color': 'white'}
              },
        gauge={
            'axis': {'range': [0, 10]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 3.5], 'color': 'red'}, 
                {'range': [3.5, 7], 'color': 'orange'},
                {'range': [7, 10], 'color': 'green'}
            ]
        }
    ))
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=0),autosize=False, width=250, height=250)
    return fig

def bar_trend_vis(df, asin):

    df = df[df.asin == asin].groupby('month_year').agg({'publishing_date' : 'first', 
                                  'list_price': 'mean', 
                                  'buybox_new_price': 'mean', 
                                  'lowest_used_price': 'mean'}).reset_index()
    
    df['release_date_gap'] = (pd.to_datetime('today') - df['publishing_date']).dt.days
    df['ratio'] = df.buybox_new_price / df.lowest_used_price * 100
    # df = df[df.ratio >= 100]
    
    df['used_book_score'] = df.apply(lambda x: np.round((ratio_score(x['ratio']) * 0.25) + (release_date_gap_score(x['release_date_gap']) * 0.75), 2), axis=1)
    
    # df_vis = df[df.asin == asin].groupby('month_year')['used_book_score'].mean().round(2).reset_index()
    df['date'] = pd.to_datetime(df['month_year'], format='%B-%Y')
    df = df.sort_values('date')
    fig = px.bar(df, x='month_year', y='used_book_score')

    return fig
