import streamlit as st
import pandas as pd
import geopandas as gpd

import unicodedata
import regex
import os
import time

import json
import plotly.express as px

standardize_to_ascii = lambda x: regex.sub(r"\p{Mn}", "", unicodedata.normalize("NFKD", x)) # replace special letters with closest alphabet


def load_geojson(states):
    states = [i for i in states if f'{i}.json' in os.listdir('data/geojson')] 
    
    df_all = pd.DataFrame()
    temp_json = 'temp_geojson.json'
    for state in states:
        file = f'data/geojson/{state}.json'
        df = gpd.read_file(file)
        df_all = pd.concat([df_all, df], ignore_index=True)

    gpd.GeoDataFrame(df_all).to_file(temp_json)

    with open(temp_json, 'r') as file:
        us_json = json.load(file)
    
    return us_json


df = pd.read_parquet('data/orders_data_test.parquet')
metadata_df = pd.read_parquet('data/metadata_test.parquet')[['asin', 'publisher']]
df = df.merge(metadata_df, on='asin')

# these should be done before storing the order data
df['purchase-date'] = pd.to_datetime(df['purchase-date']).dt.date
df['state'] = df['ship-state']
df = df[df.state.str.len() == 2]
df['city_ascii'] = df['ship-city'].str.lower().apply(standardize_to_ascii)
df = df[df['ship-state'].notna()]
df['city_state'] = df.apply(lambda x: f'{x.state.lower()}-{x.city_ascii}', axis=1)


# filt_asin = st.multiselect('Select ASIN(s)', df.asin.unique())
st.write(f"Data date range: {df['purchase-date'].min()} to {df['purchase-date'].max()}")
filt_supplier = st.radio('Select Publisher', df.publisher.unique())
filt_date = st.date_input('Select date range', ('today', 'today'))

show_map = st.checkbox('Show map')

if show_map:
    if len(filt_date) > 0:
        df_flt = df[(df['purchase-date'].between(filt_date[0], filt_date[1])) & (df.publisher == filt_supplier)]
        grp = df_flt.groupby('city_state')[['quantity', 'item-price']].sum().reset_index()
    
    unique_states = df_flt.state.str.lower().unique()
    time_start = time.time()
    geojson = load_geojson(unique_states)
    st.write(time.time() - time_start)
    
    time_start = time.time()
    fig = px.choropleth_map(grp, geojson=geojson, locations='city_state', featureidkey='properties.city_state', color='item-price',
                           color_continuous_scale="Viridis",
                           zoom=3, center = {"lat": 37.0902, "lon": -95.7129}, hover_data=['quantity'],
                           opacity=0.5,
                          )
    fig['layout']['geo']['subunitcolor']='rgba(0,0,0,0)'
    
    st.plotly_chart(fig)
    st.write(time.time() - time_start)