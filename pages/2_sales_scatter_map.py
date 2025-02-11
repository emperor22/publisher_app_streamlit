import streamlit as st
import pandas as pd
import numpy as np

import unicodedata
import regex
import difflib
from math import sin, cos, sqrt, atan2,radians

import os
import time
from datetime import date, timedelta

import json
import plotly.express as px
import plotly.graph_objects as go

from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype
)


standardize_to_ascii = lambda x: regex.sub(r"\p{Mn}", "", unicodedata.normalize("NFKD", x)) # replace special letters with closest alphabet

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df

@st.cache_data
def get_date_range(file, date_col):
    df = pd.read_parquet(file, columns=[date_col])
    df[date_col] = pd.to_datetime(df[date_col]).dt.date
    return df[date_col].min(), df[date_col].max()

@st.cache_data
def get_pubs(file, col):
    df = pd.read_parquet(file)
    return df[col].unique()

def get_dist(lat1,lon1,lat2,lon2):
  R = 6373.0

  lat1 = radians(lat1)
  lon1 = radians(lon1)
  lat2 = radians(lat2)
  lon2 = radians(lon2)

  dlon = lon2 - lon1
  dlat = lat2 - lat1

  a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
  c = 2 * atan2(sqrt(a), sqrt(1 - a))

  return R * c

def clean_city_col_us_df(df):
    df = df.copy()
    def get_state_id(x):
        state_dct = {'new york': 'ny', 'california': 'ca', 'illinois': 'il', 'florida': 'fl', 'texas': 'tx', 'pennsylvania': 'pa', 'georgia': 'ga', 'district of columbia': 'dc', 'massachusetts': 'ma', 'arizona': 'az', 'washington': 'wa', 'michigan': 'mi', 'minnesota': 'mn', 'colorado': 'co', 'maryland': 'md', 'nevada': 'nv', 'oregon': 'or', 'missouri': 'mo', 'ohio': 'oh', 'indiana': 'in', 'north carolina': 'nc', 'virginia': 'va', 'wisconsin': 'wi', 'rhode island': 'ri', 'utah': 'ut', 'tennessee': 'tn', 'louisiana': 'la', 'kentucky': 'ky', 'oklahoma': 'ok', 'connecticut': 'ct', 'nebraska': 'ne', 'hawaii': 'hi', 'new mexico': 'nm', 'alabama': 'al', 'south carolina': 'sc', 'kansas': 'ks', 'iowa': 'ia', 'arkansas': 'ar', 'idaho': 'id', 'mississippi': 'ms', 'puerto rico': 'pr', 'new jersey': 'nj', 'alaska': 'ak', 'new hampshire': 'nh', 'north dakota': 'nd', 'maine': 'me', 'south dakota': 'sd', 'west virginia': 'wv', 'montana': 'mt', 'delaware': 'de', 'vermont': 'vt', 'wyoming': 'wy'}
        return state_dct.get(x, np.nan) 

    long_state = df.loc[df.state.str.len() > 2, ['state']].drop_duplicates()
    long_state['state'] = long_state.state.str.lower().apply(standardize_to_ascii)
    long_state['new_state_id'] = long_state.state.apply(get_state_id)

    df = df.merge(long_state, on='state', how='left')
    df['state'] = np.where(df.state.str.len() != 2, df.new_state_id, df.state)
    df = df[df.state.notna()]

    df = df.drop('new_state_id', axis=1)

    city_df_replace = {r'^st ': 'saint ',
                        r'^st. ': 'saint ',
                        r' st ': ' saint ',
                        r' st. ': ' saint ',
                        r'^mc ': 'mc',
                        r'township of': '', 
                        r'township': '',
                        r'^twp ': '',
                        r' twp$': '',
                        r'^ft ': 'fort ',
                        r'-': ' ', 
                        r"'(?!s)": " "}

    # (for df cities) if left only and there's cardinal direction, remove cardinal from those cities and join again

    for i, j in city_df_replace.items():
        df['city_ascii'] = df.city_ascii.str.replace(i, j, regex=True)
    return df

def clean_city_col_uk_df(df):
    df = df.copy()
    
    df = df.drop('new_state_id', axis=1)

    city_df_replace = {r'^st ': 'saint ',
                        r'^st. ': 'saint ',
                        r' st ': ' saint ',
                        r' st. ': ' saint ',
                        r'^mc ': 'mc',
                        r'township of': '', 
                        r'township': '',
                        r'^twp ': '',
                        r' twp$': '',
                        r'^ft ': 'fort ',
                        r'-': ' ', 
                        r"'(?!s)": " "}

    # (for df cities) if left only and there's cardinal direction, remove cardinal from those cities and join again

    for i, j in city_df_replace.items():
        df['city_ascii'] = df.city_ascii.str.replace(i, j, regex=True)
    return df

@st.cache_data
def get_uk_cities_dataset(file):
    cities = pd.read_parquet(file)

    standardize_to_ascii = lambda x: regex.sub(r"\p{Mn}", "", unicodedata.normalize("NFKD", x))

    cities['city'] = cities.city.apply(standardize_to_ascii)
    cities['state'] = cities.state.apply(standardize_to_ascii)
    cities['city_state'] = cities.apply(lambda x: f'{x.state.lower()}-{x.city}', axis=1)

    city_filter_words = ['designated place', 'township of', 'town of', 'census', 'township', 'city of', 'county', 'division', 'village of', '\(historical\)']
    city_filter_words = '|'.join(city_filter_words)
    cities = cities.copy()
    cities['city'] = cities.city.str.replace(city_filter_words, '', regex=True).str.strip()


    city_replace_list = {r'-': ' ', 
                        r"'s ": 's ',
                        r"s' ": "s ",
                        r"' ": ' ', 
                        r"'": ' ', 
                        r'[^a-zA-Z ]': '', 
                        r'^st ': 'saint ',
                        r'^st. ': 'saint ',
                        r' st ': ' saint ',
                        r' st. ': ' saint ',}


    for i, j in city_replace_list.items():
        cities['city'] = cities.city.str.replace(i, j, regex=True)
        
    cities['city_state'] = cities.city.str.strip().apply(lambda x: ' '.join(x.split()))
    cities = cities.drop_duplicates(['city_state'])
    return cities


@st.cache_data
def get_us_cities_dataset(file):
    states = ['al', 'ak', 'az', 'ar', 'ca', 'co', 'ct', 'de', 'fl', 'ga', 'hi', 'id', 'il', 'in', 'ia', 'ks', 'ky', 'la', 'me', 'md', 'ma', 
          'mi', 'mn', 'ms', 'mo', 'mt', 'ne', 'nv', 'nh', 'nj', 'nm', 'ny', 'nc', 'nd', 'oh', 'ok', 'or', 'pa', 'ri', 'sc', 'sd', 'tn', 
          'tx', 'ut', 'vt', 'va', 'wa', 'wv', 'wi', 'wy']

    # cities = pd.read_csv('us_cities_map_files/uscities.csv')
    # cities['city_state'] = cities.apply(lambda x: f'{x.state_id.lower()}-{x.city_ascii.lower()}', axis=1)
    # cities = cities[['city_state', 'city_ascii', 'state_id', 'state_name', 'lat', 'lng']]

    cities = pd.read_parquet(file)

    city_filter_words = ['designated place', 'township of', 'town of', 'census', 'township', 'city of', 'county', 'division', 'village of', '\(historical\)']
    city_filter_words = '|'.join(city_filter_words)
    cities = cities.copy()
    cities['city'] = cities.city.str.replace(city_filter_words, '', regex=True).str.strip()


    city_replace_list = {r'-': ' ', 
                         r"'s ": 's ',
                         r"s' ": "s ",
                         r"' ": ' ', 
                         r"'": ' '}

    # add entry

    # dc, Washington
    # fl, Loxahatchee
    # ca, valley village
    # ca, porter ranch
    # ca, west hills
    # nc, willow spring
    # fl, Hallandale
    # ny, cortlandt manor

    for i, j in city_replace_list.items():
        cities['city'] = cities.city.str.replace(i, j, regex=True)

    cities['city'] = cities.city.str.strip().apply(lambda x: ' '.join(x.split()))
    cities = cities.drop_duplicates(['city', 'state_id'])
    cities['city_state'] = cities.apply(lambda x: f'{x.state_id.lower()}-{x.city}', axis=1)
    
    return cities

def get_agg_uk_orders_with_metadata(order_file, metadata_file, date_range, publisher, min_qty):
    orders = pd.read_parquet(order_file)
    metadata = pd.read_parquet(metadata_file)

    # these should be done before storing the order data
    orders['purchase-date'] = pd.to_datetime(orders['purchase-date']).dt.date
    orders['state'] = orders['ship-state'].str.lower().str.strip()
    orders['ship-city'] = orders['ship-city'].str.lower().str.strip()
    
    # filtering pub, dates, and channel
    orders = orders.merge(metadata, on='asin', how='inner')
    date_begin, date_end = date_range
    orders = orders[(orders.publisher == publisher) & (orders['purchase-date'].between(date_begin, date_end))]

    standardize_to_ascii = lambda x: regex.sub(r"\p{Mn}", "", unicodedata.normalize("NFKD", x))
    orders['city_ascii'] = orders['ship-city'].apply(standardize_to_ascii)

    orders = orders[orders['ship-state'].notna()]

    orders = clean_city_col_us_df(orders)

    orders['city_state'] = orders.city_ascii.str.strip().apply(lambda x: ' '.join(x.split())) # doesnt have anything related to state, just to standardize the join column with orders
    orders['city_state'] = orders.city_ascii.str.strip()
    # orders['city_state'] = orders.apply(lambda x: f'{x.state.lower()}-{x.city}', axis=1)
    orders_merge = orders.groupby(['city_state']) \
                   .agg({'quantity': 'sum', 'item-price': 'sum', 'amazon-order-id': 'count'}).reset_index()
    

    orders_merge = orders_merge[orders_merge.quantity >= min_qty]      
    
    
    return orders_merge, orders

def get_agg_us_orders_with_metadata(order_file, metadata_file, date_range, publisher, min_qty):
    orders = pd.read_parquet(order_file)
    metadata = pd.read_parquet(metadata_file)

    # these should be done before storing the order data
    orders['purchase-date'] = pd.to_datetime(orders['purchase-date']).dt.date
    orders['state'] = orders['ship-state'].str.lower().str.strip()
    orders['ship-city'] = orders['ship-city'].str.lower().str.strip()
    
    # filtering pub, dates, and channel
    orders = orders.merge(metadata, on='asin', how='inner')
    date_begin, date_end = date_range
    orders = orders[(orders.publisher == publisher) & (orders['purchase-date'].between(date_begin, date_end))]

    standardize_to_ascii = lambda x: regex.sub(r"\p{Mn}", "", unicodedata.normalize("NFKD", x))
    orders['city_ascii'] = orders['ship-city'].apply(standardize_to_ascii)

    orders = orders[orders['ship-state'].notna()]

    orders = clean_city_col_us_df(orders)

    orders['city'] = orders.city_ascii.str.strip().apply(lambda x: ' '.join(x.split()))
    orders['city'] = orders.city_ascii.str.strip()
    orders['city_state'] = orders.apply(lambda x: f'{x.state.lower()}-{x.city}', axis=1)

    orders_merge = orders.groupby(['state', 'city_state']) \
                   .agg({'quantity': 'sum', 'item-price': 'sum', 'amazon-order-id': 'count'}).reset_index()

    orders_merge = orders_merge[orders_merge.quantity >= min_qty]      
    
    
    return orders_merge, orders

@st.cache_data
def get_us_orders_with_lat_long(order_file, cities_file, metadata_file, date_range, publisher, min_qty=1):
    order_df, raw_orders = get_agg_us_orders_with_metadata(order_file, metadata_file, date_range, publisher, min_qty)
    cities_df = get_us_cities_dataset(cities_file)
    
    mrg = order_df.merge(cities_df, how='inner', on='city_state', indicator=True)
    
    return raw_orders, cities_df, mrg

@st.cache_data
def get_uk_orders_with_lat_long(order_file, cities_file, metadata_file, date_range, publisher, min_qty=1):
    order_df, raw_orders = get_agg_uk_orders_with_metadata(order_file, metadata_file, date_range, publisher, min_qty)
    cities_df = get_uk_cities_dataset(cities_file)
    
    mrg = order_df.merge(cities_df, how='inner', on='city_state', indicator=True)
    
    return raw_orders, cities_df, mrg

@st.cache_data
def load_univ_data_us(file):
    univ = pd.read_excel(file)
    return univ

@st.cache_data
def load_univ_data_uk(file):
    pass


filt_channel = st.radio('Select channel', ['US','UK'])
unique_pubs_dct = {'UK': ['Yale', 'Lippincott', 'Taylor & Francis'], 'US': ['Wiley', 'Lippincott', 'Taylor & Francis']}

filt_supplier = st.radio('Select publisher', unique_pubs_dct[filt_channel])


date_filter_mode = st.radio('Select date filter', ['Last 7 days', 'Last 14 days', 'Last 30 days', 'All', 'Date Range'])

date_now = date(2025, 1, 20) if filt_channel == 'US' else date(2025, 2, 3)

orders_file = 'data/orders_data_test.parquet' if filt_channel == 'US' else 'data/orders_data_test_uk.parquet'
orders_date_range = get_date_range(orders_file, 'purchase-date')

if date_filter_mode == 'Date Range':
    
    st.write(f"Data date range: {orders_date_range[0]} to {orders_date_range[1]}")
    filt_date = st.date_input('Select date range', (date_now, date_now))
else:
    date_dct = {'Last 7 days': date_now-timedelta(days=7), 'Last 14 days': date_now-timedelta(days=14), 'Last 30 days': date_now-timedelta(days=30), 'All': orders_date_range[0]}
    filt_date = (date_dct[date_filter_mode], date_now)
    
qty_filter = st.number_input('Minimum quantity', value=1)

if filt_channel == 'US':
    raw_orders, cities_df, grp_df = get_us_orders_with_lat_long(order_file='data/orders_data_test.parquet', cities_file='data/us_cities_from_census.parquet', metadata_file='data/metadata_test.parquet', 
                                            date_range=filt_date, publisher=filt_supplier, min_qty=qty_filter)
    univ_df = load_univ_data_us('data/us_univ_lat_lng.xlsx')
else:
    raw_orders, cities_df, grp_df = get_uk_orders_with_lat_long(order_file='data/orders_data_test_uk.parquet', cities_file='data/uk_cities_geonames.parquet', metadata_file='data/metadata_test_uk.parquet', 
                                            date_range=filt_date, publisher=filt_supplier, min_qty=qty_filter)

if filt_channel == 'US':
    analyze_in_detail_checkbox = st.checkbox('Analyze by state/city')
    if analyze_in_detail_checkbox:
        city_state_choice = st.radio('Select analysis unit', ['State', 'City'])
        
        df_grp_state = grp_df.copy()
        df_grp_state['state'] = grp_df.city_state.apply(lambda x: x.split('-')[0].upper())
        
        if city_state_choice == 'State':
            df_grp_state = df_grp_state.groupby('state')[['quantity', 'item-price']].sum().reset_index()
            
        if city_state_choice == 'City':
            df_grp_state = df_grp_state[['state', 'city_state', 'quantity', 'item-price']]
            city_analysis_state_filter = st.selectbox('Select State', df_grp_state.state.unique())
            df_grp_state = df_grp_state[df_grp_state.state == city_analysis_state_filter]
                
        df_grp_state['filter'] = [False] * len(df_grp_state)
        df_filt = st.data_editor(df_grp_state)
        df_filt = df_filt[df_filt['filter']]
        
        
        show_item_detail = st.checkbox('Show sales items', help='You have to filter some cities/states first.')
        if show_item_detail and len(df_filt) > 0:
            city_state_filt_col = 'city_state' if city_state_choice == 'City' else 'state'
            city_state_filt = df_filt[city_state_filt_col].str.lower().values
            raw_orders_display = raw_orders[raw_orders[city_state_filt_col].isin(city_state_filt)]
            grp_col_raw_order = ['state', 'asin'] if city_state_choice == 'State' else ['state', 'ship-city', 'asin']
            raw_orders_display = raw_orders_display.groupby(grp_col_raw_order).agg({'title': 'first', 'publishing_date': 'first', 
                                                                                    'item-price': 'sum', 'quantity': 'sum', 'amazon-order-id': lambda x: x.tolist()})
            st.dataframe(filter_dataframe(raw_orders_display))
        

show_map = st.checkbox('Show map')

if show_map and len(filt_date) == 2:
    zoom_map = 2.8 if filt_channel == 'US' else 3.7
    
    center_map = {'center': {'lat': 39.8283, 'lon': -98.5795}} if filt_channel == 'US' else {'center': {'lat': 55.3781, 'lon': -3.4360}}
    if analyze_in_detail_checkbox:
        zoom_map = 5
        center_map = {'center': None}
    if filt_channel == 'US':
        filt_university_check = st.checkbox('Filter by radius to university', help='Make sure to uncheck "Analyze by state/city" to ensure non-conflicting filters')
        if filt_university_check:
            univ_select = st.selectbox('Select University', univ_df.univ_name.unique())
            flt_radius = st.number_input('Select Maximum Radius', value=20)
            
            univ_df_flt = univ_df[univ_df.univ_name == univ_select]
            univ_lat, univ_lng = univ_df_flt.lat.values[0], univ_df_flt.lng.values[0]
            radius_df = cities_df.copy()
            radius_df['distance'] = radius_df.apply(lambda x: get_dist(x.lat, x.lng, univ_lat, univ_lng), axis=1)
            cities_flt_radius = radius_df[radius_df.distance <= flt_radius].city_state.values
            grp_df = grp_df[grp_df.city_state.isin(cities_flt_radius)]
            univ_df = univ_df[univ_df.univ_name == univ_select]
            
            
            show_item_detail = st.checkbox('Show sales items')
            if show_item_detail:
                raw_orders_display = raw_orders[raw_orders['city_state'].isin(cities_flt_radius)]
                raw_orders_display = raw_orders_display.groupby(['state', 'ship-city', 'asin']).agg({'title': 'first', 'publishing_date': 'first', 
                                                                                        'item-price': 'sum', 'quantity': 'sum', 'amazon-order-id': lambda x: x.tolist()})
                st.dataframe(filter_dataframe(raw_orders_display))
            
            zoom_map = 8
            center_map = {'center': None}
            
        if analyze_in_detail_checkbox:
            city_state_filt_col = 'city_state' if city_state_choice == 'City' else 'state'
            
            if len(df_filt) > 0:
                city_state_filt = df_filt[city_state_filt_col].str.lower().values
                grp_df = grp_df[grp_df[city_state_filt_col].isin(city_state_filt)]


    bubble_size_toggle = st.radio('Select metric to determine the bubble size', ['Sales Quantity', 'Sales Amount'])
    color_size_args = {'color': 'item-price', 'size': 'quantity'} if bubble_size_toggle == 'Sales Quantity' else {'color': 'quantity', 'size': 'item-price'}
    
    fig_1 = px.scatter_map(grp_df, lat="lat", lon="lng", hover_name="city_state", range_color=(0, grp_df['item-price'].quantile(0.99)), color_continuous_scale=px.colors.sequential.Agsunset, 
                           size_max=15, zoom=zoom_map, opacity=0.8, map_style='carto-darkmatter', width=1100, height=500 , **color_size_args, **center_map)

    if filt_channel == 'US':
        fig_2 = go.Scattermap(
            mode = "markers",
            lon=univ_df.lng.values, 
            lat=univ_df.lat.values, 
            hovertext=univ_df.univ_name.values,
            marker=dict(size=10, symbol='circle', opacity=0.5, color='lime'), 
            showlegend=False
            )
        
        fig_1.add_trace(fig_2)
    st.plotly_chart(fig_1)
    


# if 'selected_cities' not in ss:
#     ss.selected_cities = None
    
# if 'selected_states' not in ss:
#     ss.selected_states = None

# def on_change_city_filt():
#     ss.selected_cities = ss.filt_city
    
# def on_change_state_filt():
#     ss.selected_states = ss.filt_state

