import pandas as pd
import regex
import unicodedata
from datetime import date

standardize_to_ascii = lambda x: regex.sub(r"\p{Mn}", "", unicodedata.normalize("NFKD", x)) 

def load_order_df(order_file, metadata_file):
    df = pd.read_parquet(order_file)
    metadata_df = pd.read_parquet(metadata_file)[['asin', 'publisher']]
    df = df.merge(metadata_df, on='asin')

    # these should be done before storing the order data
    df['purchase-date'] = pd.to_datetime(df['purchase-date']).dt.date
    df['state'] = df['ship-state']
    df = df[df.state.str.len() == 2]
    df['city_ascii'] = df['ship-city'].str.lower().apply(standardize_to_ascii)
    df = df[df['ship-state'].notna()]
    df['city_state'] = df.apply(lambda x: f'{x.state.lower()}-{x.city_ascii}', axis=1)
    return df.rename(columns={'item-price': 'sales_amount'})

def filter_and_group_df(df, pub, dates):
    df_flt = df[(df['purchase-date'].between(dates[0], dates[1])) & (df.publisher == pub)].copy()
    grp = df_flt.groupby('city_state')[['quantity', 'sales_amount']].sum().reset_index()
    return grp

def load_cities_data(file):
    cities = pd.read_csv(file)
    cities['city_state'] = cities.apply(lambda x: f'{x.state_id.lower()}-{x.city_ascii.lower()}', axis=1)
    cities = cities[['city_state', 'city_ascii', 'state_id', 'lat', 'lng']]
    return cities


df = load_order_df(order_file='data/orders_data_test.parquet', metadata_file='data/metadata_test.parquet')
date1, date2 = date(2024, 11, 1), date(2024, 12, 1)

grp_df = filter_and_group_df(df, 'Taylor & Francis', (date1, date2))
    
cities_df = load_cities_data('data/uscities.csv')
cities_df.to_csv('cities_test.csv', index=False)
grp_df = grp_df.merge(cities_df, on='city_state', how='left', indicator=True)
grp_df[grp_df._merge == 'left_only'].to_csv('left_only.csv', index=False)

