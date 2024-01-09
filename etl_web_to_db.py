from pathlib import Path 
import pandas as pd
# from prefect import flow, task
# from prefect_gcp.cloud_storage import GcsBucket
# from prefect.tasks import task_input_hash
from datetime import timedelta
import pandas as pd 
from sodapy import Socrata
from sqlalchemy import create_engine
from time import time
import argparse
import os
import gzip
import shutil
import ast
import random
import numpy as np
import hashlib
import operator
from functools import reduce


def fetch(month: int, year : int) -> pd.DataFrame:
    """Read data from web into pandas Dataframe"""

    client = Socrata("data.iowa.gov", 'CDd3r7gkyMxtPzaQeM5aXnono', timeout=10000)

    print(f'Reading data for the month of {month} and year of {year}')

    results = client.get("m3tr-qhgy", order= "date desc", where = f'date_extract_m(date) = {month} and date_extract_y(date) = {year}', limit = 500000 )

    df = pd.DataFrame.from_records(results)
    print(f"Dataframe is fetched with {df.shape[0]} rows")

    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix some Dtype issues"""

    print(df.info())

    # keeping necessary columns
    print(f'keeping necessary columns')
    df = df[['invoice_line_no', 'date', 'store', 'name', 'address',
       'city', 'zipcode', 'county', 'category',
       'category_name', 'vendor_no', 'vendor_name', 'itemno', 'im_desc',
       'pack', 'bottle_volume_ml', 'state_bottle_cost', 'state_bottle_retail',
       'sale_bottles', 'sale_dollars', 'sale_liters', 'sale_gallons']]
    
    # Handling null values
    print(f'Handling null values')

    if reduce(operator.or_,df.vendor_no.isna()):
        df.dropna(subset=['vendor_no'],inplace=True)
    
    if reduce(operator.or_,df.itemno.isna()):
        df.dropna(subset=['itemno'],inplace=True)

    rest_flag = False
    try:
        df['address'] = df.groupby(['name'], sort=False)['address'].apply(lambda x: x.fillna(x.mode().iat[0]))
        rest_flag = True
    except IndexError:
        df.dropna(subset=['city','zipcode', 'county'],inplace=True)
    if(rest_flag):
        df['city'] = df.groupby(['address'], sort=False)['city'].apply(lambda x: x.fillna(x.mode().iat[0]))
        df['zipcode'] = df.groupby(['city'], sort=False)['zipcode'].apply(lambda x: x.fillna(x.mode().iat[0]))
        df['county'] = df.groupby(['city','zipcode'], sort=False)['county'].apply(lambda x: x.fillna(x.mode().iat[0]))

    rest_flag = False
    try:
        df['category'] = df.groupby(['im_desc'], sort=False)['category'].apply(lambda x: x.fillna(x.mode().iat[0]))
        rest_flag = True
    except IndexError:
        df.dropna(subset=['category_name'],inplace=True)
    if rest_flag:
        df['category_name'] = df.groupby(['category'], sort=False)['category_name'].apply(lambda x: x.fillna(x.mode().iat[0]))
    
    
    # Fixing datatype issues
    print(f'Fixing datatype issues')
    convert_dict = { 'store':int,'vendor_no':int,'itemno':int,'pack':int, 'bottle_volume_ml':float, 'state_bottle_cost':float, 'state_bottle_retail':float,
       'sale_bottles':int, 'sale_dollars':float, 'sale_liters':float, 'sale_gallons':float}
    df = df.astype(convert_dict)
    df.date = pd.to_datetime(df.date)
    df.name = df.name.str.split('/').str[0]
    df['category'] = pd.to_numeric(df['category']).astype(int)


    print(df.info())
    return df

def transform(df: pd.DataFrame) -> dict:
    """ Data Cleaning"""

    # get store DF
    name = df.name.unique().tolist()
    store_df = pd.DataFrame({'name':name})
    store_df = store_df.merge(df[['name','address','city','zipcode','county']],on='name', how = 'left').drop_duplicates(subset=['name','address']).reset_index(drop=True)
    store_df['store_id'] = store_df[['name', 'address']].sum(axis=1).map(hash)
    print(f'Store Dataframe: {store_df}')

    # get vendor DF
    vendor_number = df.vendor_no.unique().tolist()
    vendor_df = pd.DataFrame({'vendor_no':vendor_number})
    vendor_df = vendor_df.merge(df[['vendor_no','vendor_name']], on='vendor_no', how = 'left').drop_duplicates(subset=['vendor_no']).reset_index(drop=True)
    print(vendor_df.head(2))

    # get item df
    itemno = df.itemno.unique().tolist()
    item_df = pd.DataFrame({'itemno':itemno})
    item_df = item_df.merge(df[['itemno','im_desc']], on='itemno', how = 'left').drop_duplicates(subset=['itemno']).reset_index(drop=True)
    print(item_df.head(2))

    # get category df
    category = df.category.unique().tolist()
    category_df = pd.DataFrame({'category':category})
    category_df = category_df.merge(df[['category','category_name']], on='category', how = 'left').drop_duplicates(subset=['category']).reset_index(drop=True)
    category_df.rename(columns={'category':'category_no'}, inplace=True)
    print(category_df.head(2))

    # get facts df
    df = df.merge(store_df[['store_id','name','address']], on = ['name','address'], how = 'left')
    df.rename(columns={'category':'category_no'}, inplace=True)
    facts_df = df[['invoice_line_no','date','store_id','vendor_no','itemno','category_no','pack','pack', 'bottle_volume_ml',
       'state_bottle_cost', 'state_bottle_retail', 'sale_bottles',
       'sale_dollars', 'sale_liters', 'sale_gallons']]

    print(facts_df.head(2))


    return {'tbl_dim_store':store_df, 'tbl_dim_vendor': vendor_df, 'tbl_dim_item': item_df, 'tbl_dim_category': category_df, 'tbl_fact_sales': facts_df}


def write_db(df_dict : dict)-> None:
    ''''writes dataframes to db'''

    print(f'Connecting to DB')
    engine = create_engine('postgresql://root:root@localhost:5433/iowa_alcohol_sales')
    table_names = ['tbl_fact_sales','tbl_dim_category','tbl_dim_item','tbl_dim_vendor','tbl_dim_store']

    for i in table_names:
        if not engine.dialect.has_table(engine.connect(), i):
            df_dict[i].head(n=0).to_sql(name=i, con=engine, if_exists='replace')

    print(f'Added Dataframes to db')
    # Put Fact tables into db
    df_dict['tbl_fact_sales'].to_sql(name='tbl_fact_sales', con=engine, if_exists='append')
    # Dedupe values
    query1 = '''
        DELETE FROM tbl_fact_sales T1 
            USING tbl_fact_sales T2
        WHERE T1.ctid < T2.ctid       -- delete the "older" ones
        AND T1.invoice_line_no = T2.invoice_line_no;       -- list columns that define duplicates
        '''
    with engine.connect() as con:
        con.execute(query1)

    # Put Category Value into db
    df_dict['tbl_dim_category'].to_sql(name='tbl_dim_category', con=engine, if_exists='append')
    # Dedupe values
    query2 = '''
        DELETE FROM tbl_dim_category T1 
            USING tbl_dim_category T2
        WHERE T1.ctid < T2.ctid       -- delete the "older" ones
        AND T1.category_no = T2.category_no;       -- list columns that define duplicates
        '''
    with engine.connect() as con:
        con.execute(query2)
    
    # Put items df into db
    df_dict['tbl_dim_item'].to_sql(name='tbl_dim_item', con=engine, if_exists='append')

    query3 = '''
        DELETE FROM tbl_dim_item T1 
            USING tbl_dim_item T2
        WHERE T1.ctid < T2.ctid       -- delete the "older" ones
        AND T1.itemno = T2.itemno;       -- list columns that define duplicates
        '''
    with engine.connect() as con:
        con.execute(query3)

    # put vendor df into db
    df_dict['tbl_dim_vendor'].to_sql(name='tbl_dim_vendor', con=engine, if_exists='append')
    # dedupe values
    query4 = '''
    DELETE FROM tbl_dim_vendor T1 
        USING tbl_dim_vendor T2
    WHERE T1.ctid < T2.ctid       -- delete the "older" ones
    AND T1.vendor_no = T2.vendor_no;       -- list columns that define duplicates
    '''
    with engine.connect() as con:
        con.execute(query4)

    # put store df into db
    df_dict['tbl_dim_store'].to_sql(name='tbl_dim_store', con=engine, if_exists='append')
    # dedupe
    query1 = '''
    DELETE FROM tbl_dim_store T1 
        USING tbl_dim_store T2
    WHERE T1.ctid < T2.ctid       -- delete the "older" ones
    AND T1.store_id = T2.store_id;       -- list columns that define duplicates
    '''
    with engine.connect() as con:
        con.execute(query1)

    print(f'Finished adding dataframes to db')


def etl_web_to_db(year: int, month: int) -> None:
    """The main ETL Function"""


    df = fetch(month, year)
    df = clean(df)
    df_dict = transform(df)
    write_db(df_dict)

def etl_web_to_db_parent_flow(
    months: list = [1, 2], years: list = [2021]
):
    for year in years:
        for month in months:
            etl_web_to_db(year, month)


if __name__ == "__main__":

    months = [1,2,3,4,5,6,7,8,9,10,11,12]
    year = [2018,2019]
    etl_web_to_db_parent_flow(months, year)