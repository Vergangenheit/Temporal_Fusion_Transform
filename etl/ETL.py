from sqlalchemy.engine import Engine, Connection
from sqlalchemy import create_engine
from os import getenv
from typing import List, Dict, Optional
import pandas as pd
from pandas import DataFrame, Series, Timestamp
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from numpy import ndarray


def db_connection() -> Engine:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except:
        print('No ".env" file or python-dotenv not installed... Using default env variables...')
    dbname: Optional[str] = getenv('POSTGRES_DB_NAME')
    host: Optional[str] = getenv('POSTGRES_HOST')
    user: Optional[str] = getenv('POSTGRES_USERNAME')
    password: Optional[str] = getenv('POSTGRES_PASSWORD')
    port: Optional[str] = getenv('POSTGRES_PORT')

    postgres_str: str = f'postgresql://{user}:{password}@{host}:{port}/{dbname}'

    engine: Engine = create_engine(postgres_str)

    return engine


def group_hourly(df: DataFrame) -> DataFrame:
    df: DataFrame = df.copy()
    df['day']: Series = df['start_date_utc'].dt.year.astype('str') + '-' + df['start_date_utc'].dt.month.astype(
        'str') + '-' + df[
                            'start_date_utc'].dt.day.astype('str')
    df['day']: Series = pd.to_datetime(df['day'], infer_datetime_format=True)
    grouped: DataFrame = df.groupby(['plant_name_up', 'day', df.start_date_utc.dt.hour]).agg(
        {'kwh': 'mean'})
    grouped: DataFrame = grouped.reset_index(drop=False).rename(columns={'start_date_utc': 'time'})
    #     grouped: DataFrame = grouped.sort_values(by=['plant_name_up', 'day', 'time'], ascending=True, ignore_index=True)
    grouped['time'] = grouped['day'].astype('str') + ' ' + grouped['time'].astype('str') + ':00:00'
    grouped['time'] = grouped['time'].astype('datetime64[ns]')
    grouped: DataFrame = grouped.sort_values(by=['plant_name_up', 'time'], ascending=True, ignore_index=True)
    grouped.drop('day', axis=1, inplace=True)

    return grouped


def extract_weather(weather_sql: str, engine: Engine) -> DataFrame:
    weather_df: DataFrame = pd.read_sql_query(weather_sql, con=engine)
    weather_df['wind_gusts_100m_1h_ms'] = weather_df['wind_gusts_100m_1h_ms'].astype('float64')
    weather_df['wind_gusts_100m_ms'] = weather_df['wind_gusts_100m_ms'].astype('float64')
    weather_df: DataFrame = weather_df.sort_values(by=['timestamp_utc'], ascending=True, ignore_index=True)

    return weather_df
