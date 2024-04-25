import pandas_market_calendars as mcal
import pandas as pd
from datetime import timedelta, datetime
import random


def get_trading_dates(start, end=(datetime.today()-timedelta(days=1)).strftime('%Y-%m-%d')):
    nyse = mcal.get_calendar('NYSE')
    early = nyse.schedule(start_date=start, end_date=end)
    dates = mcal.date_range(early, frequency='1D')
    dates = [date.strftime('%Y-%m-%d') for date in dates]
    return dates


def get_next_trading_date(date):
    nyse = mcal.get_calendar('NYSE')
    end = pd.to_datetime(date)
    while True:
        end = end + timedelta(days=1)
        early = nyse.schedule(start_date=date, end_date=end.strftime('%Y-%m-%d'))
        temp = mcal.date_range(early, frequency='1D')
        if len(temp) == 2:
            break
    return temp[1].strftime('%Y-%m-%d')


def get_previous_trading_date(date):
    nyse = mcal.get_calendar('NYSE')
    start = pd.to_datetime(date)
    while True:
        start = start - timedelta(days=1)
        early = nyse.schedule(start_date=start.strftime('%Y-%m-%d'), end_date=date)
        temp = mcal.date_range(early, frequency='1D')
        if len(temp) == 2:
            break
    return temp[0].strftime('%Y-%m-%d')


if __name__ == '__main__':
    # print((datetime.today()-timedelta(days=1)).strftime('%Y-%m-%d'))
    # time_range = get_trading_dates('2019-01-01', '2020-01-01')
    # stocks = ['ALLE', 'CLX', 'CMG', 'DAL']
    # index = pd.MultiIndex.from_product([time_range, stocks])
    # data = pd.DataFrame({'position': [random.randint(0, 1) for _ in range(len(time_range)*4)]}, index=index)
    # data = data/4
    # data.to_pickle('sample.pkl')
    # data = pd.read_pickle('sample.pkl')
    print(get_trading_dates('2020-10-08', '2020-10-25'))
    print(1)