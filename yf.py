from datetime import timedelta, datetime
import pandas as pd
import time


def get_trading_stocks():
    with open('trading_stocks.txt', 'r') as f:
        my_list = [line.strip() for line in f]
    return my_list


def date_to_num(date):
    # Convert the date string to a datetime object
    date_obj = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    # Convert the datetime object to a Unix timestamp
    timestamp = int(date_obj.timestamp())
    return timestamp


def get_price_day(ticker, date):
    """
    :param etf: the etf name
    :return: download the correspond etf data
    """
    headers = {
        "Cookie": "GUC=AQEBCAFlB1plNEIf7QSG&s=AQAAAKqM-h5f&g=ZQYRcg; A1=d=AQABBISSrmQCEAEraiUCrnTdVEs52ORkrVoFEgEBCAFa"
                  "B2U0Zdwt0iMA_eMBAAcIhJKuZORkrVo&S=AQAAAjuM7Um3Yo-_UABnt-HWGEs; A3=d=AQABBISSrmQCEAEraiUCrnTdVEs52OR"
                  "krVoFEgEBCAFaB2U0Zdwt0iMA_eMBAAcIhJKuZORkrVo&S=AQAAAjuM7Um3Yo-_UABnt-HWGEs; A1S=d=AQABBISSrmQCEAEra"
                  "iUCrnTdVEs52ORkrVoFEgEBCAFaB2U0Zdwt0iMA_eMBAAcIhJKuZORkrVo&S=AQAAAjuM7Um3Yo-_UABnt-HWGEs&j=US; cmp="
                  "t=1694999097&j=0&u=1YNN; gpp=DBAA; gpp_sid=-1; PRF=t%3DSPY%252BXLB%252BGOOG%252BBAC%252BAAPL%26newC"
                  "hartbetateaser%3D1",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0"
                      ".0 Safari/537.36 Edg/116.0.1938.81"
    }
    start = date_to_num(date + ' 06:00:00')
    end = date_to_num(date + ' 22:00:00')
    url = 'https://query1.finance.yahoo.com/v7/finance/download/{}?period1={}&period2={}&' \
          'interval=1d&events=history&includeAdjustedClose=true'.format(ticker, start, end)
    data = pd.read_csv(url)
    if len(data) != 1:
        return None
    else:
        return data
    # requests.get(url, headers=headers).content


def get_price_range(ticker, start, end=(datetime.today()-timedelta(days=1)).strftime('%Y-%m-%d')):
    """
        :param etf: the etf name
        :return: download the correspond etf data
        """
    headers = {
        "Cookie": "GUC=AQEBCAFlB1plNEIf7QSG&s=AQAAAKqM-h5f&g=ZQYRcg; A1=d=AQABBISSrmQCEAEraiUCrnTdVEs52ORkrVoFEgEBCAFa"
                  "B2U0Zdwt0iMA_eMBAAcIhJKuZORkrVo&S=AQAAAjuM7Um3Yo-_UABnt-HWGEs; A3=d=AQABBISSrmQCEAEraiUCrnTdVEs52OR"
                  "krVoFEgEBCAFaB2U0Zdwt0iMA_eMBAAcIhJKuZORkrVo&S=AQAAAjuM7Um3Yo-_UABnt-HWGEs; A1S=d=AQABBISSrmQCEAEra"
                  "iUCrnTdVEs52ORkrVoFEgEBCAFaB2U0Zdwt0iMA_eMBAAcIhJKuZORkrVo&S=AQAAAjuM7Um3Yo-_UABnt-HWGEs&j=US; cmp="
                  "t=1694999097&j=0&u=1YNN; gpp=DBAA; gpp_sid=-1; PRF=t%3DSPY%252BXLB%252BGOOG%252BBAC%252BAAPL%26newC"
                  "hartbetateaser%3D1",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0"
                      ".0 Safari/537.36 Edg/116.0.1938.81"
    }
    start = date_to_num(start + ' 06:00:00')
    end = date_to_num(end + ' 22:00:00')
    url = 'https://query1.finance.yahoo.com/v7/finance/download/{}?period1={}&period2={}&' \
          'interval=1d&events=history&includeAdjustedClose=true'.format(ticker, start, end)
    data = pd.read_csv(url).set_index('Date')
    return data


if __name__ == '__main__':
    # print(date_to_num('2010-01-01 06:00:00'))
    # print(get_price_day('AAPL', '2023-11-21'))
    # print(get_trading_stocks())
    # data = pd.read_pickle('localData')
    # for i in data['dailyLineDict'].keys():
    #     temp = data['dailyLineDict'][i]
    #     data['dailyLineDict'][i] = data['dailyLineDict'][i].loc[[0, 1], :]
    #     data['dailyLineDict'][i] = data['dailyLineDict'][i].iloc[:, :-1]
    trading_stocks = get_trading_stocks()
    data = {}
    data['dailyLineDict'] = {}
    data['minTickDict'] = 0.01
    data['version'] = ['2009-01-01', '2016-01-01', '2016-01-01']
    for s in trading_stocks:
        priceAll = get_price_range(s, '2009-01-01', '2016-01-01')
        data['dailyLineDict'][s] = priceAll
        data['dailyLineDict'][s].index = data['dailyLineDict'][s]['Date']
        data['dailyLineDict'][s] = data['dailyLineDict'][s].iloc[:, 1:]
        print('Still downloading, do not rush haha...')
        time.sleep(1)
    pd.to_pickle(data,'localData')

    # data['version'] = ['2009-01-01', '2016-01-01', '2016-01-01']
    # a = data['dailyLineDict']['ALLE']
    # pd.to_pickle(data, 'localData')
    # print(data['dailyLineDict']['ALLE'])