import pandas as pd


def moving_average(data, n):
    ma = data.rolling(n).mean()
    # if normalized:
    #     ma = (ma - ma.min()) / (ma.max() - ma.min())
    # data['MA' + str(n)] = ma
    return ma


def e_moving_average(data, n):
    ema = data.ewm(span=n, adjust=False).mean()
    # if normalized:
    #     ema = (ema - ema.min()) / (ema.max() - ema.min())
    # data['EMA' + str(n)] = ema
    return ema


def n_day_mom(data, n):
    mom = data - data.shift(n)
    # if normalized:
    #     mom = (mom-mom.min()) / (mom.max() - mom.min())
    return mom


def n_day_rsi(data, n):
    # calculate the price change over the look-back period.
    delta = data.diff().dropna()
    # calculate the gain and loss over the look-back period
    # gain = delta.where(delta > 0, 0)
    # loss = -delta.where(delta < 0, 0)
    gain, loss = delta.copy(), delta.copy()
    gain[gain<0] = 0
    loss[loss>0] = 0

    # calculate the average gain and loss over the look-back period.
    avg_gain = gain.rolling(window=n).mean()
    avg_loss = loss.rolling(window=n).mean().abs()

    # calculate the relative strengh over the look-back period.
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    # name = 'RSI' + str(n)
    # if normalized:
    #     rsi = (rsi-rsi.min()) / (rsi.max() - rsi.min())
    # data[name] = pd.concat([pd.Series([np.nan] * (n+1)), rsi], ignore_index=True)
    return rsi.shift()


def macd(data):
    ema12 = data.ewm(span=12, adjust=False).mean()
    ema26 = data.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    # if normalized:
    #     macd = (macd - macd.min()) / (macd.max() - macd.min())
    # data['MACD'] = macd
    return macd

    # return pd.DataFrame({'macd': macd, 'signal': signal, 'histogram': histogram})


if __name__ == '__main__':
    df = pd.read_excel('OMC UN Equity.xlsx')[::-1]
    # df['MA10'] = df['Open'].rolling(10).mean()
    # df['MA30'] = df['Open'].rolling(30).mean()
    # df['MA200'] = df['Open'].rolling(200).mean()
    # df['EMA10'] = df['Open'].ewm(span=10)
    df['EMA30'] = df['Open'].ewm(span=30)
    df['EMA200'] = df['Open'].ewm(span=200)
    df = n_day_mom(df, 'Open', 10)
    df = n_day_mom(df, 'Open', 30)

    df = n_day_rsi(df, 'Open', 10)
    df = n_day_rsi(df, 'Open', 30)
    df = n_day_rsi(df, 'Open', 200)

    df = macd(df, 'Open')
    # a = df.ta.macd(close='Open', fast=12, slow=26, signal=9, append=True)
    print(1)
