from os import getcwd
from sklearn.decomposition import PCA
from backtest import *
from cal_features import *
from sklearn.neighbors import NearestNeighbors
import pickle


class knn_strategy(backtest):
    def __init__(self, begin, end, slippage):
        self.begin = begin
        self.end = end
        self.slippage = slippage
        # self.stockList = get_trading_stocks()
        # self.stockList = ['ALLE', 'CMG', 'DAL', 'ZTS', 'TPR', 'NUE', 'LMT', 'ITW', 'FLT', 'EQT']
        # self.stockList = ['CMG', 'NKE', 'ALLE', 'HSY', 'GPC']
        self.stockList = ['ALLE', 'GPC', 'HSY', 'CMG', 'NKE', 'NUE', 'EQT']
        self.datelist = get_trading_dates(self.begin, self.end)
        self.All_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA10', 'MA30', 'MA200', 'EMA10', 'EMA30',
                             'EMA200', 'MOM10', 'MOM30', 'RSI10', 'RSI30', 'RSI200', 'MACD']
        # self.features = ['H-L', 'O-C', 'MOM10', 'Volume', 'MA10', 'EMA10', 'RSI10', 'MACD']
        self.features = self.All_features + ['H-L', 'O-C']
        self.localData = pd.read_pickle('localData')
        self.past_range = 252
        self.n_day_return = 1
        self.k = 20
        self.gap = 252
        backtest.__init__(self, slippage=slippage, filePath=getcwd())

    def get_features(self, ticker):
        feature = pd.DataFrame(index=self.datelist, columns=self.features)
        if 'Open' in self.features:
            feature.loc[self.datelist, 'Open'] = self.localData['dailyLineDict'][ticker].loc[self.datelist, 'Open']
        if 'Close' in self.features:
            feature.loc[self.datelist, 'Close'] = self.localData['dailyLineDict'][ticker].loc[self.datelist, 'Close']
        if 'High' in self.features:
            feature.loc[self.datelist, 'High'] = self.localData['dailyLineDict'][ticker].loc[self.datelist, 'High']
        if 'Low' in self.features:
            feature.loc[self.datelist, 'Low'] = self.localData['dailyLineDict'][ticker].loc[self.datelist, 'Low']
        if 'Volume' in self.features:
            feature.loc[self.datelist, 'Volume'] = self.localData['dailyLineDict'][ticker].loc[self.datelist, 'Volume']
        if 'H-L' in self.features:
            feature.loc[self.datelist, 'H-L'] = self.localData['dailyLineDict'][ticker].loc[self.datelist, 'High'] - self.localData['dailyLineDict'][ticker].loc[self.datelist, 'Low']
        if 'O-C' in self.features:
            feature.loc[self.datelist, 'O-C'] = self.localData['dailyLineDict'][ticker].loc[self.datelist, 'Open'] - self.localData['dailyLineDict'][ticker].loc[self.datelist, 'Close']
        if 'MA10' in self.features:
            feature.loc[self.datelist, 'MA10'] = moving_average(
                self.localData['dailyLineDict'][ticker].loc[self.datelist, 'Close'], 10)
        if 'MA30' in self.features:
            feature.loc[self.datelist, 'MA30'] = moving_average(
                self.localData['dailyLineDict'][ticker].loc[self.datelist, 'Close'], 30)
        if 'MA200' in self.features:
            feature.loc[self.datelist, 'MA200'] = moving_average(
                self.localData['dailyLineDict'][ticker].loc[self.datelist, 'Close'], 200)
        if 'EMA10' in self.features:
            feature.loc[self.datelist, 'EMA10'] = e_moving_average(
                self.localData['dailyLineDict'][ticker].loc[self.datelist, 'Close'], 10)
        if 'EMA30' in self.features:
            feature.loc[self.datelist, 'EMA30'] = e_moving_average(
                self.localData['dailyLineDict'][ticker].loc[self.datelist, 'Close'], 30)
        if 'EMA200' in self.features:
            feature.loc[self.datelist, 'EMA200'] = e_moving_average(
                self.localData['dailyLineDict'][ticker].loc[self.datelist, 'Close'], 200)
        if 'MOM10' in self.features:
            feature.loc[self.datelist, 'MOM10'] = n_day_mom(
                self.localData['dailyLineDict'][ticker].loc[self.datelist, 'Close'], 10)
        if 'MOM30' in self.features:
            feature.loc[self.datelist, 'MOM30'] = n_day_mom(
                self.localData['dailyLineDict'][ticker].loc[self.datelist, 'Close'], 30)
        if 'RSI10' in self.features:
            feature.loc[self.datelist, 'RSI10'] = n_day_rsi(
                self.localData['dailyLineDict'][ticker].loc[self.datelist, 'Close'], 10)
        if 'RSI30' in self.features:
            feature.loc[self.datelist, 'RSI30'] = n_day_rsi(
                self.localData['dailyLineDict'][ticker].loc[self.datelist, 'Close'], 30)
        if 'RSI200' in self.features:
            feature.loc[self.datelist, 'RSI200'] = n_day_rsi(
                self.localData['dailyLineDict'][ticker].loc[self.datelist, 'Close'], 200)
        if 'MACD' in self.features:
            feature.loc[self.datelist, 'MACD'] = macd(
                self.localData['dailyLineDict'][ticker].loc[self.datelist, 'Close'])
        feature = feature.dropna()
        for i in feature.columns:
            feature.loc[:, i] = (feature.loc[:, i] - feature.loc[:, i].min()) / (feature.loc[:, i].max() - feature.loc[:, i].min())
            # feature.loc[:, i] = (feature.loc[:, i] - feature.loc[:, i].mean()) / feature.loc[:, i].std()
        return feature

    def knn_predict(self, ticker):
        data = self.get_features(ticker)
        data['return'] = self.localData['dailyLineDict'][ticker].loc[data.index, 'Close']
        data['return'] = data['return'].shift(-self.n_day_return) / data['return']
        data = data.dropna()
        signal = pd.DataFrame()
        for index, date in enumerate(data.index):
            if index < self.past_range:
                continue
            past_sample = data.iloc[index - self.past_range:index-self.n_day_return+1, :]
            train = np.array(past_sample.iloc[:, :-1])
            model = NearestNeighbors(n_neighbors=self.k)
            model.fit(train)
            new = np.array([data.iloc[index, :-1]])
            distances, indices = model.kneighbors(new)
            # a = list(indices)
            ret = past_sample.iloc[indices[0], -1].mean() - 1
            # ret = (past_sample.iloc[indices[0], -1] - 1).tolist()
            # pastdate = past_sample.iloc[indices[0], -1].index
            std = past_sample.iloc[indices[0], -1].std()
            signal = pd.concat([signal, pd.DataFrame(
                index=[date],
                data={'ret': [ret], 'std': [std]}
            )])
        # signal = signal.shift()
        return signal.dropna()

    def knn_predict1(self, ticker, gap):
        data = self.get_features(ticker)
        data['return'] = self.localData['dailyLineDict'][ticker].loc[data.index, 'Close']
        data['return'] = data['return'].shift(-self.n_day_return) / data['return']
        data = data.dropna()
        signal = pd.DataFrame()
        for index, date in enumerate(data.index):
            if index < self.past_range + gap:
                continue
            past_sample = data.iloc[index - self.past_range-gap:index-self.n_day_return+1-gap, :]
            train = np.array(past_sample.iloc[:, :-1])
            model = NearestNeighbors(n_neighbors=self.k)
            model.fit(train)
            new = np.array([data.iloc[index, :-1]])
            distances, indices = model.kneighbors(new)
            # a = list(indices)
            ret = past_sample.iloc[indices[0], -1].mean() - 1
            # ret = (past_sample.iloc[indices[0], -1] - 1).tolist()
            # pastdate = past_sample.iloc[indices[0], -1].index
            std = past_sample.iloc[indices[0], -1].std()
            signal = pd.concat([signal, pd.DataFrame(
                index=[date],
                data={'ret': [ret], 'std': [std]}
                # data={'ret': [ret], 'std': [std], 'pastdate': [pastdate]}
            )])
        signal = signal.shift()
        return signal.dropna()

    def knn_predict2(self, ticker, gap):
        data = self.get_features(ticker)
        data['return'] = self.localData['dailyLineDict'][ticker].loc[data.index, 'Close']
        data['return'] = data['return'].shift(-self.n_day_return) / data['return']
        data = data.dropna()
        pca = PCA(3)
        principal_components = pca.fit_transform(np.array(data.iloc[:, :-1]))
        data = pd.DataFrame(index=data.index, data=np.column_stack((principal_components, data['return'])))
        signal = pd.DataFrame()
        for index, date in enumerate(data.index):
            if index < self.past_range + gap:
                continue
            past_sample = data.iloc[index - self.past_range-gap:index-self.n_day_return+1-gap, :]
            train = np.array(past_sample.iloc[:, :-1])
            model = NearestNeighbors(n_neighbors=self.k)
            model.fit(train)
            new = np.array([data.iloc[index, :-1]])
            distances, indices = model.kneighbors(new)
            # a = list(indices)
            ret = past_sample.iloc[indices[0], -1].mean() - 1
            # ret = (past_sample.iloc[indices[0], -1] - 1).tolist()
            # pastdate = past_sample.iloc[indices[0], -1].index
            std = past_sample.iloc[indices[0], -1].std()
            signal = pd.concat([signal, pd.DataFrame(
                index=[date],
                data={'ret': [ret], 'std': [std]}
                # data={'ret': [ret], 'std': [std], 'pastdate': [pastdate]}
            )])
        signal = signal.shift()
        return signal.dropna()

    def knn_predict3(self, ticker, gap):
        data = self.get_features(ticker)
        loaded_model = pickle.load(open('model.sav', 'rb'))
        data['return'] = self.localData['dailyLineDict'][ticker].loc[data.index, 'Close']
        data['return'] = data['return'].shift(-self.n_day_return) / data['return']
        data = data.dropna()
        signal = pd.DataFrame()
        for index, date in enumerate(data.index):
            if index < self.past_range + gap:
                continue
            past_sample = data.iloc[index - self.past_range-gap:index-self.n_day_return+1-gap, :]
            train = np.array(past_sample.iloc[:, :-1])
            model = NearestNeighbors(n_neighbors=self.k)
            model.fit(train)
            new = np.array([data.iloc[index, :-1]])
            distances, indices = model.kneighbors(new)
            # a = list(indices)
            ret = loaded_model.predict(np.array([past_sample.iloc[indices[0], -1].tolist()])) - 1
            ret1 = (past_sample.iloc[indices[0], -1] - 1).tolist()
            # pastdate = past_sample.iloc[indices[0], -1].index
            std = past_sample.iloc[indices[0], -1].std()
            signal = pd.concat([signal, pd.DataFrame(
                index=[date],
                data={'ret': [ret], 'std': [std]}
                # data={'ret': [ret], 'std': [std], 'pastdate': [pastdate]}
            )])
        signal = signal.shift()
        return signal.dropna()

    def simple_knn(self, signal):
        signal['sig'] = 0
        signal.loc[signal['ret'] > 1 * signal['std']/self.k, 'sig'] = 1
        signal.loc[signal['ret'] < -1 * signal['std']/self.k, 'sig'] = -1
        signal.loc[signal['ret'] > 2 * signal['std']/self.k, 'sig'] = 2
        signal.loc[signal['ret'] < -2 * signal['std']/self.k, 'sig'] = -2
        signal.iloc[0, -1] = 0
        for n, today in enumerate(signal.index):
            if n:
                if int(signal.loc[today, 'sig']) == 1:
                    if int(signal.iloc[n-1, -1]) < 1:
                        signal.loc[today, 'sig'] = 0
                if int(signal.loc[today, 'sig']) == -1:
                    if int(signal.iloc[n-1, -1]) > -1:
                        signal.loc[today, 'sig'] = 0
                continue
        long = (signal['sig'] > 0)
        short = (signal['sig'] < 0)
        return [long, short]

    def calcWeight(self):
        weightDf = pd.DataFrame()
        for index, ticker in enumerate(self.stockList):
            # signal = self.knn_predict1(ticker, self.gap)
            signal = self.knn_predict2(ticker, self.gap)
            long, short = self.simple_knn(signal)
            if not index:
                weightDf = pd.DataFrame(index=signal.index, columns=self.stockList, data=0)
            weightDf.loc[short, ticker] += -1 / len(self.stockList)
            weightDf.loc[long, ticker] += 1 / len(self.stockList)
        return weightDf


if __name__ == '__main__':
    # a = pd.read_pickle('localData')['dailyLineDict']['ALLE']
    test = knn_strategy('2018-11-14', '2023-11-28', slippage=0)
    # test = knn_strategy('2016-11-14', '2021-11-28', slippage=0)
    weightDf = test.calcWeight()
    result = test.fastBacktest(weightDf, versionA=False)
    print(1)
