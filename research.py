# Test the normality of the returns
from knn_strategy import *
from sklearn.linear_model import LinearRegression
import pickle
from scipy.stats import shapiro, kstest
import statsmodels.api as sm
import matplotlib.pyplot as plt


class reg_knn(knn_strategy):
    def __init__(self, begin, end, slippage):
        knn_strategy.__init__(self, begin=begin, end=end, slippage=slippage)

    def reg_weight(self, ticker, gap):
        data = self.get_features(ticker)
        data['return'] = self.localData['dailyLineDict'][ticker].loc[data.index, 'Close']
        data['return'] = data['return'].shift(-self.n_day_return) / data['return']
        data = data.dropna()
        sample = pd.DataFrame()
        for index, date in enumerate(data.index):
            if index < self.past_range + gap:
                continue
            past_sample = data.iloc[index - self.past_range - gap:index - self.n_day_return + 1 - gap, :]
            train = np.array(past_sample.iloc[:, :-1])
            model = NearestNeighbors(n_neighbors=self.k)
            model.fit(train)
            new = np.array([data.iloc[index, :-1]])
            distances, indices = model.kneighbors(new)
            # a = list(indices)
            ret = (past_sample.iloc[indices[0], -1] - 1).tolist()
            # pastdate = past_sample.iloc[indices[0], -1].index
            std = past_sample.iloc[indices[0], -1].std()
            actual_return = float(data.loc[date, ['return']]) - 1
            sample = pd.concat([sample, pd.DataFrame(
                index=[date],
                # data={'ret': [ret], 'std': [std]}
                data={'ret': [ret], 'std': [std], 'actual_return': actual_return}
            )])
        return sample.dropna()


if __name__ == '__main__':
    reg = reg_knn('2017-01-01', '2020-12-31', 0)
    sample = reg.reg_weight('CMG', reg.gap)
    X = np.array(sample['ret'].tolist())
    y = sample['actual_return']
    regressor = LinearRegression(fit_intercept=False)
    regressor.fit(X, y)
    sample['pre'] = regressor.predict(X)
    sample['knn'] = sample['ret'].apply(lambda x: sum(x) / len(x))
    filename = 'model.sav'
    pickle.dump(regressor, open(filename, 'wb'))
