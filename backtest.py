# backtest for python
import matplotlib.pyplot as plt
from abc import abstractmethod

import numpy as np
import pandas as pd
from trading_dates import *
from yf import *

'''Considering the frequent data retrieval from servers during PNL computation, 
the first consideration is to maintain a local daily data packet'''
'''1. Network instability, serial computing logic, and restarting once timeout occurs;
2. Redundant data requests, wasting server traffic;'''


class localDataMaintainer(object):
    '''
    class to matin local daily data
    '''

    def __init__(self):
        self.data = pd.read_pickle('localData')
        self.stocks = self.data['dailyLineDict'].keys()
        self.lastDay = self.data['version'][-1]
        self.allTradingDay = get_trading_dates(self.lastDay)

    def __updateDailyLine(self):
        '''
        Update local daily data
        :return: None
        '''
        print('daily line data updating...')
        begin = (pd.to_datetime(self.lastDay) + timedelta(days=1)).strftime('%Y-%m-%d')
        for s in self.stocks:
            print('Still downloading, do not rush haha...')
            priceAll = get_price_range(s, begin)
            self.data['dailyLineDict'][s] = pd.concat([self.data['dailyLineDict'][s], priceAll])

    def update(self):
        '''
        public
        :return: None
        '''
        print("The backtesting interval supported by the current data:" + self.data['version'][0] + '~' +
              self.data['version'][1])
        # self.allTradingDay = ['2022-07-11', ]

        if len(self.allTradingDay) < 3:
            print('The current data is already the latest! Exited the update program!')
            # return
        # self.allTradingDay=self.allTradingDay[1:-1]
        if input('The version of the local data is ' + self.lastDay + '，expected update time: ' + str(
                len(self.allTradingDay) * 2) + 'min，Still want to update? N+Enter to exit，else+Enter to continue') == 'N':
            print('We do not update local date this time.')
            return
        self.__updateDailyLine()
        self.data['version'] = ['2009-01-01', self.allTradingDay[-1], self.allTradingDay[-1]]
        pd.to_pickle(self.data, 'localData')
        print('done!')


class backtest(object):
    '''
    Classes for backtesting strategy performance
    '''

    def __init__(self, slippage, filePath, initCap=1e8, isSplit=False, isSegment=False):
        '''
        :param slippage: slippage
        :param filePath: filePath
        :param initCap: The initial capital, deflaut is 1e8
        :param isSplit:
        :param isSegment:
        '''
        self.slippage = slippage
        self.isSplit = isSplit
        self.initCap = initCap
        self.isSegment = isSegment
        self.filePath = filePath
        # First prepare the data
        self.__initializeData()

    # We write these functions as private
    def __initializeData(self):
        '''
        read the local data into memory
        :return:
        '''
        localData = pd.read_pickle('localData')
        # 预读日线数据到缓存
        self.dailyLineDict = localData['dailyLineDict']
        self.minTickDict = localData['minTickDict']
        self.version = localData['version']

    def __getLocalSupportDataVersion(self):
        '''
        The version of local data
        :return: the start date and end date of the data, recent update date
        '''
        return self.version

    def __getTradingDateList(self):
        '''
        Calculate the backtesting interval supported by the current data
        :return:
        '''
        # The begin and the end of the local data
        begin, end, self.dataVersion = self.__getLocalSupportDataVersion()
        begin, end = pd.to_datetime([begin, end])
        weight = self.weightDf.copy(deep=True)
        weight['date'] = pd.to_datetime(weight.index)
        date = weight.date
        # the begin and end of the weight data
        weightBegin = date.min()
        weightEnd = date.max()

        # the begin and the end that can be calculated
        self.begin = str(max(begin, weightBegin)).split(' ')[0]
        self.end = str(min(end, weightEnd)).split(' ')[0]
        self.tradingDateList = get_trading_dates(self.begin, self.end)
        weight = weight[(weight.date >= pd.to_datetime(self.begin)) & (weight.date <= pd.to_datetime(self.end))]
        self.weight = weight

    def __dynamicLeverage(self, pnl):
        '''
        :param pnl: input pnl，index is date，with only one column
        :return: the leverage on corresponding date
        '''
        # the volatility of destination
        destStdDev = 0.1
        # the judge to be 0
        limitPct = 1e-4
        # short lookback period
        shortBack = 60
        # long lookback period
        longBack = 250
        # minCount:
        minCount = 12
        initCap = self.initCap
        balance = pnl.copy(deep=True)
        balance.index = pd.to_datetime(balance.index)
        start_time = str(balance.index[0])[:10]
        end_time = str(balance.index[-1])[:10]
        trade_date = get_trading_dates(start_time, end_time)
        balance = balance.loc[trade_date]  # get the trading day part
        balance = balance / initCap
        pct_change = balance - balance.shift(1).fillna(1)
        pct_change = pct_change.T.apply(lambda x: float(x) if abs(float(x)) > limitPct else np.nan)
        leverage = pd.Series()
        for idx, date in enumerate(pct_change.index):
            if idx < shortBack:
                leverage.loc[idx] = 1
            elif (idx >= shortBack) & (idx < longBack):
                if len(pct_change.iloc[:idx].dropna()) < minCount:
                    leverage.loc[idx] = 1
                else:
                    leverage.loc[idx] = destStdDev / pct_change.iloc[:idx].dropna().std() / np.sqrt(250)
            else:
                if len(pct_change.iloc[idx - shortBack:idx].dropna()) < minCount:
                    if len(pct_change.iloc[idx - longBack:idx].dropna()) < minCount:
                        leverage.loc[idx] = 0
                    else:
                        leverage.loc[idx] = destStdDev / pct_change.iloc[idx - longBack:idx].dropna().iloc[
                                                         -minCount:].std() / np.sqrt(250)
                else:
                    leverage.loc[idx] = destStdDev / pct_change.iloc[idx - shortBack:idx].dropna().std() / np.sqrt(250)
        leverage = leverage.apply(lambda x: x if x <= 5 else 5)  # The maximum leverage is 5
        leverage.index = balance.index
        return leverage

    def _int(self, position):
        '''
        Round up the position value
        :param position:
        :return:
        '''
        position = np.ceil(position)
        return position

    def _calcPosition(self):
        '''
        calculate daily position based on realized calcWeight
        :return: dataframe，daily position
        index is date and stock name；
        column names are："product":stock name,'marketPrice': Entry price (excluding sliding points) for calculating returns,
        'yesterdayClose': Close price of yesterday,'todayClose': close price of today,'todayPosition': Expected position,
        'minTick':
        '''
        weight = self.weightDf
        positionDf = pd.DataFrame()
        dynamicLev = self.dynamicLev
        for index, today in enumerate(self.tradingDateList):
            print(today)
            todayLev = float(dynamicLev.loc[today])
            # Get the date of last trading day
            if index:
                yesterday = self.tradingDateList[index - 1]
            else:
                yesterday = get_previous_trading_date(today)
            # Weight today，series
            todayWeightSeries = weight.loc[today]
            stocks = todayWeightSeries.index
            for stock in stocks:
                # get the basic information of the target stock
                minTick = self.minTickDict

                # if the weight today is 0
                if todayWeightSeries[stock] == 0:
                    if index:
                        try:
                            # whether there is position yesterday
                            stockYesterday = positionDf.loc[[(yesterday, stock)]]
                        except:
                            # if we cannot loc, there is no postion yesterday
                            continue
                        # if the target stock has position yesterday
                        if len(stockYesterday)>0:
                            # Yesterday is not 0, today is 0
                            for s in stockYesterday.iterrows():
                                if s[1]['todayPosition'] == 0:
                                    continue
                                # Today's Open price
                                marketPrice = float(self.dailyLineDict[stock].loc[today, 'Open'])
                                # Today's Close price
                                todayClose = float(self.dailyLineDict[stock].loc[today, 'Close'])
                                # Yesterday's Close price
                                yesterdayClose = float(self.dailyLineDict[stock].loc[yesterday, 'Close'])
                                # write the data
                                positionDf = pd.concat([positionDf,
                                                        pd.DataFrame(
                                                            index=pd.MultiIndex.from_product([[today, ], [stock, ]]),
                                                            data={'marketPrice': marketPrice,
                                                                  'yesterdayClose': yesterdayClose,
                                                                  'todayClose': todayClose,
                                                                  'todayPosition': 0, 'minTick': minTick})])
                    # if this product does not have expected position today
                    continue
                # The following is the logic is that the product have expected position today
                # Today's Open price
                marketPrice = float(self.dailyLineDict[stock].loc[today, 'Open'])
                # Today's Close price
                todayClose = float(self.dailyLineDict[stock].loc[today, 'Close'])
                # Yesterday's Close price
                yesterdayClose = float(self.dailyLineDict[stock].loc[yesterday, 'Close'])
                todayVolume = float(self.dailyLineDict[stock].loc[today, 'Volume'])
                if todayVolume:
                    # If today is position is not 0
                    # Use yesterday's close price to calculate expected position today
                    todayPosition = self.initCap * todayLev * todayWeightSeries[stock] / yesterdayClose
                    todayPosition = self._int(todayPosition)
                    # write data
                    positionDf = pd.concat([positionDf,
                                            pd.DataFrame(index=pd.MultiIndex.from_product([[today, ], [stock, ]]),
                                                         data={'marketPrice': marketPrice,
                                                               'yesterdayClose': yesterdayClose,
                                                               'todayClose': todayClose,
                                                               'todayPosition': todayPosition, 'minTick': minTick})])
                else:

                    # There is no volumn on that day
                    try:
                        # Get yesterday's position
                        contractYesterday = positionDf.loc[[(yesterday, stock)]]
                        todayPosition = float(contractYesterday.todayPosition)
                        print(stock + today + 'volume 0& Yesterday has position and fail to adjust')
                    except:
                        todayPosition = 0
                        print(stock + today + 'volume 0& Yesterday has no position and cannot open position today')
                    # write data
                    positionDf = pd.concat([positionDf,
                                            pd.DataFrame(index=pd.MultiIndex.from_product([[today, ], [stock, ]]),
                                                         data={'marketPrice': marketPrice,
                                                               'yesterdayClose': yesterdayClose,
                                                               'todayClose': todayClose,
                                                               'todayPosition': todayPosition, 'minTick': minTick})])

        return positionDf

    def __calcPnl(self):
        '''
        calculate pnl
        :param position:  pre-processed position dataframe
        :param isSplit: boolean，
        :return: the net value
        '''
        position = self.position
        # calculate the position change every day
        position['delta'] = position['todayPosition'] - position.groupby('stock')['todayPosition'].shift(1).fillna(0)
        position['delta'] = position['delta'].astype('int')
        position = position.set_index('date')
        # Use pnl to calculate profit，Three parts，slippage loss + overnight profit + day profit
        pnl = position.groupby('stock'). \
            apply(lambda x: - self.slippage * x['minTick'] * abs(x['delta']) +
                            x['todayPosition'].shift(1).fillna(0) * (x['marketPrice'] - x['yesterdayClose']).fillna(0) +
                           x['todayPosition'] * (x['todayClose'] - x['marketPrice'])).reset_index()
        if len(pnl) != len(position):
            pnl = pnl.set_index('stock')
            pnl = pnl.stack().reset_index()
            pnl.columns = ['stock', 'date', 0]
        pnl.to_csv('detail.csv')
        pnl = pnl.groupby('date')[0].sum().reset_index()
        pnl = pnl.set_index('date')
        # let the profit of first day to be 0
        pnl.iloc[0] = 0
        pnl.columns = ['internPnl', ]
        tmp = pd.DataFrame(index=self.tradingDateList, columns=['internPnl', ], data=0)
        tmp.loc[pnl.index, 'internPnl'] = pnl
        pnl = tmp
        pnl['balance'] = pnl.cumsum() + self.initCap
        return pnl

    def __getTradingAnalyse(self, netValue):
        '''
        calculate performance
        :param pnl: PNL
        :return: backtest performance
        '''
        # natural day
        netValue.index = pd.to_datetime(netValue.index)
        tmp = pd.Series(data=np.NaN, index=pd.date_range(netValue.index[0], netValue.index[-1], freq='D'))
        tmp.loc[netValue.index] = netValue
        netValue = tmp
        # fill in the non-trading day data
        netValue = netValue.fillna(method='ffill')
        # 2
        dailyReturn = netValue - netValue.shift(1).fillna(1)
        days = len(dailyReturn)
        years = (dailyReturn.index[-1] - dailyReturn.index[0]).days / 365
        dailyReturn = dailyReturn.iloc[1:]
        # 3
        mean = np.mean(dailyReturn)
        # 4
        std = np.std(dailyReturn) * ((days - 1) / (days - 2)) ** 0.5
        # annulized std
        # 8
        annualStd = np.round(365 ** 0.5 * std, 6)
        # sharpe = 250 ** 0.5 * mean / std
        # 9
        sharpe = np.round(365 ** 0.5 * mean / std, 6)
        win = sum(dailyReturn > 0) / sum(dailyReturn != 0)
        winLoss = dailyReturn[dailyReturn > 0].mean() / np.abs(dailyReturn[dailyReturn < 0].mean())
        drawdown = netValue.cummax() - netValue
        # 7
        maxDd = np.round(max(drawdown), 6)
        # 5
        absReturn = np.round(netValue.iloc[-1] - 1, 6)
        # 6
        annualReturn = np.round(absReturn / years, 6)
        # 10
        calmar = np.round(absReturn / years / maxDd, 6)
        performance = pd.DataFrame([self.begin, self.end, self.slippage, self.initCap] +
                                   [annualStd, sharpe, calmar, absReturn, annualReturn, maxDd] +
                                   [self.dataVersion, win, winLoss],
                                   index=['begin', 'end', 'slippage',
                                          'initCap', 'stddev(annual)', 'sharpe',
                                          'calmar', 'absReturn', 'annualReturn',
                                          'maxDrawdown', 'dataVersion', 'win', 'winLoss'])
        # performance.T.to_csv(self.filePath + '\\Backtest.csv', index=False)
        return performance

    def __reportor(self, netValue, position, performance, version):
        netValue.to_csv(self.filePath + '\\netValue&Slippage' + str(self.slippage) + version + '.csv')
        performance.T.to_csv(self.filePath + '\\performance&Slippage' + str(self.slippage) + version + '.csv',
                             index=False)
        position[['date', 'stock', 'todayPosition']].to_csv(
            self.filePath + '\\position&Slippage' + str(self.slippage) + version + '.csv', index=False)
        position['count'] = position['todayPosition'] - position.groupby('stock')['todayPosition'].shift(1).fillna(0)
        position['count'] = position['count'].astype('int')
        execution = position[['date', 'stock', 'count', 'marketPrice']]
        execution['filledPrice'] = execution['marketPrice'] + np.sign(execution['count']) * position[
            'minTick'] * self.slippage
        execution = execution[abs(execution['count']) > 0]
        execution.to_csv(self.filePath + '\\order&Slippage' + str(self.slippage) + version + '.csv', index=False)

    # 以下是开放的接口
    @abstractmethod
    def calcWeight(self):
        '''
        rewrite when using
        :return: dataframe
        '''
        pass

    def fastBacktest(self, weightDf, versionA=False):
        '''
        fast backtest
        :return:
        '''

        self.weightDf = weightDf
        if versionA:
            slippage = self.slippage
            self.slippage = 1

            #
            self.__getTradingDateList()
            self.dynamicLev = pd.Series(index=self.tradingDateList, data=1)
            print('start calculating version A position...')
            self.position = self._calcPosition()
            print('finished')
            self.position = self.position.reset_index()
            self.position.columns = ['date', 'stock', 'marketPrice', 'yesterdayClose', 'todayClose',
                                     'todayPosition', 'minTick']
            netValue = self.__calcPnl()
            netValue['pnl'] = netValue.balance / self.initCap
            performance = self.__getTradingAnalyse(netValue.pnl)
            self.__reportor(netValue, self.position, performance, '')

            self.slippage = slippage
            balance = netValue['balance']
            if len(balance) < 250:
                print('The backtest period is too short!')
                return
            else:
                if self.weight.iloc[250].date >= pd.to_datetime('2012-01-01'):
                    self.weightDf = self.weightDf.iloc[250:]
                else:
                    self.weightDf = self.weightDf[self.weightDf.index >= pd.to_datetime('2012-01-01')]
                self.__getTradingDateList()
                self.dynamicLev = self.__dynamicLeverage(balance)
                self.dynamicLev[-len(self.weightDf):].to_csv(self.filePath + '\\dynamicLeverage.csv')
                self.tradingDateList = self.tradingDateList[-len(self.weightDf):]
            self.position = self._calcPosition()
            self.position = self.position.reset_index()
            self.position.columns = ['date', 'future', "product", 'marketPrice', 'yesterdayClose', 'todayClose',
                                     'todayPosition', 'multiplier', 'minTick']
            netValue = self.__calcPnl()
            netValue['pnl'] = netValue.balance / self.initCap
            performance = self.__getTradingAnalyse(netValue.pnl)
            self.__reportor(netValue, self.position, performance, '@A')
            # return the performance
            position = self.position[['date', 'future', "product", 'todayPosition']]
            position = position.set_index(['date', ])
            netValue.index = pd.to_datetime(netValue.index)
            return performance, netValue, position
        else:
            self.__getTradingDateList()
            self.dynamicLev = pd.Series(index=self.tradingDateList, data=1)
            self.position = self._calcPosition()
            self.position = self.position.reset_index()
            self.position.columns = ['date', 'stock', 'marketPrice', 'yesterdayClose', 'todayClose',
                                     'todayPosition', 'minTick']
            netValue = self.__calcPnl()
            netValue['pnl'] = netValue.balance / self.initCap
            plt.plot(netValue['pnl'])
            plt.show()
            performance = self.__getTradingAnalyse(netValue.pnl)
            self.__reportor(netValue, self.position, performance, '')
            # 返回性能表现
            position = self.position[['date', 'stock', 'todayPosition']]
            position = position.set_index(['date', ])
            netValue.index = pd.to_datetime(netValue.index)
            return performance, netValue, position

    def completedBacktest(self):
        '''
        output complete files
        :return:
        '''
        pass


if __name__ == '__main__':
    mt = localDataMaintainer()
    mt.update()
    # test = backtest(0, './')
    # data = pd.read_csv('knn.csv', index_col=0)
    # data = pd.merge(data, pd.read_csv('knn1.csv', index_col=0), left_index=True, right_index=True)
    # data = pd.merge(data, pd.read_csv('knn2.csv', index_col=0), left_index=True, right_index=True)
    # data = pd.merge(data, pd.read_csv('knn3.csv', index_col=0), left_index=True, right_index=True)
    # data = data/4
    # data[data<0] = 0
    # data = pd.read_pickle('sample.pkl')
    # data = data.unstack()
    # data.columns = ['ALLE', 'CLX', 'CMG', 'DAL']
    # test.fastBacktest(data)
