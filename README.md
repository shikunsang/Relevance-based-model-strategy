# KNN Strategy for Predicting Future Returns

## Overview

The basic idea behind our group's strategy is to identify historical days that share similar features with the current day. By doing so, we believe that the future returns on these similar past days will also be comparable to today's returns.

We achieve this using the K-Nearest Neighbors (KNN) algorithm. Specifically:

1. For a given time point **t**, we examine historical data to find days that closely resemble the current day.
2. These similar days serve as reference points for predicting future returns.

## Components

Our strategy consists of the following components:

1. **yf.py**: This script retrieves financial data from Yahoo Finance.
2. **backtest.py**: I've personally developed this backtesting engine.
3. **knn_strategy.py**: The heart of our strategy lies in this script.
4. **cal_features.py**: Get the features we need from the raw data.

The performance of the strategy is in the result folder.



