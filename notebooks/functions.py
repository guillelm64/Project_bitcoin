import logging, sys
logging.disable(sys.maxsize)
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.graphics.tsaplots as sgt
from pylab import rcParams
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric
from prophet.diagnostics import performance_metrics
import datetime
from dateutil.relativedelta import relativedelta
import pickle as pck
from pmdarima.arima import auto_arima
from sktime.forecasting.model_selection import temporal_train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import math 



def adf_test(timeseries):
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    print(dfoutput)
    

def kpss_test(timeseries):
    print("Results of KPSS Test:")
    kpsstest = kpss(timeseries, regression="c", nlags = "auto") 
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    print(kpss_output)
    
    
def create_sequences(data, seq_size = 1):
    d = []

    for index in range(len(data) - seq_size):
        d.append(data[index: index + seq_size])

    return np.array(d)
    
def preprocess(data_raw, seq_size, train_split):

    data = create_sequences(data_raw, seq_size)

    num_train = int(train_split * data.shape[0])

    X_train = data[:num_train, :-1, :]
    y_train = data[:num_train, -1, :]

    X_test = data[num_train:, :-1, :]
    y_test = data[num_train:, -1, :]

    return X_train, y_train, X_test, y_test

def show_plot(df, col1, col2):
    fig = px.line(df, x = col1, y = col2)

    return fig.show()
    

def show_halving_plots(df):

    bitcoin_12 = df[df["date"] < "2013-01-01"]
    bitcoin_13 = df[(df["date"] >= "2013-01-01") & (df["date"] < "2014-01-01")]
    bitcoin_14 = df[(df["date"] >= "2014-01-01") & (df["date"] < "2015-01-01")]
    bitcoin_15 = df[(df["date"] >= "2015-01-01") & (df["date"] < "2016-01-01")]
    bitcoin_16 = df[(df["date"] >= "2016-01-01") & (df["date"] < "2017-01-01")]
    bitcoin_17 = df[(df["date"] >= "2017-01-01") & (df["date"] > "2018-01-01")]
    bitcoin_18 = df[(df["date"] >= "2018-01-01") & (df["date"] < "2019-01-01")]
    bitcoin_19 = df[(df["date"] >= "2019-01-01") & (df["date"] < "2020-01-01")]
    bitcoin_20 = df[(df["date"] >= "2020-01-01") & (df["date"] < "2021-01-01")]
    bitcoin_21 = df[(df["date"] >= "2021-01-01") & (df["date"] < "2022-01-01")]
    bitcoin_22 = df[(df["date"] >= "2022-01-01")]
    fig = px.line(bitcoin_12, y = "close", x = "date")
    fig.add_vline(x = "2012-11-28 00:00:00", line_width = 5, line_dash = "dash", line_color = "black")
    fig.show()
    fig = px.line(bitcoin_16, y = "close", x = "date")
    fig.add_vline(x = "2016-07-09 00:00:00", line_width = 5, line_dash = "dash", line_color = "black")
    fig.show()
    fig = px.line(bitcoin_20, y = "close", x = "date")
    fig.add_vline(x = "2020-05-11 00:00:00", line_width = 5, line_dash = "dash", line_color = "black")
    
    return fig.show()


def decomposing(df):
    rcParams['figure.figsize'] = 18, 8
    decomposition = sm.tsa.seasonal_decompose(df["close"], model='additive')
    fig = decomposition.plot()
    return plt.show()

def acf_pacf(df):

    fig, axes = plt.subplots(1, 2, figsize= (30, 10))

    fig1 = sm.graphics.tsa.plot_acf(df,
                               lags = 20,                               
                               ax=axes[0])
    fig2 = sm.graphics.tsa.plot_pacf(df, lags = 20, 
                                ax=axes[1], method = "ywm")
    return fig1, fig2


def plot_predictions(df):
    bitcoin_21_22 = df[df["date"] >= "2021-01-01"]
    AR_model_2 = pck.load(open("../models/AR_model_2.pkl", 'rb'))
    preds = AR_model_2.predict(start = 494, end = 616)
    
    return preds.head()
def residuals_AR_plot(df):
    df_2 = df.copy()
    bitcoin_21_22 = df_2[df_2["date"] >= "2021-01-01"]

    bitcoin_21_22["AR_predictions_1"] = AR_model_2.predict(start = 494, end = 616)
  
    fig, ax = plt.subplots(1,figsize=(16,8))
    sns.scatterplot(dara = bitcoin_21_22, y = "AR_predictions_1", x = "close");
    sns.lineplot(data = bitcoin_21_22, x = 'close', y = 'close',color = 'black');
    return plt.show()

def future_pred(df):
    bitcoin_21_22 = df[df["date"] >= "2021-01-01"]
    model = pck.load(open("../models/AR_model_2.pkl", 'rb'))
    future_pred = model.predict(start = "2022-09-10", end = "2022-09-29", dynamic=False) 
    plt.plot(df["close"])
    plt.plot(future_pred, color='orange')
    return plt.show()