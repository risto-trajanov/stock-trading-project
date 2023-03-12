import pandas as pd
import os
import pickle
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pandas_datareader import DataReader as pdr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import joblib

siccd_codes = pd.read_csv('./siccodes12.csv')
model = LinearRegression()
qt = QuantileTransformer(output_distribution="normal")
poly = PolynomialFeatures(degree=2)

def get_industry(sicc):
    filtered_row = siccd_codes[((siccd_codes['start'] < sicc) & (sicc < siccd_codes['end']))]
    return filtered_row['industry'].values[0] if not filtered_row.empty else 'Other'

def qt_df(d):
    x = qt.fit_transform(d)
    return pd.DataFrame(x, columns=d.columns, index=d.index)

def quantile_transform(df, features):
    df[features] = df.groupby("date", group_keys=False)[features].apply(qt_df)
    # return df

def qt_ser(s):
    x = s.copy()
    x = x.to_numpy().reshape(-1, 1)
    x = qt.fit_transform(x).flatten()
    return pd.Series(x, index=s.index)

def quantile_transform_target(df, target):
    df["target"] = df.groupby("date", group_keys=False)[target].apply(qt_ser)
    # return df

def get_numerical_features(df):
    numerical_features = df.select_dtypes(include='number').columns.tolist()
    print(f'There are {len(numerical_features)} numerical features:', '\n')
    print(numerical_features)
    return numerical_features

def get_categorical_features(df):
    categorical_features = df.select_dtypes(exclude='number').columns.tolist()
    print(f'There are {len(categorical_features)} categorical features:', '\n')
    print(categorical_features)
    return categorical_features

def train(df, features, file_name, model, optimize=False, poly_flag=True):
    #train and predict

    predictions = None
    if 'target' not in df.columns.tolist():
        df['target'] = df['ret']
    df = df[features + ['target', 'ret']]
    xgb_reg = model
    if poly_flag:
        pipe = make_pipeline(poly, xgb_reg)
    else:
        pipe = make_pipeline(xgb_reg)
    dates = ["2005-01", "2010-01", "2015-01", "2020-01", "3000-01"]
    for train_date, end_date in zip(dates[:-1], dates[1:]):
        # print(1)
        filter1 = df.index.get_level_values("date") < train_date
        filter2 = df.index.get_level_values("date") < end_date

        train = df[filter1]
        test = df[~filter1 & filter2]

        Xtrain = train[features]
        ytrain = train["target"]
        Xtest = test[features]
        ytest = test["target"]

        # save
        file = f'./models/{file_name}_{train_date}.pkl'
        if os.path.exists(file):
            pipe = pickle.load(open(file, "rb"))
        else:
            pipe.fit(Xtrain, ytrain)
            pickle.dump(pipe, open(file, "wb"))
        print('pipe set score: ' + str(pipe.score(Xtrain, ytrain)))
        pred = pipe.predict(Xtest)
        print('Test set score: ' + str(pipe.score(Xtest,ytest)))
        pred = pd.Series(pred, index=test.index)
        predictions = pd.concat((predictions, pred))

    df["predict"] = predictions
    joblib.dump(pipe, f'./models/{file_name}.joblib')
    return df
    # rank(df)


def rank(df, numstocks = 200):
    df = df.dropna(subset=["predict"])

    numstocks = numstocks

    df["rnk"] = df.groupby("date").predict.rank(method="first", ascending=False)
    best = df[df.rnk<=numstocks]

    df["rnk"] = df.groupby("date").predict.rank(method="first")
    worst = df[df.rnk<=numstocks]

    best_rets = best.groupby("date").ret.mean()
    worst_rets = worst.groupby("date").ret.mean()
    rets = pd.concat((best_rets, worst_rets), axis=1)
    rets.columns = ["best", "worst"]
        
    evaluate(rets)

def evaluate(rets):
    rets_org = pd.read_csv("./rets_orig.csv", parse_dates=["date"], index_col="date")
    rets_org.index = rets_org.index.to_period("M")
    ff = pdr("F-F_Research_Data_5_Factors_2x3", "famafrench", start=2005)[0]/100

    rets_org["mkt"] = ff["Mkt-RF"] + ff["RF"]
    rets_org["rf"] = ff["RF"]
    results = 1.3*rets["best"] - 0.3*rets["worst"]
    results = results.values.tolist()
    rets_org["MY_ls"] = results
    rets_org['best'] = rets['best'].values.tolist()
    rets_org['worst'] = rets['worst'].values.tolist()

    (1+rets_org[["best", "worst", "mkt", "MY_ls"]]).cumprod().plot()

    ls = (1+rets_org.MY_ls).cumprod()
    lsmax = ls.expanding().max()
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax2.plot(rets_org.index.to_timestamp(), lsmax, 'b-')
    ax1.plot(rets_org.index.to_timestamp(), ls/lsmax - 1, 'r-')

    ax1.set_xlabel('Date')
    ax2.set_ylabel('Prior Maximum', color='b')
    ax1.set_ylabel('Drawdown', color='r')

    print(12*rets_org[["best", "worst", "mkt", "MY_ls"]].mean())
    print(np.sqrt(12)*rets_org[["best", "worst", "mkt", "MY_ls"]].std())

    xrets = rets_org[["best", "worst", "mkt", "MY_ls"]].subtract(rets_org.rf, axis=0)
    sharpes = np.sqrt(12)*xrets.mean()/xrets.std()
    print(f"Sharp ratio")
    print(sharpes)
    