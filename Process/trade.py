import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import QuantileTransformer
from joblib import load

from sqlalchemy import create_engine
import pymssql
from datetime import datetime

from alpaca.trading.client import TradingClient
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.trading.requests import MarketOrderRequest, GetAssetsRequest
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass

import warnings
warnings.filterwarnings('ignore')


class Trade:
    def __init__(self) -> None:
        self.KEY = "PKADLK3NVSMDQ7CZY6BE"
        self.SECRET_KEY = "Mb8PUpwLhVV4DgVCDeaI0qGvHleZkgMg1KY179nH"
        self.trading_client = TradingClient(self.KEY, self.SECRET_KEY, paper=True)

        self.qt = QuantileTransformer(output_distribution="normal")
        self.pipe = load("Finance_Project_RidgeReg.joblib")
        self.symbols = None
        self.features = ["acc", "agr", "beta", "bm", "ep", "gma", "lev", 
                    "mom12m", "mom1m", "operprof", "roic", "roaq", 
                    "retvol", "saleinv", "currat"]

    def set_features(self, features):
        self.features = features

    def set_pipe(self, pipe):
        self.pipe = pipe

    def set_keys(self, key, secret_key):
        self.KEY = key
        self.SECRET_KEY = secret_key
        self.trading_client = TradingClient(self.KEY, self.SECRET_KEY, paper=True)

    def get_today_table(self):
        server = 'fs.rice.edu'
        database = 'stocks'
        username = 'stocks'
        password = '6LAZH1'
        string = "mssql+pymssql://" + username + ":" + password + "@" + server + "/" + database 
        conn = create_engine(string).connect()

        df = pd.read_sql(
        """
        select ticker, date, mve, acc, agr, beta, bm, ep, gma, idiovol, lev, mom12m, mom1m, 
            operprof, roeq, roic, roaq, retvol, saleinv, currat
        from today
        where price > 5
        """, 
        conn
        )
        conn.close()


        df = df.dropna()
        df = df.set_index("ticker")
        df = df.sort_values(by="mve")
        df = df.iloc[:-500]
        return df

    def get_account_equity(self):
        account = self.trading_client.get_account()
        cash = float(account.cash)
        equity = float(account.equity)
        print(f"Equity {equity}, Cash {cash}")
        return equity, cash

    def predict_and_rank(self, df, numstocks=100):
        av_assets = self.trading_client.get_all_assets()

        av_assets = [
            x for x in av_assets 
            if (x.asset_class[:]=='us_equity') 
            and (x.status[:]=='active')
        ]
        pipe = self.pipe
        self.symbols = [x.symbol for x in av_assets]
        symbols = self.symbols
        tradable = [x.tradable for x in av_assets]
        shortable = [x.shortable for x in av_assets]
        numstocks = numstocks #best stocks
        numstocks2 = round(numstocks*30/130) #worst stocks

        threshold_top = round(numstocks*1.5)
        threshold_bot = round(numstocks2*1.5)

        trans_features = self.qt.fit_transform(df[self.features])
        trans_features = pd.DataFrame(trans_features, columns=self.features)
        df["predict"] = pipe.predict(trans_features)
        df["tradable"] = pd.Series(tradable, index=symbols)
        df["shortable"] = pd.Series(shortable, index=symbols)
        df = df.fillna(value=False, inplace=False)

        data_client = StockHistoricalDataClient(self.KEY, self.SECRET_KEY)
        params = StockLatestQuoteRequest(symbol_or_symbols=df.index.to_list())
        quotes = data_client.get_stock_latest_quote(params)

        df["ask"] = [quotes[x].ask_price for x in df.index]
        df["bid"] = [quotes[x].bid_price for x in df.index]

        #Get worst stocks
        df["rnk"] = df.predict.rank(method="first")
        worst = df[df.rnk<=threshold_bot]
        worst = worst.sort_values(by='rnk')

        #Get best stocks
        df["rnk"] = df.predict.rank(method="first", ascending=False)
        best = df[df.rnk<=threshold_top]
        best = best.sort_values(by='rnk')

        return df, best, worst

    def position_retriever(self, tradable, shortable):
        """Get the postions out of the account and the nummber of long tickers and short tickers

        Args:
            tradable (list): tradable stocks
            shortable (list): shortable stocks
            symbols (list): all the stocks

        Returns:
            tuple: positions(DF), long_ntickers(int), short_ntickers(int)
        """
        symbols = self.symbols
        all_positions = self.trading_client.get_all_positions()
        
        tick_val = []
        qty_val = []

        for x in all_positions:
            tick_val.append(x.symbol)
            qty_val.append(float(x.qty))

        positions = pd.DataFrame({'Ticker':list(tick_val), 'Qty':list(qty_val)})
        positions = positions.set_index('Ticker')
        positions['Long'] = positions['Qty'] > 0
        
        positions["tradable"] = pd.Series(tradable, index=symbols)
        positions["shortable"] = pd.Series(shortable, index=symbols)
        positions = positions.fillna(value=False, inplace=False)
        
        sell = [True 
            if (positions.loc[ticker, 'Long']) 
            and (positions.loc[ticker, 'shortable'])
            else False
            for ticker in positions.index]

        positions["sell"] = pd.Series(sell, index=positions.index)

        buy = [True 
                if (not positions.loc[ticker, 'Long'])
                and (positions.loc[ticker, 'tradable'])
                else False
                for ticker in positions.index]
        positions["buy"] = pd.Series(buy, index=positions.index)

        long_nticker = sum(positions['sell'])
        short_nticker = sum(positions['buy'])


        positions[positions['Qty']>0]

        
        return positions, long_nticker, short_nticker

    def get_out_of_positions(self, df, positions):
        sell_failed, buy_failed = 0, 0
        for tick in positions.index: 
            if positions.loc[tick, "sell"]:
                try:
                    market_order_data = MarketOrderRequest(
                        symbol=tick,
                        #qty=-positions.loc[tick, "trade"],
                        qty=abs(positions.loc[tick, "Qty"]),
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY
                    )
                    market_order = self.trading_client.submit_order(
                        order_data=market_order_data
                    )
                except Exception as e:
                    print(e)
                    sell_failed += 1
                    print(f"sell order for {tick} failed")
            
            if positions.loc[tick, "buy"]:
                try:
                    market_order_data = MarketOrderRequest(
                        symbol=tick,
                        qty=abs(positions.loc[tick, "Qty"]),
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY
                    )
                    market_order = self.trading_client.submit_order(
                        order_data=market_order_data
                    )
                except Exception as e:
                    print(e)
                    buy_failed += 1
                    print(f"buy order for {tick} failed")
                    #Buying back the shorted stocks might often fail, if we don't have many to invest
        return sell_failed, buy_failed


    def get_portfolio(self, best, worst, cash, long_nticker=0, short_nticker=0):
        if long_nticker != 0:
            long_per_stock = 1.3*cash / long_nticker
        else:
            long_per_stock = 0
            
        if short_nticker != 0:
            short_per_stock = 0.3*cash / short_nticker
        else:
            short_per_stock = 0  

        worst = worst.sort_values(by="rnk")
        best = best.sort_values(by="rnk")

        try:
            if short_nticker > len(worst[worst.shortable & (worst.bid>0)].rnk.tolist()):
                short_nticker =  len(worst[worst.shortable & (worst.bid>0)].rnk.tolist()) - 1
            short_cutoff = worst[worst.shortable & (worst.bid>0)].rnk.iloc[short_nticker-1]
            worst["target"] = np.where(
                worst.shortable & (worst.bid>0) & (worst.rnk<=short_cutoff),
                -short_per_stock/worst.bid, 
                0
            )
            worst["target"] = worst.target.astype(int)
        except:
            worst["target"] = 0
            
        try:
            if long_nticker > len(best[worst.shortable & (best.bid>0)].rnk.tolist()):
                long_nticker =  len(best[worst.shortable & (best.bid>0)].rnk.tolist()) - 1
            long_cutoff = best[best.tradable & (best.ask>0)].rnk.iloc[long_nticker-1]
            best["target"] = np.where(
                best.tradable & (best.ask>0) & (best.rnk<=long_cutoff), 
                long_per_stock/best.ask, 
                0
            )
            best["target"] = best.target.astype(int)
        except:
            best["target"] = 0
        
        return best, worst

    def make_trade(self, best, worst):
        for tick in worst.index: 
            if worst.loc[tick, "target"]<0:
                try:
                    market_order_data = MarketOrderRequest(
                        symbol=tick,
                        qty=-worst.loc[tick, "target"],
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY
                    )
                    market_order = self.trading_client.submit_order(
                        order_data=market_order_data
                    )
                    print(market_order)
                except Exception as e:
                    print(e)
                    print(f"sell order for {tick} failed")

        for tick in best.index: 
            if best.loc[tick, "target"]>0:
                try:
                    market_order_data = MarketOrderRequest(
                        symbol=tick,
                        qty=best.loc[tick, "target"],
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY
                    )
                    market_order = self.trading_client.submit_order(
                        order_data=market_order_data
                    )
                    print(market_order)
                except Exception as e:
                    print(e)
                    print(f"buy order for {tick} failed")

    def save_results(self, best, worst):
        trading_client = self.trading_client = TradingClient(self.KEY, self.SECRET_KEY, paper=True)
        today = datetime.today().strftime("%Y-%m-%d")
        worst['type'] = 'Worst'
        best['type'] = 'Best'

        worst["date"] = today
        best["date"] = today

        try:
            df2 = pd.read_csv(f"./results/trade_data.csv", index_col="ticker")
            df2 = df2[df2.date != today]
            df2 = pd.concat((df2, pd.concat((worst, best))))
            df2.to_csv(f"./results/trade_data.csv")
        except:
            df2 = pd.concat((worst, best))
            df2.to_csv("./results/trade_data.csv")

        account = trading_client.get_account()
        account = dict(account)
        account = pd.DataFrame(pd.Series(account)).T
        account["date"] = today

        try:
            d = pd.read_csv("./results/account.csv")
            d = d[d.date != today]
            account = pd.concat((d, account))
            account.to_csv("./results/account.csv")
        except:
            account.to_csv("./results/account.csv")

        positions = trading_client.get_all_positions()
        positions = {x.symbol: x.qty for x in positions}
        positions = pd.DataFrame(pd.Series(positions))
        positions["date"] = today

        try:
            d = pd.read_csv("./results/positions.csv")
            d = d[d.date != today]
            positions = pd.concat((d, positions))
            positions.to_csv("./results/positions.csv")
        except:
            positions.to_csv("./results/positions.csv")