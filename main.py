import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import scipy.stats as stats
import io

# Streamlitの設定
st.title('Stock Analysis with PCA and K-means Clustering')

# 銘柄の入力ボックスを追加
tickers = st.text_area('Enter stock tickers (separated by spaces):', '9535 AAPL MSFT')
ticker_list = [ticker.strip() for ticker in tickers.split()]
start_date = st.date_input('Start date:', pd.to_datetime('2020-01-01'))
end_date = st.date_input('End date:', pd.to_datetime('today'))

if st.button('Run Analysis'):
    results = []
    figures = []  # グラフを保存するリスト

    for ticker in ticker_list:
        # .Tを追加するロジック
        if len(ticker) == 4 and ticker.isdigit():
            ticker += '.T'

        # 株価データを取得
        data = yf.download(ticker, start=start_date, end=end_date)

        if data.empty:
            st.write(f"No data found for {ticker}.")
        else:
            st.write(f"Analysis for {ticker}")

            # 特徴量の作成
            data['SMA50'] = data['Close'].rolling(window=50).mean()
            data['SMA200'] = data['Close'].rolling(window=200).mean()
            data['Volatility'] = data['Close'].pct_change().rolling(window=50).std()
            data = data.dropna()

            # 特徴量の標準化
            features = ['Close', 'SMA50', 'SMA200', 'Volatility']
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data[features])

            # PCAの実行
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(data_scaled)
            pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
            pca_df['Date'] = data.index
            pca_df = pca_df.set_index('Date')

            # K-meansクラスタリング
            kmeans = KMeans(n_clusters=3, random_state=0)
            clusters = kmeans.fit_predict(principal_components)
            pca_df['Cluster'] = clusters

            # 投資判断の関数を定義
            def investment_decision(cluster):
                if cluster == 0:
                    return "Buy"
                elif cluster == 2:
                    return "Sell"
                else:
                    return "Hold"

            pca_df['Decision'] = pca_df['Cluster'].apply(investment_decision)

            # クラスタリングに基づく取引のシミュレーション
            pca_df['Close'] = data['Close']
            pca_df['Position'] = 0
            position = 0

            buy_signals = []
            sell_signals = []

            for i in range(1, len(pca_df)):
                if pca_df['Decision'].iloc[i] == 'Buy' and position == 0:
                    pca_df.loc[pca_df.index[i], 'Position'] = 1
                    position = 1
                    buy_signals.append(pca_df.index[i])
                elif pca_df['Decision'].iloc[i] == 'Sell' and position == 1:
                    pca_df.loc[pca_df.index[i], 'Position'] = -1
                    position = 0
                    sell_signals.append(pca_df.index[i])
                else:
                    pca_df.loc[pca_df.index[i], 'Position'] = pca_df['Position'].iloc[i-1]

            pca_df['Strategy_Returns'] = pca_df['Close'].pct_change() * pca_df['Position'].shift(1)

            # 累積リターンを計算
            pca_df['Cumulative_Strategy_Returns'] = (pca_df['Strategy_Returns'] + 1).cumprod()
            pca_df['Cumulative_Market_Returns'] = (pca_df['Close'].pct_change().fillna(0) + 1).cumprod()

            # パフォーマンス指標の計算
            strategy_returns = pca_df['Strategy_Returns'].dropna()
            market_returns = pca_df['Close'].pct_change().dropna()
            sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
            alpha, beta = stats.linregress(market_returns, strategy_returns)[0:2]

            # 最新の終値と投資判断を取得
            latest_close = data['Close'].iloc[-1]
            latest_decision = pca_df['Decision'].iloc[-1]

            # 結果をリストに追加
            results.append([ticker, latest_close, latest_decision, sharpe_ratio, alpha, beta])

    # 結果をデータフレームとして表示
    results_df = pd.DataFrame(results, columns=['Ticker', 'Latest Close', 'Investment Decision', 'Sharpe Ratio', 'Alpha', 'Beta'])
    st.write(results_df)

    # 各銘柄のグラフと詳細を表示
    for ticker in ticker_list:
        # .Tを追加するロジック
        if len(ticker) == 4 and ticker.isdigit():
            ticker += '.T'

        # 株価データを取得
        data = yf.download(ticker, start=start_date, end=end_date)

        if not data.empty:
            # 特徴量の作成
            data['SMA50'] = data['Close'].rolling(window=50).mean()
            data['SMA200'] = data['Close'].rolling(window=200).mean()
            data['Volatility'] = data['Close'].pct_change().rolling(window=50).std()
            data = data.dropna()

            # 特徴量の標準化
            features = ['Close', 'SMA50', 'SMA200', 'Volatility']
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data[features])

            # PCAの実行
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(data_scaled)
            pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
            pca_df['Date'] = data.index
            pca_df = pca_df.set_index('Date')

            # K-meansクラスタリング
            kmeans = KMeans(n_clusters=3, random_state=0)
            clusters = kmeans.fit_predict(principal_components)
            pca_df['Cluster'] = clusters


            # 投資判断の関数を定義
            def investment_decision(cluster):
                if cluster == 0:
                    return "Buy"
                elif cluster == 2:
                    return "Sell"
                else:
                    return "Hold"


            pca_df['Decision'] = pca_df['Cluster'].apply(investment_decision)

            # クラスタリングに基づく取引のシミュレーション
            pca_df['Close'] = data['Close']
            pca_df['Position'] = 0
            position = 0

            buy_signals = []
            sell_signals = []

            for i in range(1, len(pca_df)):
                if pca_df['Decision'].iloc[i] == 'Buy' and position == 0:
                    pca_df.loc[pca_df.index[i], 'Position'] = 1
                    position = 1
                    buy_signals.append(pca_df.index[i])
                elif pca_df['Decision'].iloc[i] == 'Sell' and position == 1:
                    pca_df.loc[pca_df.index[i], 'Position'] = -1
                    position = 0
                    sell_signals.append(pca_df.index[i])
                else:
                    pca_df.loc[pca_df.index[i], 'Position'] = pca_df['Position'].iloc[i - 1]

            pca_df['Strategy_Returns'] = pca_df['Close'].pct_change() * pca_df['Position'].shift(1)

            # 累積リターンを計算
            pca_df['Cumulative_Strategy_Returns'] = (pca_df['Strategy_Returns'] + 1).cumprod()
            pca_df['Cumulative_Market_Returns'] = (pca_df['Close'].pct_change().fillna(0) + 1).cumprod()

            # 結果のプロット
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(pca_df.index, pca_df['Cumulative_Market_Returns'], label='Market Returns', color='blue')
            ax.plot(pca_df.index, pca_df['Cumulative_Strategy_Returns'], label='Strategy Returns', color='orange')
            ax.scatter(buy_signals, pca_df.loc[buy_signals]['Cumulative_Strategy_Returns'], marker='^', color='g',
                       label='Buy Signal', s=100)
            ax.scatter(sell_signals, pca_df.loc[sell_signals]['Cumulative_Strategy_Returns'], marker='v', color='r',
                       label='Sell Signal', s=100)
            ax.set_title(f'Cumulative Returns: {ticker} (Buy in Cluster 0, Sell in Cluster 2)')
            ax.set_xlabel('Date')
            ax.set_ylabel('Cumulative Returns')
            ax.legend()
            st.pyplot(fig)
            figures.append(fig)  # グラフをリストに追加

            # 最終的なリターンを表示
            final_market_return = pca_df['Cumulative_Market_Returns'].iloc[-1]
            final_strategy_return = pca_df['Cumulative_Strategy_Returns'].iloc[-1]

            st.write(f"Final Market Return: {final_market_return}")
            st.write(f"Final Strategy Return: {final_strategy_return}")

            # 主成分のペアプロット
            fig2 = sns.pairplot(pca_df, hue='Cluster', vars=['PC1', 'PC2'])
            st.pyplot(fig2)
            figures.append(fig2)  # ペアプロットをリストに追加

            # 最新の終値と投資判断を表示
            latest_close = data['Close'].iloc[-1]
            latest_decision = pca_df['Decision'].iloc[-1]

            st.write(f"Latest Close Price: {latest_close}")
            st.write(f"Investment Decision: {latest_decision}")


    # ダウンロードボタンを追加する関数
    def download_dataframes(df, figures):
        output = io.BytesIO()

        # データフレームをExcelファイルに保存
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Results', index=False)

            for i, fig in enumerate(figures):
                sheet_name = f'Figure_{i + 1}'
                fig.savefig(f'{sheet_name}.png', format='png')
                worksheet = writer.sheets['Results']
                worksheet.insert_image(f'H{i * 25 + 1}', f'{sheet_name}.png')

        data = output.getvalue()
        return data


    # グラフとデータのダウンロードボタンを追加
    if st.button('Download Results'):
        data = download_dataframes(results_df, figures)
        st.download_button(label='Download Results', data=data, file_name='results.xlsx',
                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')