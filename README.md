# PowerX Strategy with Basic Monte Carlo

MT5用のPowerXストラテジーとBasic Decomposition Monte Carlo法を組み合わせたトレーディングシステムの実装です。

## 機能

- **マルチタイムフレーム分析**: 複数の時間足を使用して、より堅牢な取引判断
- **複数のテクニカル指標の組み合わせ**:
  - RSI (相対力指数)
  - ストキャスティクス
  - MACD (移動平均収束拡散法)
  - SuperTrend
  - ATR (Average True Range)
- **基本分解モンテカルロ法によるポジションサイジング**:
  - トレード結果に基づいて数列を更新
  - 勝った場合: 数列の左端と右端の数字を削除
  - 負けた場合: ベット額を右端に追加
  - 次のベット額 = 左端と右端の数字の合計
- **自動SL/TPの設定**: ATRに基づいた動的なストップロスとテイクプロフィットの設定
- **MT5との統合**: MetaTrader 5プラットフォームとの直接接続

## 要件

- Python 3.8以上
- MetaTrader 5がインストールされ、実行されていること
- 以下のPythonパッケージ:
  - MetaTrader5
  - numpy
  - pandas
  - PyYAML
  - TA-Lib (技術的指標計算用)

## インストール

1. リポジトリをクローン:
```bash
git clone https://github.com/yourusername/MTF_Weighted_Dominance.git
cd MTF_Weighted_Dominance
```

2. 仮想環境を作成してアクティブ化:
```bash
python -m venv .venv
# Windowsの場合
.venv\Scripts\activate
# Linuxの場合
source .venv/bin/activate
```

3. 依存関係をインストール:
```bash
pip install -e .
```

**注意**: TA-Libは事前にシステムにインストールされている必要があります。インストール方法は[TA-Libの公式ドキュメント](https://ta-lib.org/)を参照してください。

## 使用方法

1. 設定ファイルを作成:
```bash
python main.py
```
これにより、デフォルトの設定ファイル`config.yaml`が作成されます。

2. 設定ファイルを編集して、MT5の接続情報とストラテジーのパラメータを設定します。

3. ストラテジーを実行:
```bash
python main.py
```

または、コマンドライン引数を使用して設定を上書きすることもできます:
```bash
python main.py --symbol USDJPY --timeframe M5 --higher-timeframe H1 --login 12345 --password yourpassword --server YourBroker-Live
```

## 設定パラメータ

設定ファイル`config.yaml`には以下のパラメータがあります:

```yaml
mt5:
  login: null  # MT5アカウントログイン
  password: null  # MT5アカウントパスワード
  server: null  # MT5サーバー名
strategy:
  symbol: EURUSD  # 取引通貨ペア
  timeframe: M15  # 取引時間足
  higher_timeframe: H1  # 上位時間足
  rsi_period: 7  # RSI期間
  stoch_k_period: 14  # ストキャスティクス %K期間
  stoch_smooth_period: 3  # ストキャスティクスの平滑化期間
  macd_fast_period: 12  # MACD速い期間
  macd_slow_period: 26  # MACD遅い期間
  macd_signal_period: 9  # MACDシグナル期間
  atr_period: 14  # ATR期間
  supertrend_multiplier: 4.0  # SuperTrend乗数
  supertrend_period: 10  # SuperTrend期間
  sl_multiplier: 1.5  # ストップロスATR乗数
  tp_multiplier: 3.0  # テイクプロフィットATR乗数
  allow_longs: true  # ロングエントリーを許可
  allow_shorts: true  # ショートエントリーを許可
execution:
  check_interval: 5  # チェック間隔（秒）
```

## ライセンス

MITライセンスの下で公開されています。詳細については[LICENSE](LICENSE)ファイルを参照してください。

## 注意事項

このソフトウェアは「現状のまま」提供され、いかなる保証もありません。実口座での使用は自己責任で行ってください。十分なテストを行い、トレーディングリスクを理解した上でご使用ください。
