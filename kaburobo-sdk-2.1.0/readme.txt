■■カブロボにご参加いただきましてありがとうございます■■

このパッケージは、カブロボ用の資料です。
ご利用になる際は、web 上に掲載されている利用規約に同意して頂く必要がございます。

パッケージは、必要に応じて都度アップデートしますので、
最新版をお使いになるようにお願いします。


■インストール方法
doc/help/install.html
を参照してください。


■再インストール方法
すでにインストールされている場合は、一度kaburoboフォルダを削除してください。
再度、解凍しインストールしてください。
このとき、kaburoboフォルダ以下にファイルを作成している場合は、
そのファイルをバックアップしておいてください。


■ファイル構成

○ヘルプファイル
doc/help/install.html                   インストール方法
doc/help/robot-config.html              ロボット起動設定用XMLファイル解説
doc/help/makerobot.html                 ロボット作成方法
doc/help/datalist.pdf                   提供データ一覧
doc/help/kaburobo_building_rule.pdf     カブロボ作成ルール/仕様
doc/help/kaburobo_programming_rule.pdf  Java プログラマー用補足資料
doc/javadoc/index.html                  APIドキュメント(javadoc)

○ディレクトリ構成
doc             ドキュメント
config          設定ファイル
robot           サンプルロボット
data**          検証パックディレクトリ
lib             ライブラリ

○検証可能期間（付属株式データ）
2004年1月5日から2006年12月29日までの期間です。

○データについてのご注意
ストップ高や売買停止期間などが銘柄によっては存在し、データが null になる可能性が
あります。
プログラミングをする際にはnullチェックを必ず行うようにしてください。


■変更点
いくつか修正がありましたので、カブロボ SDK をアップデートしました。

2008/04/14 version 2.1.0
【新機能】新しいテクニカル関数として騰落レシオが実装されました。AnalysisManager#getUpDownRatio()から実行出来ます。
【改良】自動損切り値をrobot-config.xmlに記述することで、ユーザーによる設定が可能になりました。
【修正】InformationManager#getEtfNikkei225List および #getEtfTopixList の戻り値が不正だった問題を修正しました。

2008/01/21 version 2.1.0 RC2
【改良】成績表作成ツールのETF対応に伴う内部修正を行いました。
【修正】2.1.0 RC1より最終成績表作成時に例外が発生することがあった問題を修正しました。

2008/01/07 version 2.1.0 RC1
【改良】上場前銘柄を、InformationManager#getStockList(Category) にて取得出来ないようにしました。
【修正】InformationManager#getEtfList() にて null を含んだ ArrayList が返る場合があった問題を修正しました。
【修正】株式分割または併合にて端株が売れない問題を修正しました。
        (配布しているSDKでは発生いたしません。日々の運用にて発生する場合がありました)

2007/11/12 version 2.1.0 beta1
【新機能】ETF(TOPIX、日経平均株価225種)の売買に対応しました。
          詳しくは添付の doc/help/makerobot.html の「ETFの利用について」をご覧ください。
【新機能】簡単に降順ソートができる QuickSort クラスを追加しました。
【改良】2値で成否を返すメソッド EnumTradeResult#isSucceeded() を追加しました。
【修正】PortfolioManager#getPortfolioByDate,Intervalにおいて株数０要素１のリストが返される事があるのを修正

2007/07/10 version 2.0.3
【修正】信用残系のランキングがまれに取得出来ない事があるのを修正
【修正】MarketRankingにおいて期間外を指定した際にエラーが発生する問題を修正
【修正】アップティックルールのチェックを指値価格の丸めより前に行うように修正

2007/06/22 version 2.0.2
【修正】InformationManagerから銘柄リストを取得した場合に非対象銘柄が返る場合があった問題を修正
【修正】InformationManager#getMarketIndexByInterval, getMarketExchangeByInterval, getMarketDetailByIntervalの
        返すリストに終了指定日付のデータが含まれていない問題を修正

2007/05/29 version 2.0.1
【新機能】PortfolioManagerに指定銘柄を保有しているか取得するメソッドを追加
【修正】特別な環境下において銘柄組み入れ比率などの設定が正しく反映されない問題を修正

2007/05/22 version 2.0.0
【改良】指数平滑移動平均(EMA)の計算精度を向上し、デフォルトのデータ読み込み数を集計数の4倍に変更
【改良】修正移動平均(RMA)の計算精度を向上し、デフォルトのデータ読み込み数を集計数の2倍に変更
【修正】InformationManager#getStock****ByIntervalの戻値のリストに「開始指定日時＝４本値引け時刻」も含めるように修正
【変更】robot-config.xml を変更し、デフォルトのバックテスト期間を2006年の1年間としました。

2007/04/27 version 2.0.0 rc2
【改良】クロスプラットフォームにおける出力表示フォーマットを調整
【改良】最終成績表において、ポートフォリオが巨大になった場合の対応を改良
【改良】MovingAverage#getBaseIndexExponentialArray(double[], int, int) を追加しました。
【改良】double[] 対応の getBaseIndex() を以下のテクニカル指標クラスに追加しました。
　　　　・BollingerBand
　　　　・RSI
　　　　・VolumeRatio
　　　　・DMI
　　　　・Envelope
　　　　・Momentum
【修正】TimeManager#getStockDetailByDate(Calendar), TimeManager#getStockDetailByDate(Calendar, Stock)において、
　　　　2.0.0 RC1 より直近のデータを返していた問題を修正
【修正】getStockDetail() 系メソッドにおいて、スクリーニング時当日の 5 指標が取得できていた問題を修正
　　　　以下 5 指標についてはスクリーニング時当日は null が返ります。
　　　　・信用買い残 (getCreditBuying)
　　　　・信用売り残 (getCreditBuying)
　　　　・逆日歩     (getNegativeIpd)
　　　　・融資残     (getLoans)
　　　　・貸株残     (getStockLoans)
【修正】条件付特別注文の約定条件の不具合を修正
【修正】銘柄組入比率を掛ける基準値を取引余力から資産評価額に修正
【修正】Rオシレーターの計算不具合を修正、デフォルトの集計日数を20に変更
【修正】ストキャスティクスの計算不具合を修正

2007/04/03 version 2.0.0 rc1
【新機能】条件付特別注文を追加しました。OrderManager#setLastOrderSpecial で利用可能です。
【改良】逆指値注文の約定価格の仕様を変更。注文価格より１呼値単位分、買いは高く、売りは安い約定価格になります。
【改良】不正防止の為、OrderManager#addOrder()をDeprecatedに変更
【改良】成績解析の精度を向上
【修正】beta2で廃止されたOrderManager#checkMaxOrderQtyとOrderManager#checkMaxOrderQtyReverseを互換性維持のため復元
【修正】新規買い逆指値注文の必要取引余力を成行注文と同等の計算方法に変更

2007/03/13 version 2.0.0 beta2
【新機能】InformationManager#getListedDate()に銘柄上場日を返す関数を追加
【新機能】InformationManager#getStockList(int date)に指定有効日数以上の銘柄リストを返す関数を追加
【新機能】InformationManager#getAvailableDateCount()に有効日数を返す関数を追加
【新機能】Stock#getAvailableDateCount()に有効日数を返す関数を追加
【改良】発注時の取引余力計算に手数料も同時に考慮するように改良
【改良】上場直後銘柄のテクニカル分析の計算方法を改良
【改良】上場直後銘柄のMarketRankingの計算方法を改良
【改良】決算データの取得可能日を現実的な２ヵ月後に変更
【修正】OrderManager#cancelOrderByIndex()でキャンセル注文を取り消すとNullPointerExceptionが発生する問題を修正
【修正】MovingAverage#getIndexExponentialにおいてIndexOutOfBoundsExceptionが発生する問題を修正
【修正】OrderManager#checkMaxOrderQtyReverseにおいてArithmeticExceptionが発生する問題を修正

2007/03/01 version 2.0.0 beta1
【新機能】4本値データのキャッシュすることで計算量の多いロボットのバックテストを高速化
【新機能】統計的な数学関数を扱うKaburoboMathクラスを追加（staticメソッドのみのライブラリクラス）
【新機能】各注文に対して、どのファイルの何行目から出たものなのかを自動集計
【新機能】全ての注文に対して個別に注文理由をつけられるメソッドを追加
【新機能】MarketDetail（市況情報）のデータを追加し、APIの仕様を変更
　　　　　出来高、売買代金、値上がり株数、値下がり株数、変わらず株数、総株数、
　　　　　値付け率、時価総額、単純平均、加重平均
【新機能】Java6に対応
【新機能】Stock.getCreditTradeType()で信用貸借銘柄の区別を取得出来る機能を追加
【新機能】Portfolio及びPortfolioHistoryに以下のメソッドを追加
　　　　　保有日数（日）、評価損益（円）、評価損益（％）、現在株価（円）
【新機能】出力ログのポートフォリオをより詳細に把握出来るように表示項目を追加
【新機能】検証パックの切り替えを設定ファイルにて変更出来る機能を追加（ディレクトリ名を指定）
【新機能】初期資金を設定ファイルで変更できる機能を追加
【新機能】新規注文の銘柄組入比率上限値を設定ファイルで変更できる機能を追加
【新機能】信用売りの有無を設定ファイルで変更できる機能を追加
【改良】過去５日間の出来高平均による注文量上限値を変更
【改良】InformationManager.getStockList()で上場済み銘柄のみ返すようにする（getStockListAllで既存の動作）
【改良】本戦環境と同様に、スクリーニング時の当日海外指標・外国為替データはNullが返るように変更
【改良】SDKの付属データを50銘柄を標準とするように変更
【改良】4本値のデータに限りNullPointerExceptionが出ないように変更（出来高0の日のデータ仕様を変更）
【改良】テクニカル分析においてNullPointerExceptionが出ないように変更
【改良】最終成績表において、最終ポートフォリオを最終日の引けで反対売買したとみなすように変更
【改良】最終成績表の表示を変更
【改良】逆指値注文の必要取引余力を指値注文と同等の計算方法に変更
【改良】複数タイムゾーンに対応するように改良
【修正】値幅制限が計算出来ない場合に取引余力以上の注文が出せてしまう問題を修正
【修正】MarketIndexの期間指定においてデータが取得できなかった問題を修正
【修正】BolingerBandをBollingerBandに修正

2006/09/04 version 1.1.3
【新機能】AnalysisManager#getFunctionInMovingAverage()
	以下の"関数In移動平均"（単純、修正、加重）のメソッド群を追加
	・RSI
	・%Rオシレーター
	・ボリュームレシオ
	・ストキャスティクス
	・サイコロジカルラインKとD
	・モメンタム
	・出来高
	・真の高値・真の安値・真の値幅
【新機能】DMIにおいて"真の高値（TrueHigh）"・"真の安値"・"真の値幅"を実装
【改良】TimeManager#getClosingDateListでデータが無い範囲を指定しても要素0のリストを返すように変更
【改良】MACDのデフォルトEMA最大読み込み数を中期集計数の5倍に変更
【改良】MovingAverageのデフォルトEMA最大読み込み数を集計数の5倍に変更
【修正】MarketIndex(日経平均・TOPIX他)の取得日付の誤差を修正
【修正】GoldenCrossのNull問題を修正
【修正】MovingAverageの修正移動平均の計算方法を修正

2006/08/21 version 1.1.2
【改良】オブジェクトメモがシリアライズされていない場合にExceptionを出すように変更
【改良】新エントリー基準に対応（トータル約定金額2000万以上に変更）
【改良】ファイルログ出力時に既存のファイルが存在したら上書きするように変更
【改良】エントリーツール使用時にはKABUROBO_HOME/log.txtにログを出力するように変更
【改良】CategoryをSerializableに変更

2006/08/11 version 1.1.1
【新機能】以下のテクニカル指標を追加
	・ストキャスティクス
	・ゴールデンクロス
	・RCI
	・StockPriceRC
	・サイコロジカルライン
	・Rオシレーター
【新機能】Stock#getDisableDate()を本戦用に提供開始（バックテスト時においては無いので注意して下さい）
【改良】全枚数の反対売買の際に、未単元株を注文出来るように変更
【修正】RobotLogManagerのレベルを1～5の範囲になるように修正
【修正】MarketRankingにおいて、PERとPBRが-1の銘柄を除くように修正
【修正】同一のキャンセル注文が複数出せてしまう問題を修正
【修正】Ichimoku#getIndexLeadSpan1でのNull問題を修正
【修正】値幅制限の計算方法を修正

2006/07/29 version 1.1.0
・コンテスト本戦対応

2006/07/27 version 1.0.5
【重要変更】カブロボの約定方法を一部改定　詳細：doc/help/kaburobo_execution_rule.pdf→programing_rule.pdfに統一
【重要変更】注文時において、過去５日間の平均出来高の5％を一度に出来る注文上限枚数に設定
【新機能】OrderManager#orderCancelAll()を追加
【新機能】反対売買において、後場にキャンセルをしてから注文内容を変えての再注文を可能に
【改良】OrderManager#cancelOrder()系を上記機能追加に伴い変更
【改良】決算データ・PER/PBRを連結決算ベースに変更
【修正】OrderManager#orderDetail()を修正
【修正】RSIの計算でNullが出る事があるのを修正
【修正】AnalysisManager#getMomentum()のスペルミスを修正
【修正】一度に複数日のランキングが計算出来ない問題を修正
【修正】信用取引時の金利・口座管理料が正しく引かれない問題を修正
【修正】Portfolio#getFirstExecQty()でNullが返ることがある問題を修正

2006/07/20 version 1.0.4
【新機能】PerformanceAgentに運用日数と経過日数を名称変更して追加
【改良】TimeManagerの高速化に伴い、SDK全体の速度が向上
【修正】キャンセル注文とシステム自動損切りが同時に起こる際のNullPointerExceptionを修正

2006/07/16 version 1.0.3
【新機能】PerformanceAgentに年次利回りボラティリティを追加
【改良】PerformanceAgentの年次平均損益の計算方法を変更
【改良】PerformanceAgentのシャープレシオの計算方法を変更
【修正】VolumeRatioの計算式を修正
【修正】AnalysisManagerの速度が遅くなっていたのを修正

2006/07/07 version 1.0.2
【新機能】OrderHistoryManager#getOrderHistoryByCustomerTradeId(Integer customerTradeId)を追加
【新機能】PortfolioManager#getPortfolio(Integer tradeId)を追加
【新機能】各種テクニカル指標にn回前に遡って指標を取れるメソッドを追加（前日のRSI等が取得可能になりました）
【改良】Portfolio,PortfolioHistory,OrderHistoryのSerializable実装
【修正】PerformanceAgent#getProfitAll()を修正
【修正】PerformanceAgent#getRiskRatio()を修正
【修正】NotExecutedOrder#orderCancel()を修正
【修正】空売りの際の必要金額計算において値幅制限まで考慮していなかった問題を修正
【修正】空白を含むKABUROBO_HOMEのパスの場合にロボットの実行ができない問題を修正

2006/06/26 version 1.0.1
【新機能】OrderHistoryManager#getOrderHistoryByTradeId(Integer TradeId)を追加
【新機能】Portfolio#getFirstExecQty()を追加
【新機能】PortfolioHistory#getFirstExecQty()を追加
【新機能】OrderHistory#isNewTrade()を追加
【新機能】CostManagere#getCorrectCreditCost(Portfolio portfolio)を追加
【新機能】OrderHistoryManager#getOrderHistoryByDay(Calendar date)を追加
【改良】Portfolio#getCreditCostAll()をgetCreditCost()に変更
【改良】Portfolio#getPortfolioHistoryId()を削除
【改良】赤字の場合のPERをnullから-1に変更
【修正】年初来高値・安値がnullだったのを修正
【修正】StockのHashCodeが正しくなかった問題を修正
【修正】InformationManager#getMarketRanking(EnumMarketRankingType type)を修正
【修正】InformationManager#getStockAccountByDate(Calendar date)を修正
【修正】AssetManager#getAssetHistory(Calendar date)を修正
【修正】TimeManager#getExecDateList(Calendar date1,Calendar date2)を修正
【修正】OrderHistoryManager#getOrderHistoryByDate(Calendar date)を修正
【修正】OrderHistoryManager#getOrderHistoryByInterval(Calendar date1,Calendar date2)を修正
【修正】同一TradeIDの注文が、前場と後場に分けて出すことで２重に出せてしまう問題を修正
【修正】検証期間のチェックの問題を修正

2006/06/20 version 1.0.0
【新機能】提供データ（datalist.pdf）にMarketRankingを追加
【新機能】InformationManager#getCategory(Integer categoryCode)を追加
【改良】InformationManager#getMarketRankigByIntervalを削除
【改良】パフォーマンス評価ツールの「年間平均純利益」を「平均年間純損益」に変更
【改良】提供データのStockDetailの逆日歩・融資残・貸株残を変更
【修正】キャンセル注文時のバグを修正
【修正】提供データの米国Indexのnullを修正
【修正】MarketIndexの未来データが見えてしまうバグを修正
【修正】PerformanceAgentの表示バグを修正
【修正】運用期間が１日の時に発生する問題を修正
【修正】InformationManager#getStockAccount(Calendar date)のバグを修正

2006/06/16 version 1.0.0 rc3
・各種バグ修正
・StockCodeを引数にしていたメソッドをStockクラスの引数に変更
・CategoryCodeをCategoryクラスへ移行
・データクラス系をインターフェースに変更
・MACDの高速化
・約定処理の高速化
・データベースを一部修正
・パフォーマンスチェックツールの日経平均を資産曲線レートに適合
・バックテスト用の50銘柄拡張検証パックの追加（エントリー用では無い）

2006/06/09 version 1.0.0 rc2
・各種バグ修正
・TimeManager.getBusinessDay(Calender date, Integer days)の追加
・TimeManager.getBusinessDay(Integer days)の追加
・TimeManager.getBusinessDayList(Calendar pastDate, Calendar futureDate)の追加
・TimeManager.getNextBusinessDay(Integer days)の削除
・TimeManager.getNextBusinessDay(Calendar date, Integer days)の削除
・TimeManager.getLastBusinessDay(Integer days)の削除
・TimeManager.getLastBusinessDay(Calendar date, Integer days)の削除
・AbstractRobot.terminateメソッドの変更
・SDKの成績表示機能追加
・カブロボパフォーマンス評価ツールの追加
・プログラミング詳細ルールのドキュメント追加

2006/06/01 version 1.0.0 rc1
・各種バグ修正
・テクニカル分析に以下の指標を追加
	相対力指数		RSI
	移動平均		MovingAverage
	一目均衡表		Ichimoku
	ボリュームレシオ	VolumeRatio
	ボラティリティ	Volatility
	ボリンジャーバンド	BolingerBand
	ＤＭＩ		DMI
	エンベロープ		Envelope
	ＨＬバンド		HLBand
	指数平滑化平均	MACD
	モメンタム		Momentum
	黄金分割比		GoldenSectionRatio
・提供データの追加
	(datalist.pdfを参照)

2006/05/30 version 1.0.0 beta2
・各種バグの修正
・整合性のない注文のチェック機構の追加
・AbstractRobot.terminateメソッドの追加
・TimeManager.getScreeningDateメソッドの削除
・RobotLogManager.showScreeningStartLogメソッドの削除
・RobotLogManager.showScreeningEndtLogメソッドの削除
・RobotLogManager.showOrderStartLogメソッドの削除
・RobotLogManager.showOrderEndLogメソッドの削除
・ロボット設定ファイルに<system-log>タグの追加
・AbstractRobot.runメソッドをorderメソッドに名前変更
・Mac Unixでの動作確認（kaburobo.shの追加）
・Stockクラスでの注文メソッドの追加
・業種データの提供（Stockクラス）
・従来のStockクラスとStockAllクラスをStockインターフェースクラスに統合
・ScreeningManagerクラスの削除
・AbstractObjectMemoの削除
・Stringメモの廃止
・提供銘柄を50銘柄から20銘柄に変更

2006/05/24 version 1.0.0 beta1
・ベータ公開

■ご注意事項
*このSDKは、カブロボにご参加いただく以外の目的ではご利用になれません。
また、独自に配布や流用することはできません。株価等のデータの正確性について保証しておりません。

(c) 2004-2008 kaburobo.jp and Trade Science Corp. All rights reserved.
http://kaburobo.jp/