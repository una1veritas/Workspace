

import java.util.ArrayList;

import jp.tradesc.superkaburobo.sdk.robot.AbstractRobot;
import jp.tradesc.superkaburobo.sdk.trade.AnalysisManager;
import jp.tradesc.superkaburobo.sdk.trade.AssetManager;
import jp.tradesc.superkaburobo.sdk.trade.EnumAnalysisSpan;
import jp.tradesc.superkaburobo.sdk.trade.InformationManager;
import jp.tradesc.superkaburobo.sdk.trade.MemoManager;
import jp.tradesc.superkaburobo.sdk.trade.OrderManager;
import jp.tradesc.superkaburobo.sdk.trade.PortfolioManager;
import jp.tradesc.superkaburobo.sdk.trade.RobotLogManager;
import jp.tradesc.superkaburobo.sdk.trade.TradeAgent;
import jp.tradesc.superkaburobo.sdk.trade.analysis.technicalindex.MovingAverage;
import jp.tradesc.superkaburobo.sdk.trade.analysis.technicalindex.RSI;
import jp.tradesc.superkaburobo.sdk.trade.data.Portfolio;
import jp.tradesc.superkaburobo.sdk.trade.data.Stock;
import jp.tradesc.superkaburobo.sdk.trade.data.StockData;

/**
 * サンプルロボット
 * 
 * @author (c) 2004-2008 kaburobo.jp and Trade Science Corp. All rights reserved.
 * 
 * 
 */
public class SampleRobot extends AbstractRobot {
	@Override
	public void order(TradeAgent tradeAgent) {	
		// 保存してあるメモを取得します。
		SampleObjectMemo memo = (SampleObjectMemo) tradeAgent.getMemoManager().getObjectMemo();
		// メモが null の時のことを考えて null チェックをします。
		if(memo == null) return;
		
		// 反対売買注文を必要に応じて行います。
		orderReverseByRate(15, -6, 60); // 15%で利確し、-6%で損切りし、60日保有で強制反対売買します。
		orderReverseBySignal(memo);     // シグナルと反対のポジションを解消します。
		
		// 新規注文を行います。
		orderNew(memo);
		
		// 後場に同じものを購入しないように、メモの注文予定リストを削除します。
		// 前場に約定しなかった注文は自動的に後場に引き継がれますので問題ありません。
		memo.setMemoList(null);
		tradeAgent.getMemoManager().setObjectMemo(memo);
	}
	
	/**
	 * 新規売買注文を行います。
	 * @param memo メモオブジェクト
	 */
	protected void orderNew(SampleObjectMemo memo){
		// 総資産の10パーセントを１銘柄に投資する上限とします。
		long orderMoney = (long)(AssetManager.getInstance().getTotalAssetValue() * 0.1);
		
		// 各種マネージャを取得します。
		InformationManager im = InformationManager.getInstance();
		OrderManager       om = OrderManager.getInstance();
		
		// スクリーニング結果の銘柄を注文します。
		for (SampleObjectRecord stockmemo: memo.getMemoList()) {
			Stock stock = stockmemo.getStock();
			
			// すでに保有している銘柄の場合スキップします。
			if(isHolding(stock)) continue;
			
			// ある銘柄の終値を取得します。失敗したらその銘柄はスキップします。
			StockData stockSession = im.getStockSession(stock);
			if(stockSession == null) continue;
			if(stockSession.getClosingPrice() == null) continue;
			int closingPrice = stockSession.getClosingPrice();
			
			// 終値から購入(売却)株数を概算で求め、正確に丸めた数字を算出します。
			// 空売り時でも、数量は絶対値になります。
			int qty = om.checkRoundUnit(stock, (int)(orderMoney / closingPrice));
			
			// 丸めた結果が0の時は注文しません。
			if (qty <= 0) continue;
			
			// 注文の成否を格納する変数 (true:成功, false:失敗)
			boolean result;
			
			// 注文をします。
			if (stockmemo.isBuy()) {
				// 現物買いをします。
				
				// 現物成行注文
				result = om.orderActualNowMarket(stock, qty);
				
			} else {
				// 空売りをします。空売りはアップティックルールにより指値のみです。
				
				// 終値から呼び値を調べます
				int spread = om.checkSpread(closingPrice);
				
				// 指値はアップティックルールを違反しないように、前日終値以下では指しません。
				// 前日終値+1Tickの価格で指します。
				result = om.orderCreditNowLimit(
						stock, 
						closingPrice + spread,
						-1 * qty); // 売り注文を出す場合は、マイナスの株数を入力します
			}
			
			// 注文の成否によって出力を変えます。
			// なお、LastOrderResultによって失敗した時の理由がわかります。
			String retMsg = "";       // 成功/失敗
			if (result){
				retMsg = "成功";
				
				// 注文に成功したので、注文の理由を設定します。
				// この注文の成否をチェックせずに setLastOrder～ メソッドを実行するのは危険です。
				om.setLastOrderReason(stockmemo.getReason());
			} else {
				retMsg = "失敗";
			}
			
			// 買いと売りの文言を切り替えます。
			String buysell = "";
			if (stockmemo.isBuy()) {
				buysell = "買い";
			} else {
				buysell = "売り";
			}
			
			// ログ出力用のメッセージを作ります。
			String msg = String.format("新規%s注文しました:%s 銘柄コード:%4d 注文株数:%,d 注文理由:%s 注文結果:%s",
					buysell,
					retMsg,
					stock.getStockCode(),
					qty,
					stockmemo.getReason(),
					om.getLastOrderResult().toString()
					);
			
			// 作成したメッセージをログに出力します。
			RobotLogManager.getInstance().log(msg, 3);
		}
	}
	
	/**
	 * 反対売買注文を行います。<BR>
	 * 一定の損益率に達したら利益確定または損切りを行います。
	 * @param profitTakingRate 利益確定する利益率(%)です。5.0なら、5%の利益率で利益確定します。<BR>正の値(プラスの値)を入れてください。
	 * @param lossCutRate      損切りをする損失率(%)です。-2.0なら、-2%の利益率(=2%の損失率)で損切りします。<BR>負の値(マイナスの値)を入れてください。
	 * @param maxHoldingDays   最大保有日数です。その日数を超えて保有していたならそのトレードの利益率にかかわらず反対売買します。
	 */
	protected void orderReverseByRate(double profitTakingRate, double lossCutRate, int maxHoldingDays){
		// ポートフォリオマネージャを取得します。
		PortfolioManager pm = PortfolioManager.getInstance();
		
		// ポートフォリオを参照して、必要があれば反対売買を行います。
		ArrayList<Portfolio> portfolioList = pm.getPortfolio();
		for (Portfolio portfolio : portfolioList) {
			String reason = "";
			
			// シグナルが出ていないか調べます。
			// シグナルが出ていれば reason 変数に理由を入れます。
			double rate = portfolio.getProfitRate();
			if (rate > profitTakingRate || rate < lossCutRate) {
				// 反対売買の対象です。
				
				// 理由を設定します。
				if (rate > 1) {
					reason = "利益確定:" + rate;
				} else {
					reason = "損切り:" + rate;
				}
				
			} else if (maxHoldingDays < portfolio.getHoldingDays()) {
				// 保有日数が制限を超えると反対売買します。
				reason = "保有日数制限越え";
			}
			
			// シグナルが出ていれば反対売買します。
			if (0 < reason.length()) {
				orderReverseForPortfolio(portfolio, reason);
			}
		}
	}
	
	/**
	 * 反対売買注文を行います。<br>
	 * シグナルをみて、シグナルと反対のポジションがあればそれを解消します。
	 * @param memo メモオブジェクト
	 */
	protected void orderReverseBySignal(SampleObjectMemo memo){
		// ポートフォリオマネージャを取得します。
		PortfolioManager pm = PortfolioManager.getInstance();
		
		// ポートフォリオを参照して、必要があれば反対売買を行います。
		ArrayList<Portfolio> portfolioList = pm.getPortfolio();
		for (Portfolio portfolio : portfolioList) {
			String reason = "";
			
			// シグナルが出ていないか調べます。
			// シグナルが出ていれば reason 変数に理由を入れます。
			if ((portfolio.getExecQty().intValue() < 0) && 
					(memo.contain(true, portfolio.getStock()))) {
				// 空売りしているのに買いシグナルがでていたら反対売買をします。
				reason = "空売りに買いシグナル";
				
			} else if ((portfolio.getExecQty().intValue() > 0) &&
					(memo.contain(false, portfolio.getStock()))) {
				// 現物買いしているのに売りシグナルがでていたら反対売買をします。
				reason = "現物買いに売りシグナル";
				
			}
			
			// シグナルが出ていれば反対売買します。
			if (0 < reason.length()) {
				orderReverseForPortfolio(portfolio, reason);
			}
		}
	}
	
	/**
	 * 指定のポートフォリオに対して反対売買を実施し、ログに残します。<br>
	 * @param portfolio 反対売買を行いたいポートフォリオ
	 * @param reason    注文の理由
	 */
	protected void orderReverseForPortfolio(Portfolio portfolio, String reason){
		// ポートフォリオから直接反対売買注文が出せます。
		boolean result = portfolio.orderReverseNowMarketAll();
		
		// 文言を注文の成否で切り替えます。
		String retMsg = "";
		if(result){
			retMsg = "成功";
			
			// 注文が成功していたら理由をつけます。
			OrderManager.getInstance().setLastOrderReason(reason);
		} else {
			retMsg = "失敗";
		}
		
		// ログ出力用のメッセージを作ります。
		String msg = String.format(
				"反対売買注文しました:%s 銘柄コード:%4d 注文株数:%,d 注文理由:%s 注文結果:%s",
				retMsg,
				portfolio.getStock().getStockCode(),
				portfolio.getExecQty().intValue(),
				reason,
				OrderManager.getInstance().getLastOrderResult().toString()
				);
		
		// ログを出力します。
		RobotLogManager.getInstance().log(msg, 3);
	}
	
	@Override
	public void screening(TradeAgent tradeAgent) {
		// 分析手法管理クラスを取得します。
		AnalysisManager analysisManager = tradeAgent.getAnalysisManager();
		
		// オブジェクトメモを用意し、null チェックを行います。
		SampleObjectMemo memo = (SampleObjectMemo) MemoManager.getInstance().getObjectMemo();
		if(null == memo) memo = new SampleObjectMemo();
		
		// 注文予定リストを用意し、null チェックを行います。
		ArrayList<SampleObjectRecord> memoList = memo.getMemoList();
		if(null == memoList) memoList = new ArrayList<SampleObjectRecord>();

		// 対象銘柄の取得をします
		ArrayList<Stock> stockList = tradeAgent.getInformationManager().getStockList();
		
		// 分析手法の取得をします。
		// このRSIは日足ベースで10回分のデータを利用する分析を行います。
		RSI rsi = analysisManager.getRSI(EnumAnalysisSpan.DAILY, 10);
		MovingAverage ma20 = analysisManager.getMovingAverage(
				EnumAnalysisSpan.DAILY, 20);
		MovingAverage ma40 = analysisManager.getMovingAverage(
				EnumAnalysisSpan.DAILY, 40);
		
		for (Stock stock : stockList) {
			// 分析手法メソッドクラスから、実際の数値を出力します。
			double rsiVal = rsi.getIndexSimple(stock);
			double maVal20 = ma20.getIndexSimple(stock);
			double maVal40 = ma40.getIndexSimple(stock);
			
			final double rsiBorderBuy  = 25.0;
			final double rsiBorderSell = 90.0;
			
			// - RSI が 25 以下
			// - 40日移動平均が20日移動平均を下回っている
			// の両方を満たす場合買いと見なします。
			
			// - RSI が 90 以上
			// - 20日移動平均が40日移動平均を下回っている
			// の両方を満たす場合売りと見なします。
			
			if (rsiVal <= rsiBorderBuy && maVal20 > maVal40) {
				// 購入対象銘柄が見つかったら購入予定を追加します。
				
				// シグナルの強さを算出します。
				double strength = (rsiBorderBuy - rsiVal) + (maVal20 - maVal40);
				
				// 購入予定に追加
				memoList.add(new SampleObjectRecord(stock, true, strength, "RSI 買いシグナル発生:" + rsiVal));
				
			} else if (rsiVal >= rsiBorderSell && maVal20 < maVal40) {
				// 売却対象銘柄が見つかったら売却予定を追加します。
				
				// シグナルの強さを算出します。
				double strength = (rsiVal - rsiBorderSell) + (maVal40 - maVal20);
				
				// 売り予定に追加
				memoList.add(new SampleObjectRecord(stock, false, strength, "RSI 売りシグナル発生:" + rsiVal));
			}
		}
		
		// シグナルの強さでソートします。
		memo.sortStockMemoByStrength();
		
		// オブジェクトメモにセットします。
		memo.setMemoList(memoList);
		
		// オブジェクトメモを登録します。
		MemoManager.getInstance().setObjectMemo(memo);
	}
	
	/**
	 * 対象の銘柄がポートフォリオのなかにあるかを調べます。<br>
	 * @param stock 対象銘柄
	 * @return ポートフォリオに含まれている(所有している)場合 true、そうでない場合(stockがnullの場合も) false 
	 */
	public boolean isHolding(Stock stock){
		// null チェックを行います。
		if (null == stock) return false;
		
		// 検索対象の銘柄コードを取得します。
		int target_code = stock.getStockCode().intValue();
		
		// ポートフォリオを取得します。
		ArrayList<Portfolio> portfolios = PortfolioManager.getInstance().getPortfolio();
		
		// ポートフォリオリストのなかから一致するものを探します。
		for (Portfolio portfolio: portfolios) {
			// ポートフォリオの銘柄コードを取得する
			int stockcode = portfolio.getStock().getStockCode().intValue();
			
			// ポートフォリオの銘柄コードと、検索対象の銘柄コードが一致するか調べます。
			if(stockcode == target_code) return true; // 一致したら所持していることになります。
		}
		// 見あたりませんでした
		return false;
	}
}
