

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

import jp.tradesc.superkaburobo.sdk.trade.data.Stock;

/**
 * オブジェクトメモクラスです。<br>
 * スクリーニングの結果を格納します。
 * @author (c) 2004-2008 kaburobo.jp and Trade Science Corp. All rights reserved.
 */
public class SampleObjectMemo implements Serializable{
	private static final long serialVersionUID = 5651047511229005177L;
	
	/**
	 * 注文対象リスト
	 */
	private ArrayList<SampleObjectRecord> memoList  = new ArrayList<SampleObjectRecord>();
	
	/**
	 * 注文対象リストを返します
	 * @return 注文対象の ArrayList です。null が返されることはありません。
	 */
	public ArrayList<SampleObjectRecord> getMemoList() {
		return memoList;
	}
	
	/**
	 * 購入銘柄リストを設定します。<br>
	 * null を設定された場合、null ではなく空の ArrayList が設定されます。
	 * @param memoList 設定するメモリスト
	 */
	public void setMemoList(ArrayList<SampleObjectRecord> memoList) {
		if (null == memoList){
			this.memoList.clear();
		} else {
			this.memoList = memoList;
		}
	}
	
	/**
	 * 注文予定リストをシグナルの強さによりソートします。
	 */
	public void sortStockMemoByStrength(){
		Collections.sort(memoList, new stockMemoSorter());
	}

	// コンパレータクラスです。シグナルの強さを比較します。
	private static class stockMemoSorter implements Comparator<SampleObjectRecord>, Serializable{
		private static final long serialVersionUID = 5532134792677963169L;
		public int compare(SampleObjectRecord o1, SampleObjectRecord o2) {
			return (int) Math.round(o2.getStrength() - o1.getStrength());
		}
		
	}
	
	/** 
	 * 対象銘柄が注文予定リストに含まれていないかを確認します。
	 * @param isBuy 買い注文を調べたい場合 true、売り注文を調べたい場合は false
	 * @param stock この銘柄が
	 * @return 見つかれば true、見あたらなければ false が返ります。
	 */
	public boolean contain(boolean isBuy, Stock stock){
		int stock_code = stock.getStockCode().intValue();
		for (SampleObjectRecord memo: memoList) {
			if(
				(memo.isBuy() == isBuy) &&
				(memo.getStock().getStockCode().intValue() == stock_code)
			) {
				return true;
			}
		}
		return false;
	}
}
