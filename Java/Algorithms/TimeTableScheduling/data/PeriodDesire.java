/*
 * 作成日: 2006/11/27
 *
 * TODO この生成されたファイルのテンプレートを変更するには次へジャンプ:
 * ウィンドウ - 設定 - Java - コード・スタイル - コード・テンプレート
 */
package data;

/**
 * @author 浦島太郎
 *
 * TODO この生成された型コメントのテンプレートを変更するには次へジャンプ:
 * ウィンドウ - 設定 - Java - コード・スタイル - コード・テンプレート
 */
public class PeriodDesire {
	//時間のid
    private int period_id;
    //時間の希望度（要望の評価にDBを使用しない時に使用）
    private int preferred_level;
	/**
	 * @return period_id を戻します。
	 */
	public int getPeriod_id() {
		return period_id;
	}
	/**
	 * @param period_id period_id を設定。
	 */
	public void setPeriod_id(int period_id) {
		this.period_id = period_id;
	}
	
	public int getPreferred_level() {
		return preferred_level;
	}
	public void setPreferred_level(int preferred_level) {
		this.preferred_level = preferred_level;
	}
	public boolean equals(Object o){
		if(!(o instanceof PeriodDesire))
			return false;
		PeriodDesire pd = (PeriodDesire)o;
		return pd.period_id==period_id;
	}
	public int hashCode(){
		return period_id;
	}
}
