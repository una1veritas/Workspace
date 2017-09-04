/*
 * 作成日: 2007/07/18
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
public class ValuateSQLsPenalty {
	private ValuateSQLs v_sqls;
	private int penalty;
	
	public ValuateSQLsPenalty(){
		v_sqls = null;
		penalty = 0;
	}
	/**
	 * @return penalty を戻します。
	 */
	public int getPenalty() {
		return penalty;
	}
	/**
	 * @param penalty penalty を設定。
	 */
	public void setPenalty(int penalty) {
		this.penalty = penalty;
	}
	/**
	 * @return v_sqls を戻します。
	 */
	public ValuateSQLs getV_sqls() {
		return v_sqls;
	}
	/**
	 * @param v_sqls v_sqls を設定。
	 */
	public void setV_sqls(ValuateSQLs v_sqls) {
		this.v_sqls = v_sqls;
	}
	public void addPenalty(int penalty) {
		this.penalty += penalty;
	}
}
