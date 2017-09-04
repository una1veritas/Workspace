/*
 * 作成日: 2006/12/08
 *
 * TODO この生成されたファイルのテンプレートを変更するには次へジャンプ:
 * ウィンドウ - 設定 - Java - コード・スタイル - コード・テンプレート
 */
package data;

import java.util.ArrayList;

/**
 * @author masayoshi
 *
 * TODO この生成された型コメントのテンプレートを変更するには次へジャンプ:
 * ウィンドウ - 設定 - Java - コード・スタイル - コード・テンプレート
 */
public class ValuateSQLs {
	private ArrayList v_sqls;
	private int weight;
	
	public ValuateSQLs(){
		v_sqls = new ArrayList();
	}
	
	public void addValuateSQL(ValuateSQL v_sql){
		v_sqls.add(v_sql);
	}
	
	/**
	 * @return v_sqls を戻します。
	 */
	public ArrayList getV_sqls() {
		return v_sqls;
	}
	/**
	 * @param v_sqls v_sqls を設定。
	 */
	public void setV_sqls(ArrayList v_sqls) {
		this.v_sqls = v_sqls;
	}
	/**
	 * @return weight を戻します。
	 */
	public int getWeight() {
		return weight;
	}
	/**
	 * @param weight weight を設定。
	 */
	public void setWeight(int weight) {
		this.weight = weight;
	}
}
