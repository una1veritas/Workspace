/*
 * 作成日: 2006/12/08
 *
 * TODO この生成されたファイルのテンプレートを変更するには次へジャンプ:
 * ウィンドウ - 設定 - Java - コード・スタイル - コード・テンプレート
 */
package data;

/**
 * @author masayoshi
 *
 * TODO この生成された型コメントのテンプレートを変更するには次へジャンプ:
 * ウィンドウ - 設定 - Java - コード・スタイル - コード・テンプレート
 */
public class SQL {
	private String query;
	private int type;
	
	public SQL(int t, String q) throws Exception {
		if ( t == 3 )
			throw new Exception("case 3 occurred!");
		type = t;
		query = q;
	}
	
	/**
	 * @return sql を戻します。
	 */
	//public String getSql() {
	public String query() {
		return query;
	}
	/**
	 * @param sql sql を設定。
	 */
	public void query(String str) {
		query = str;
	}
	/**
	 * @return type を戻します。

	public int type() {
		return type;
	}
		 */
	/**
	 * @param type type を設定。
	 */
	public void type(int t) throws Exception {
		if ( t == 3 )
			throw new Exception("case 3 occurred!");
		type = t;
	}
	
}
