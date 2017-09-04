/*
 * 作成日: 2006/12/03
 *
 * TODO この生成されたファイルのテンプレートを変更するには次へジャンプ:
 * ウィンドウ - 設定 - Java - コード・スタイル - コード・テンプレート
 */
package data;

import java.util.ArrayList;

/**
 * @author 浦島太郎
 *
 * TODO この生成された型コメントのテンプレートを変更するには次へジャンプ:
 * ウィンドウ - 設定 - Java - コード・スタイル - コード・テンプレート
 */
public class PeriodTeacher {
	private int period_id;
	private ArrayList teacher_ids;
	
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
	/**
	 * @return teacher_ids を戻します。
	 */
	public ArrayList getTeacher_ids() {
		return teacher_ids;
	}
	/**
	 * @param teacher_ids teacher_ids を設定。
	 */
	public void setTeacher_ids(ArrayList teacher_ids) {
		this.teacher_ids = teacher_ids;
	}
}
