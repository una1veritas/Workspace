/*
 * 作成日: 2006/11/27
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
public class Teacher {
	private int processor_id;
	//担当可能時間
	private ArrayList periods;
	//0:Japanese based, 1:English based, 2:don't care
	private ArrayList qualifications;
	//0:常勤講師 1:非常勤講師
	//private int employment;
	//一週間の担当下限
	//private int total_periods_lb;
	//一週間の担当上限
	//private int total_periods_ub;
	//担当日数の下限
	//private int total_days_lb;
	//担当日数の上限
	//private int total_days_ub;
	//private int wage_level;
	
	/**
	 * @return processor_id を戻します。
	 */
	public int getProcessor_id() {
		return processor_id;
	}
	/**
	 * @param processor_id processor_id を設定。
	 */
	public void setProcessor_id(int processor_id) {
		this.processor_id = processor_id;
	}
	/**
	 * @return periods を戻します。
	 */
	public ArrayList getPeriods() {
		return periods;
	}
	/**
	 * @param periods periods を設定。
	 */
	public void setPeriods(ArrayList periods) {
		this.periods = periods;
	}
	public boolean equals(Object o){
		if(!(o instanceof Teacher))
			return false;
		Teacher t = (Teacher)o;
		return t.processor_id==processor_id;
	}
	public int hashCode(){
		return processor_id;
	}
}
