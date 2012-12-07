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
public class Lecture {
	private int task_id;
	//講義の種類
	private int qualification_id;
	//講師の人数の下限
	private int required_processors_lb;
	//講師の人数の上限
	private int required_processors_ub;
	//開講可能時間
	private ArrayList periods;
	
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
	/**
	 * @return qualification_id を戻します。
	 */
	public int getQualification_id() {
		return qualification_id;
	}
	/**
	 * @param qualification_id qualification_id を設定。
	 */
	public void setQualification_id(int qualification_id) {
		this.qualification_id = qualification_id;
	}
	/**
	 * @return required_processors_lb を戻します。
	 */
	public int getRequired_processors_lb() {
		return required_processors_lb;
	}
	/**
	 * @param required_processors_lb required_processors_lb を設定。
	 */
	public void setRequired_processors_lb(int required_processors_lb) {
		this.required_processors_lb = required_processors_lb;
	}
	/**
	 * @return required_processors_ub を戻します。
	 */
	public int getRequired_processors_ub() {
		return required_processors_ub;
	}
	/**
	 * @param required_processors_ub required_processors_ub を設定。
	 */
	public void setRequired_processors_ub(int required_processors_ub) {
		this.required_processors_ub = required_processors_ub;
	}
	/**
	 * @return task_id を戻します。
	 */
	public int getTask_id() {
		return task_id;
	}
	/**
	 * @param task_id task_id を設定。
	 */
	public void setTask_id(int task_id) {
		this.task_id = task_id;
	}
	public boolean equals(Object o){
		if(!(o instanceof Lecture))
			return false;
		Lecture l = (Lecture)o;
		return l.task_id==task_id;
	}
	public int hashCode(){
		return task_id;
	}
}
