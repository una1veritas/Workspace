/*
 * 作成日: 2007/04/02
 *
 * TODO この生成されたファイルのテンプレートを変更するには次へジャンプ:
 * ウィンドウ - 設定 - Java - コード・スタイル - コード・テンプレート
 */
package data;

import java.io.*;

public class Tablet implements Serializable {
	//講義のid
	private int task_id;
	//時間のid
	private int period_id;
	//講師のid
	private int processor_id;
	
	private Tablet() {
		//
	}
	
	public Tablet(int task, int period, int processor){
		task_id = task;
		period_id = period;
		processor_id = processor;
	}
	
	
	public Tablet(Tablet t){
		task_id = t.task_id;
		period_id = t.period_id;
		processor_id = t.processor_id;
	}
	
	
	public int getPeriod_id() {
		return period_id;
	}
	/**
	 * @param period_id period_id を設定。
	 */
	 
	private void setPeriod_id(int id) {
		this.period_id = id;
	}
	
	/**
	 * @return processor_id を戻します。
	 */
	public int getProcessor_id() {
		return processor_id;
	}
	/**
	 * @param processor_id processor_id を設定。
	 */
	private void setProcessor_id(int id) {
		this.processor_id = id;
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
	private void setTask_id(int id) {
		this.task_id = id;
	}
	
	public String toString(){
		return "("+task_id+", "+period_id+", "+processor_id+")";
	}
	
	public boolean equals(Object o){
		if(!(o instanceof Tablet))
			return false;
		Tablet ttr = (Tablet)o;
		return (ttr.task_id==task_id)&&(ttr.period_id==period_id)&&(ttr.processor_id==processor_id);
	}
	
	public int hashCode(){
		return task_id*10000+period_id*100+processor_id;
	}
}
