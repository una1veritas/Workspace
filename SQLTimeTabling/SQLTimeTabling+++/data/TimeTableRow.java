/*
 * 作成日: 2007/04/02
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
public class TimeTableRow {
	//講義のid
	private int task_id;
	//時間のid
	private int period_id;
	//講師のid
	private int processor_id;
	
	public TimeTableRow(int task_id, int period_id, int processor_id){
		this.task_id = task_id;
		this.period_id = period_id;
		this.processor_id = processor_id;
	}
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
	
	public String toString(){
		return "("+task_id+","+period_id+","+processor_id+")";
	}
	
	public boolean equals(Object o){
		if(!(o instanceof TimeTableRow))
			return false;
		TimeTableRow ttr = (TimeTableRow)o;
		return (ttr.task_id==task_id)&&(ttr.period_id==period_id)&&(ttr.processor_id==processor_id);
	}
	
	public int hashCode(){
		return task_id*10000+period_id*100+processor_id;
	}
}
