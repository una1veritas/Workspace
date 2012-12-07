package data;

/**
 * @author masayoshi
 *
 * TODO 作業と作業時間の組
 *
 */
public class TaskPeriod {
	private int task_id;
	private int period_id;
	
	public TaskPeriod(int task_id, int period_id){
		this.task_id = task_id;
		this.period_id = period_id;
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
		if(!(o instanceof TaskPeriod))
			return false;
		TaskPeriod tp = (TaskPeriod)o;
		return (tp.task_id==task_id)&&(tp.period_id==period_id);
	}
	
	public int hashCode(){
		return task_id*100+period_id;
	}
}
