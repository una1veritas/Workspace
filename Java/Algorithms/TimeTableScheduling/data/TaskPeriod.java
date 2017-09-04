package data;

/**
 * @author masayoshi
 *
 * TODO ��Ƃƍ�Ǝ��Ԃ̑g
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
	 * @return period_id ��߂��܂��B
	 */
	public int getPeriod_id() {
		return period_id;
	}
	/**
	 * @param period_id period_id ��ݒ�B
	 */
	public void setPeriod_id(int period_id) {
		this.period_id = period_id;
	}
	/**
	 * @return task_id ��߂��܂��B
	 */
	public int getTask_id() {
		return task_id;
	}
	/**
	 * @param task_id task_id ��ݒ�B
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
