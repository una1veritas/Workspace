package data;

public class TaskPeriodNum {
	private int task_id;
	private int period_id;
	private int num;
	
	public TaskPeriodNum(int task_id,int period_id){
		this.task_id = task_id;
		this.period_id = period_id;
		this.num = 1;
	}

	public int getNum() {
		return num;
	}

	public void setNum(int num) {
		this.num = num;
	}

	public int getPeriod_id() {
		return period_id;
	}

	public void setPeriod_id(int period_id) {
		this.period_id = period_id;
	}

	public int getTask_id() {
		return task_id;
	}

	public void setTask_id(int task_id) {
		this.task_id = task_id;
	}
	
	public void updateNum(){
		num++;
	}
	
	public String toString(){
		return "("+task_id+","+period_id+","+num+")";
	}
	
	public boolean equals(Object o){
		if(!(o instanceof TaskPeriodNum))
			return false;
		TaskPeriodNum tpn = (TaskPeriodNum)o;
		return (tpn.task_id==task_id)&&(tpn.period_id==period_id);
	}
	
	public int hashCode(){
		return task_id*100+period_id;
	}
}
