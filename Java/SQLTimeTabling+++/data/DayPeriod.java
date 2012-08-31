package data;

/**
 * @author masayoshi
 *
 * TODO 日付と作業時間の組(作業時間を日付ごとに分ける時使用)
 *
 */

public class DayPeriod {
	private int day_id;
	private int period_id;
	
	public DayPeriod(){
	}
	public DayPeriod(int day_id , int period_id){
		this.day_id = day_id;
		this.period_id = period_id;
	}
	public int getDay_id() {
		return day_id;
	}
	public void setDay_id(int day_id) {
		this.day_id = day_id;
	}
	public int getPeriod_id() {
		return period_id;
	}
	public void setPeriod_id(int period_id) {
		this.period_id = period_id;
	}
	public boolean equals(Object o){
		if(!(o instanceof DayPeriod))
			return false;
		DayPeriod dp = (DayPeriod)o;
		return (dp.day_id==day_id)&&(dp.period_id==period_id);
	}
	public String toString(){
		return "("+day_id+","+period_id+")";
	}
}
