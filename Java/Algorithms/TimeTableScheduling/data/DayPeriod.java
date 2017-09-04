package data;

/**
 * @author masayoshi
 *
 * TODO ���t�ƍ�Ǝ��Ԃ̑g(��Ǝ��Ԃ���t���Ƃɕ����鎞�g�p)
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
