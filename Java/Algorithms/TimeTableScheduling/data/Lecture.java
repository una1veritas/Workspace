package data;

import java.util.ArrayList;
import java.util.Iterator;


/**
 * @author masayoshi
 *
 * TODO 講義の情報
 *
 */

public class Lecture {
	//講義のid()
	private int task_id;
	//講義の種類
	private int qualification_id;
	//講師の人数の下限
	private int required_processors_lb;
	//講師の人数の上限
	private int required_processors_ub;
	//時間の連続(講義時間が連続する時間の作成時使用)
	private int task_series_num;
	//開講可能な時間のリスト(リストの中身Object:PeriodDesire)
	private ArrayList periods;
	
	public ArrayList getPeriods() {
		return periods;
	}
	public void setPeriods(ArrayList periods) {
		this.periods = periods;
	}
	public int getQualification_id() {
		return qualification_id;
	}
	public void setQualification_id(int qualification_id) {
		this.qualification_id = qualification_id;
	}
	public int getRequired_processors_lb() {
		return required_processors_lb;
	}
	public void setRequired_processors_lb(int required_processors_lb) {
		this.required_processors_lb = required_processors_lb;
	}
	public int getRequired_processors_ub() {
		return required_processors_ub;
	}
	public void setRequired_processors_ub(int required_processors_ub) {
		this.required_processors_ub = required_processors_ub;
	}
	public int getTask_id() {
		return task_id;
	}
	public void setTask_id(int task_id) {
		this.task_id = task_id;
	}
	
	/**
	 * @return task_series_num を戻します。
	 */
	public int getTask_series_num() {
		return task_series_num;
	}
	/**
	 * @param task_series_num task_series_num を設定。
	 */
	public void setTask_series_num(int task_series_num) {
		this.task_series_num = task_series_num;
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
	
	//period_idに開講可能かどうかか調べる
	//(開講できる:true できない:false)
	public boolean isPeriod(int period_id){
		Iterator i = periods.iterator();
		while(i.hasNext()){
			PeriodDesire pd = (PeriodDesire)i.next();
			//period_idに講義が開講可能
			if(pd.getPeriod_id()==period_id)return true;
		}
		return false;
	}
	//確認用（変更予定）
	public String toString(){
		String result = "task_id:"+task_id+"task_series_num:"+task_series_num;
		return result;
	}
}
