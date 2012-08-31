package data;

import java.util.ArrayList;
import java.util.Iterator;

public class Teacher {
	//講師のid
	private int processor_id;
	//担当可能な時間のリスト(リストの中身Object:PeriodDesire)
	private ArrayList periods;
	//担当可能な種類のリスト(リストの中身Object:Integer)
	private ArrayList qualifications;
	
	public ArrayList getPeriods() {
		return periods;
	}
	public void setPeriods(ArrayList periods) {
		this.periods = periods;
	}
	public int getProcessor_id() {
		return processor_id;
	}
	public void setProcessor_id(int processor_id) {
		this.processor_id = processor_id;
	}
	public ArrayList getQualifications() {
		return qualifications;
	}
	public void setQualifications(ArrayList qualifications) {
		this.qualifications = qualifications;
	}
	public String toString(){
		String a = "processor_id:"+processor_id+"\n";
		Iterator i = qualifications.iterator();
		while(i.hasNext()){
			int t = ((Integer)i.next()).intValue();
			a += "qualification_id:"+t+"\n";
		}
		return a;
	}
	//qualification_idを担当できるか調べる(担当できる:true できない:false)
	public boolean isQualification(int qualification_id){
		Iterator i = qualifications.iterator();
		while(i.hasNext()){
			Integer a = (Integer)i.next();
			//講師が講義を担当できるか
			if(a.intValue()==qualification_id) return true;
		}
		return false;
	}
	//period_idに担当できるか調べる(担当できる:true できない:false)
	public boolean isPeriod(int period_id){
		Iterator i = periods.iterator();
		while(i.hasNext()){
			PeriodDesire pd = (PeriodDesire)i.next();
			//period_idに講師tが担当可能
			if(pd.getPeriod_id()==period_id)return true;
		}
		return false;
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
