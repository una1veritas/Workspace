package data;

import java.util.ArrayList;
import java.util.Iterator;

public class Teacher {
	//ut‚Ìid
	private int processor_id;
	//’S“–‰Â”\‚ÈŠÔ‚ÌƒŠƒXƒg(ƒŠƒXƒg‚Ì’†gObject:PeriodDesire)
	private ArrayList periods;
	//’S“–‰Â”\‚Èí—Ş‚ÌƒŠƒXƒg(ƒŠƒXƒg‚Ì’†gObject:Integer)
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
	//qualification_id‚ğ’S“–‚Å‚«‚é‚©’²‚×‚é(’S“–‚Å‚«‚é:true ‚Å‚«‚È‚¢:false)
	public boolean isQualification(int qualification_id){
		Iterator i = qualifications.iterator();
		while(i.hasNext()){
			Integer a = (Integer)i.next();
			//ut‚ªu‹`‚ğ’S“–‚Å‚«‚é‚©
			if(a.intValue()==qualification_id) return true;
		}
		return false;
	}
	//period_id‚É’S“–‚Å‚«‚é‚©’²‚×‚é(’S“–‚Å‚«‚é:true ‚Å‚«‚È‚¢:false)
	public boolean isPeriod(int period_id){
		Iterator i = periods.iterator();
		while(i.hasNext()){
			PeriodDesire pd = (PeriodDesire)i.next();
			//period_id‚Éutt‚ª’S“–‰Â”\
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
