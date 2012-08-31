/*
 * 作成日: 2006/11/27
 *
 * TODO この生成されたファイルのテンプレートを変更するには次へジャンプ:
 * ウィンドウ - 設定 - Java - コード・スタイル - コード・テンプレート
 */
package data.notdb;

import java.util.ArrayList;
import java.util.Iterator;

import data.PeriodDesire;

/**
 * @author 浦島太郎
 *
 * TODO この生成された型コメントのテンプレートを変更するには次へジャンプ:
 * ウィンドウ - 設定 - Java - コード・スタイル - コード・テンプレート
 */
public class TeacherDetail {
	//講師のid
	private int processor_id;
	//担当可能時間
	private ArrayList periods;
	//0:Japanese based, 1:English based, 2:don't care
	private ArrayList qualifications;
	//0:常勤講師 1:非常勤講師
	private int employment;
	//一週間の担当下限
	private int total_periods_lb;
	//一週間の担当上限
	private int total_periods_ub;
	//担当日数の下限
	//private int total_days_lb;
	//担当日数の上限
	private int total_days_ub;
	//賃金レベル
	//private int wage_level;
	public TeacherDetail(int processor_id){
		this.processor_id=processor_id;
	}
	public TeacherDetail(){
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
	 * @return periods を戻します。
	 */
	public ArrayList getPeriods() {
		return periods;
	}
	/**
	 * @param periods periods を設定。
	 */
	public void setPeriods(ArrayList periods) {
		this.periods = periods;
	}
	
	public boolean equals(Object o){
		if(!(o instanceof TeacherDetail))
			return false;
		TeacherDetail t = (TeacherDetail)o;
		return t.processor_id==processor_id;
	}
	public int hashCode(){
		return processor_id;
	}
	
	/**
	 * @return qualifications を戻します。
	 */
	public ArrayList getQualifications() {
		return qualifications;
	}
	/**
	 * @param qualifications qualifications を設定。
	 */
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
	
	public int getPeriodDesire(int period_id){
		Iterator i = periods.iterator();
		while(i.hasNext()){
			PeriodDesire pd = (PeriodDesire)i.next();
			//period_idに講師tが担当可能
			if(pd.getPeriod_id()==period_id)return pd.getPreferred_level();
		}
		return 0;
	}
	/**
	 * @return employment を戻します。
	 */
	public int getEmployment() {
		return employment;
	}
	/**
	 * @param employment employment を設定。
	 */
	public void setEmployment(int employment) {
		this.employment = employment;
	}
	/**
	 * @return total_periods_lb を戻します。
	 */
	public int getTotal_periods_lb() {
		return total_periods_lb;
	}
	/**
	 * @param total_periods_lb total_periods_lb を設定。
	 */
	public void setTotal_periods_lb(int total_periods_lb) {
		this.total_periods_lb = total_periods_lb;
	}
	/**
	 * @return total_periods_ub を戻します。
	 */
	public int getTotal_periods_ub() {
		return total_periods_ub;
	}
	/**
	 * @param total_periods_ub total_periods_ub を設定。
	 */
	public void setTotal_periods_ub(int total_periods_ub) {
		this.total_periods_ub = total_periods_ub;
	}
	public int getTotal_days_ub() {
		return total_days_ub;
	}
	public void setTotal_days_ub(int total_days_ub) {
		this.total_days_ub = total_days_ub;
	}
	
}
