/*
 * �쐬��: 2006/11/27
 *
 * TODO ���̐������ꂽ�t�@�C���̃e���v���[�g��ύX����ɂ͎��փW�����v:
 * �E�B���h�E - �ݒ� - Java - �R�[�h�E�X�^�C�� - �R�[�h�E�e���v���[�g
 */
package data.notdb;

import java.util.ArrayList;
import java.util.Iterator;

import data.PeriodDesire;

/**
 * @author �Y�����Y
 *
 * TODO ���̐������ꂽ�^�R�����g�̃e���v���[�g��ύX����ɂ͎��փW�����v:
 * �E�B���h�E - �ݒ� - Java - �R�[�h�E�X�^�C�� - �R�[�h�E�e���v���[�g
 */
public class TeacherDetail {
	//�u�t��id
	private int processor_id;
	//�S���\����
	private ArrayList periods;
	//0:Japanese based, 1:English based, 2:don't care
	private ArrayList qualifications;
	//0:��΍u�t 1:���΍u�t
	private int employment;
	//��T�Ԃ̒S������
	private int total_periods_lb;
	//��T�Ԃ̒S�����
	private int total_periods_ub;
	//�S�������̉���
	//private int total_days_lb;
	//�S�������̏��
	private int total_days_ub;
	//�������x��
	//private int wage_level;
	public TeacherDetail(int processor_id){
		this.processor_id=processor_id;
	}
	public TeacherDetail(){
	}
	/**
	 * @return processor_id ��߂��܂��B
	 */
	public int getProcessor_id() {
		return processor_id;
	}
	/**
	 * @param processor_id processor_id ��ݒ�B
	 */
	public void setProcessor_id(int processor_id) {
		this.processor_id = processor_id;
	}
	/**
	 * @return periods ��߂��܂��B
	 */
	public ArrayList getPeriods() {
		return periods;
	}
	/**
	 * @param periods periods ��ݒ�B
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
	 * @return qualifications ��߂��܂��B
	 */
	public ArrayList getQualifications() {
		return qualifications;
	}
	/**
	 * @param qualifications qualifications ��ݒ�B
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
	//qualification_id��S���ł��邩���ׂ�(�S���ł���:true �ł��Ȃ�:false)
	public boolean isQualification(int qualification_id){
		Iterator i = qualifications.iterator();
		while(i.hasNext()){
			Integer a = (Integer)i.next();
			//�u�t���u�`��S���ł��邩
			if(a.intValue()==qualification_id) return true;
		}
		return false;
	}
	//period_id�ɒS���ł��邩���ׂ�(�S���ł���:true �ł��Ȃ�:false)
	public boolean isPeriod(int period_id){
		Iterator i = periods.iterator();
		while(i.hasNext()){
			PeriodDesire pd = (PeriodDesire)i.next();
			//period_id�ɍu�tt���S���\
			if(pd.getPeriod_id()==period_id)return true;
		}
		return false;
	}
	
	public int getPeriodDesire(int period_id){
		Iterator i = periods.iterator();
		while(i.hasNext()){
			PeriodDesire pd = (PeriodDesire)i.next();
			//period_id�ɍu�tt���S���\
			if(pd.getPeriod_id()==period_id)return pd.getPreferred_level();
		}
		return 0;
	}
	/**
	 * @return employment ��߂��܂��B
	 */
	public int getEmployment() {
		return employment;
	}
	/**
	 * @param employment employment ��ݒ�B
	 */
	public void setEmployment(int employment) {
		this.employment = employment;
	}
	/**
	 * @return total_periods_lb ��߂��܂��B
	 */
	public int getTotal_periods_lb() {
		return total_periods_lb;
	}
	/**
	 * @param total_periods_lb total_periods_lb ��ݒ�B
	 */
	public void setTotal_periods_lb(int total_periods_lb) {
		this.total_periods_lb = total_periods_lb;
	}
	/**
	 * @return total_periods_ub ��߂��܂��B
	 */
	public int getTotal_periods_ub() {
		return total_periods_ub;
	}
	/**
	 * @param total_periods_ub total_periods_ub ��ݒ�B
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
