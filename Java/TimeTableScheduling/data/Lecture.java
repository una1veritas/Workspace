package data;

import java.util.ArrayList;
import java.util.Iterator;


/**
 * @author masayoshi
 *
 * TODO �u�`�̏��
 *
 */

public class Lecture {
	//�u�`��id()
	private int task_id;
	//�u�`�̎��
	private int qualification_id;
	//�u�t�̐l���̉���
	private int required_processors_lb;
	//�u�t�̐l���̏��
	private int required_processors_ub;
	//���Ԃ̘A��(�u�`���Ԃ��A�����鎞�Ԃ̍쐬���g�p)
	private int task_series_num;
	//�J�u�\�Ȏ��Ԃ̃��X�g(���X�g�̒��gObject:PeriodDesire)
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
	 * @return task_series_num ��߂��܂��B
	 */
	public int getTask_series_num() {
		return task_series_num;
	}
	/**
	 * @param task_series_num task_series_num ��ݒ�B
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
	
	//period_id�ɊJ�u�\���ǂ��������ׂ�
	//(�J�u�ł���:true �ł��Ȃ�:false)
	public boolean isPeriod(int period_id){
		Iterator i = periods.iterator();
		while(i.hasNext()){
			PeriodDesire pd = (PeriodDesire)i.next();
			//period_id�ɍu�`���J�u�\
			if(pd.getPeriod_id()==period_id)return true;
		}
		return false;
	}
	//�m�F�p�i�ύX�\��j
	public String toString(){
		String result = "task_id:"+task_id+"task_series_num:"+task_series_num;
		return result;
	}
}
