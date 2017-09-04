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
public class LectureDetail {
	//�u�`��id
	private int task_id;
	//�u�`�̎��
	private int qualification_id;
	//�u�t�̐l���̉���
	private int required_processors_lb;
	//�u�t�̐l���̏��
	private int required_processors_ub;
	//�J�u�\����
	private ArrayList periods;
	
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
	/**
	 * @return qualification_id ��߂��܂��B
	 */
	public int getQualification_id() {
		return qualification_id;
	}
	/**
	 * @param qualification_id qualification_id ��ݒ�B
	 */
	public void setQualification_id(int qualification_id) {
		this.qualification_id = qualification_id;
	}
	/**
	 * @return required_processors_lb ��߂��܂��B
	 */
	public int getRequired_processors_lb() {
		return required_processors_lb;
	}
	/**
	 * @param required_processors_lb required_processors_lb ��ݒ�B
	 */
	public void setRequired_processors_lb(int required_processors_lb) {
		this.required_processors_lb = required_processors_lb;
	}
	/**
	 * @return required_processors_ub ��߂��܂��B
	 */
	public int getRequired_processors_ub() {
		return required_processors_ub;
	}
	/**
	 * @param required_processors_ub required_processors_ub ��ݒ�B
	 */
	public void setRequired_processors_ub(int required_processors_ub) {
		this.required_processors_ub = required_processors_ub;
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
		if(!(o instanceof LectureDetail))
			return false;
		LectureDetail l = (LectureDetail)o;
		return l.task_id==task_id;
	}
	public int hashCode(){
		return task_id;
	}
	
	//period_id�ɊJ�u�\���ǂ��������ׂ�(�J�u�ł���:true �ł��Ȃ�:false)
	public boolean isPeriod(int period_id){
		Iterator i = periods.iterator();
		while(i.hasNext()){
			PeriodDesire pd = (PeriodDesire)i.next();
			//period_id�ɍu�`���J�u�\
			if(pd.getPeriod_id()==period_id)return true;
		}
		return false;
	}
}
