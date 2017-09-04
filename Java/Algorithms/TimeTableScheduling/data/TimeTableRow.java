/*
 * �쐬��: 2007/04/02
 *
 * TODO ���̐������ꂽ�t�@�C���̃e���v���[�g��ύX����ɂ͎��փW�����v:
 * �E�B���h�E - �ݒ� - Java - �R�[�h�E�X�^�C�� - �R�[�h�E�e���v���[�g
 */
package data;

/**
 * @author masayoshi
 *
 * TODO ���̐������ꂽ�^�R�����g�̃e���v���[�g��ύX����ɂ͎��փW�����v:
 * �E�B���h�E - �ݒ� - Java - �R�[�h�E�X�^�C�� - �R�[�h�E�e���v���[�g
 */
public class TimeTableRow {
	//�u�`��id
	private int task_id;
	//���Ԃ�id
	private int period_id;
	//�u�t��id
	private int processor_id;
	
	public TimeTableRow(int task_id, int period_id, int processor_id){
		this.task_id = task_id;
		this.period_id = period_id;
		this.processor_id = processor_id;
	}
	/**
	 * @return period_id ��߂��܂��B
	 */
	public int getPeriod_id() {
		return period_id;
	}
	/**
	 * @param period_id period_id ��ݒ�B
	 */
	public void setPeriod_id(int period_id) {
		this.period_id = period_id;
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
	
	public String toString(){
		return "("+task_id+","+period_id+","+processor_id+")";
	}
	
	public boolean equals(Object o){
		if(!(o instanceof TimeTableRow))
			return false;
		TimeTableRow ttr = (TimeTableRow)o;
		return (ttr.task_id==task_id)&&(ttr.period_id==period_id)&&(ttr.processor_id==processor_id);
	}
	
	public int hashCode(){
		return task_id*10000+period_id*100+processor_id;
	}
}
