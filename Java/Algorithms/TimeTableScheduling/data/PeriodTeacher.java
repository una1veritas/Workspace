/*
 * �쐬��: 2006/12/03
 *
 * TODO ���̐������ꂽ�t�@�C���̃e���v���[�g��ύX����ɂ͎��փW�����v:
 * �E�B���h�E - �ݒ� - Java - �R�[�h�E�X�^�C�� - �R�[�h�E�e���v���[�g
 */
package data;

import java.util.ArrayList;

/**
 * @author �Y�����Y
 *
 * TODO ���̐������ꂽ�^�R�����g�̃e���v���[�g��ύX����ɂ͎��փW�����v:
 * �E�B���h�E - �ݒ� - Java - �R�[�h�E�X�^�C�� - �R�[�h�E�e���v���[�g
 */
public class PeriodTeacher {
	private int period_id;
	private ArrayList teacher_ids;
	
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
	 * @return teacher_ids ��߂��܂��B
	 */
	public ArrayList getTeacher_ids() {
		return teacher_ids;
	}
	/**
	 * @param teacher_ids teacher_ids ��ݒ�B
	 */
	public void setTeacher_ids(ArrayList teacher_ids) {
		this.teacher_ids = teacher_ids;
	}
}
