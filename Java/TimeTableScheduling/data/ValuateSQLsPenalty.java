/*
 * �쐬��: 2007/07/18
 *
 * TODO ���̐������ꂽ�t�@�C���̃e���v���[�g��ύX����ɂ͎��փW�����v:
 * �E�B���h�E - �ݒ� - Java - �R�[�h�E�X�^�C�� - �R�[�h�E�e���v���[�g
 */
package data;

/**
 * @author �Y�����Y
 *
 * TODO ���̐������ꂽ�^�R�����g�̃e���v���[�g��ύX����ɂ͎��փW�����v:
 * �E�B���h�E - �ݒ� - Java - �R�[�h�E�X�^�C�� - �R�[�h�E�e���v���[�g
 */
public class ValuateSQLsPenalty {
	private ValuateSQLs v_sqls;
	private int penalty;
	
	public ValuateSQLsPenalty(){
		v_sqls = null;
		penalty = 0;
	}
	/**
	 * @return penalty ��߂��܂��B
	 */
	public int getPenalty() {
		return penalty;
	}
	/**
	 * @param penalty penalty ��ݒ�B
	 */
	public void setPenalty(int penalty) {
		this.penalty = penalty;
	}
	/**
	 * @return v_sqls ��߂��܂��B
	 */
	public ValuateSQLs getV_sqls() {
		return v_sqls;
	}
	/**
	 * @param v_sqls v_sqls ��ݒ�B
	 */
	public void setV_sqls(ValuateSQLs v_sqls) {
		this.v_sqls = v_sqls;
	}
	public void addPenalty(int penalty) {
		this.penalty += penalty;
	}
}
