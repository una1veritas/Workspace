/*
 * �쐬��: 2006/12/08
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
public class ValuateSQL {
	private int type;
	private String sql;
	
	/**
	 * @return sql ��߂��܂��B
	 */
	public String getSql() {
		return sql;
	}
	/**
	 * @param sql sql ��ݒ�B
	 */
	public void setSql(String sql) {
		this.sql = sql;
	}
	/**
	 * @return type ��߂��܂��B
	 */
	public int getType() {
		return type;
	}
	/**
	 * @param type type ��ݒ�B
	 */
	public void setType(int type) {
		this.type = type;
	}
	
}
