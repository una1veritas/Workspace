/*
 * �쐬��: 2006/12/08
 *
 * TODO ���̐������ꂽ�t�@�C���̃e���v���[�g��ύX����ɂ͎��փW�����v:
 * �E�B���h�E - �ݒ� - Java - �R�[�h�E�X�^�C�� - �R�[�h�E�e���v���[�g
 */
package data;

import java.util.ArrayList;

/**
 * @author masayoshi
 *
 * TODO ���̐������ꂽ�^�R�����g�̃e���v���[�g��ύX����ɂ͎��փW�����v:
 * �E�B���h�E - �ݒ� - Java - �R�[�h�E�X�^�C�� - �R�[�h�E�e���v���[�g
 */
public class ValuateSQLs {
	private ArrayList v_sqls;
	private int weight;
	
	public ValuateSQLs(){
		v_sqls = new ArrayList();
	}
	
	public void addValuateSQL(ValuateSQL v_sql){
		v_sqls.add(v_sql);
	}
	
	/**
	 * @return v_sqls ��߂��܂��B
	 */
	public ArrayList getV_sqls() {
		return v_sqls;
	}
	/**
	 * @param v_sqls v_sqls ��ݒ�B
	 */
	public void setV_sqls(ArrayList v_sqls) {
		this.v_sqls = v_sqls;
	}
	/**
	 * @return weight ��߂��܂��B
	 */
	public int getWeight() {
		return weight;
	}
	/**
	 * @param weight weight ��ݒ�B
	 */
	public void setWeight(int weight) {
		this.weight = weight;
	}
}
