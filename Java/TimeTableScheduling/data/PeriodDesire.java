/*
 * �쐬��: 2006/11/27
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
public class PeriodDesire {
	//���Ԃ�id
    private int period_id;
    //���Ԃ̊�]�x�i�v�]�̕]����DB���g�p���Ȃ����Ɏg�p�j
    private int preferred_level;
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
	
	public int getPreferred_level() {
		return preferred_level;
	}
	public void setPreferred_level(int preferred_level) {
		this.preferred_level = preferred_level;
	}
	public boolean equals(Object o){
		if(!(o instanceof PeriodDesire))
			return false;
		PeriodDesire pd = (PeriodDesire)o;
		return pd.period_id==period_id;
	}
	public int hashCode(){
		return period_id;
	}
}
