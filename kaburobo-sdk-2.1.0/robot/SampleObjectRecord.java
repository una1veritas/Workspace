

import java.io.Serializable;

import jp.tradesc.superkaburobo.sdk.trade.data.Stock;

/**
 * �����P�ʂ̃��������i�[����N���X�ł��B<br>
 * �ȉ��̏�����舵���܂��B<br>
 * <ul>
 * <li>����</li>
 * <li>�����E����敪</li>
 * <li>�������R</li>
 * </ul>
 * @author (c) 2004-2008 kaburobo.jp and Trade Science Corp. All rights reserved.
 */
public class SampleObjectRecord implements Serializable{
	private static final long serialVersionUID = 6825887897748591904L;
	/** �Ώۖ����ł�, ��{�I�� null �͂���܂��񂪁A���������̏ꍇ�� null �ɂȂ�܂��B */
	private Stock   stock = null;
	/** �����V�O�i��������V�O�i�����������܂��B<br>true: ����<br>false: ���s */
	private boolean isBuy = true;
	/** �V�O�i���̋������Βl�ŕ\���܂��B */
	private double  signalStrengh = 0;
	/** �V�O�i���������R�ł��B */
	private String  reason = "";
	
	/**
	 * �R���X�g���N�^�ł��B
	 * @param stock  �Ώۖ���
	 * @param isBuy  ���������肩�Atrue�Ŕ����Afalse�Ŕ���
	 * @param signalStrength �V�O�i���̋���(��Βl)
	 * @param reason �V�O�i���������R
	 */
	public SampleObjectRecord(Stock stock, boolean isBuy, double signalStrength, String reason){
		this.stock         = stock;
		this.isBuy         = isBuy;
		this.signalStrengh = signalStrength;
		this.reason        = reason;
	}
	
	/**
	 *  �Ώۖ������擾���܂��B
	 * @return �Ώۖ���
	 */
	public Stock   getStock(){
		return stock;
	}
	
	/** ���������肩���擾���܂��B
	 * @return ���������肩<br>
	 * true: �����Afalse: ����
	 */
	public boolean isBuy(){
		return isBuy;
	}
	
	/**
	 * �V�O�i���̋������擾���܂��B
	 * @return �V�O�i���̋����B<br>
	 * �T���v���R�[�h�ł͔���Ɣ����̋�ʂ͂���܂���B<br>
	 */
	public double getStrength(){
		return signalStrengh;
	}
	
	/**
	 * �V�O�i���������R���擾���܂��B
	 * @return �V�O�i���������R
	 */
	public String getReason(){
		return reason;
	}
}
