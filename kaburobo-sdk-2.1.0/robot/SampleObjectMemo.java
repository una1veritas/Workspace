

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

import jp.tradesc.superkaburobo.sdk.trade.data.Stock;

/**
 * �I�u�W�F�N�g�����N���X�ł��B<br>
 * �X�N���[�j���O�̌��ʂ��i�[���܂��B
 * @author (c) 2004-2008 kaburobo.jp and Trade Science Corp. All rights reserved.
 */
public class SampleObjectMemo implements Serializable{
	private static final long serialVersionUID = 5651047511229005177L;
	
	/**
	 * �����Ώۃ��X�g
	 */
	private ArrayList<SampleObjectRecord> memoList  = new ArrayList<SampleObjectRecord>();
	
	/**
	 * �����Ώۃ��X�g��Ԃ��܂�
	 * @return �����Ώۂ� ArrayList �ł��Bnull ���Ԃ���邱�Ƃ͂���܂���B
	 */
	public ArrayList<SampleObjectRecord> getMemoList() {
		return memoList;
	}
	
	/**
	 * �w���������X�g��ݒ肵�܂��B<br>
	 * null ��ݒ肳�ꂽ�ꍇ�Anull �ł͂Ȃ���� ArrayList ���ݒ肳��܂��B
	 * @param memoList �ݒ肷�郁�����X�g
	 */
	public void setMemoList(ArrayList<SampleObjectRecord> memoList) {
		if (null == memoList){
			this.memoList.clear();
		} else {
			this.memoList = memoList;
		}
	}
	
	/**
	 * �����\�胊�X�g���V�O�i���̋����ɂ��\�[�g���܂��B
	 */
	public void sortStockMemoByStrength(){
		Collections.sort(memoList, new stockMemoSorter());
	}

	// �R���p���[�^�N���X�ł��B�V�O�i���̋������r���܂��B
	private static class stockMemoSorter implements Comparator<SampleObjectRecord>, Serializable{
		private static final long serialVersionUID = 5532134792677963169L;
		public int compare(SampleObjectRecord o1, SampleObjectRecord o2) {
			return (int) Math.round(o2.getStrength() - o1.getStrength());
		}
		
	}
	
	/** 
	 * �Ώۖ����������\�胊�X�g�Ɋ܂܂�Ă��Ȃ������m�F���܂��B
	 * @param isBuy ���������𒲂ׂ����ꍇ true�A���蒍���𒲂ׂ����ꍇ�� false
	 * @param stock ���̖�����
	 * @return ������� true�A��������Ȃ���� false ���Ԃ�܂��B
	 */
	public boolean contain(boolean isBuy, Stock stock){
		int stock_code = stock.getStockCode().intValue();
		for (SampleObjectRecord memo: memoList) {
			if(
				(memo.isBuy() == isBuy) &&
				(memo.getStock().getStockCode().intValue() == stock_code)
			) {
				return true;
			}
		}
		return false;
	}
}
