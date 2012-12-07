/*
 * �쐬��: 2007/09/13
 *
 * TODO ���̐������ꂽ�t�@�C���̃e���v���[�g��ύX����ɂ͎��փW�����v:
 * �E�B���h�E - �ݒ� - Java - �R�[�h�E�X�^�C�� - �R�[�h�E�e���v���[�g
 */
package data;

import java.util.ArrayList;
import java.util.Iterator;

/**
 * @author masayoshi
 *
 * TODO DayPeriod�̏������Ǘ�
 *
 */

public class OrderDayPeriods {
	//�e�[�u��days��number,�e�[�u��hours��number���ɕ���DayPeriod�̃��X�g
	private ArrayList day_period_list;
	
	public OrderDayPeriods(ArrayList day_period_list){
		this.day_period_list = day_period_list;
	}
	
	//period_id�ɑ΂��ē���day_id�Ŏ��ɂ���next_period_id��Ԃ�
	//next_period_id���Ȃ���-1��Ԃ�
	public int getNextPeriods(int period_id){
		Iterator i = day_period_list.iterator();
		while(i.hasNext()){
			DayPeriod dp = (DayPeriod)i.next();
			if(dp.getPeriod_id()==period_id){
				if(i.hasNext()){
					DayPeriod ndp = (DayPeriod)i.next();
					if(ndp.getDay_id()==dp.getDay_id())return ndp.getPeriod_id();
					else return -1;
				}
			}
			
		}
		return -1;
	}
	public void printOrderDayPeriods(){
		System.out.println("(day_id,period_id)");
		Iterator i = day_period_list.iterator();
		while(i.hasNext()){
			DayPeriod dp = (DayPeriod)i.next();
			System.out.println(dp.toString());
		}
	}
	/**
	 * @return day_period_list ��߂��܂��B
	 */
	public ArrayList getDay_period_list() {
		return day_period_list;
	}
	/**
	 * @param day_period_list day_period_list ��ݒ�B
	 */
	public void setDay_period_list(ArrayList day_period_list) {
		this.day_period_list = day_period_list;
	}
}
