/*
 * 作成日: 2007/09/13
 *
 * TODO この生成されたファイルのテンプレートを変更するには次へジャンプ:
 * ウィンドウ - 設定 - Java - コード・スタイル - コード・テンプレート
 */
package data;

import java.util.ArrayList;
import java.util.Iterator;

import java.io.*;
/**
 * @author masayoshi
 *
 * TODO DayPeriodの順序を管理
 *
 */

public class OrderDayPeriods implements Serializable {
	//テーブルdaysのnumber,テーブルhoursのnumber順に並んだDayPeriodのリスト
	private ArrayList day_period_list;
	
	public OrderDayPeriods(ArrayList day_period_list){
		this.day_period_list = day_period_list;
	}
	
	private OrderDayPeriods() {
		
	}
	
	//period_idに対して同じday_idで次にくるnext_period_idを返す
	//next_period_idがない時-1を返す
	public int getNextPeriod(int period_id){
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
	
	/*
	public void printOrderDayPeriods(){
		System.out.println("(day_id,period_id)");
		Iterator i = day_period_list.iterator();
		while(i.hasNext()){
			DayPeriod dp = (DayPeriod)i.next();
			System.out.println(dp.toString());
		}
	}
	*/
	/**
	 * @return day_period_list を戻します。
	 */
	 /*
	public ArrayList getDay_period_list() {
		return day_period_list;
	}
	*/
	/**
	 * @param day_period_list day_period_list を設定。
	 */
	 /*
	public void setDay_period_list(ArrayList day_period_list) {
		this.day_period_list = day_period_list;
	}
	*/
}
