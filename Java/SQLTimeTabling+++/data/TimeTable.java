/*
 * 作成日: 2007/04/02
 *
 * TODO この生成されたファイルのテンプレートを変更するには次へジャンプ:
 * ウィンドウ - 設定 - Java - コード・スタイル - コード・テンプレート
 */
package data;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Set;
import java.util.TreeSet;

import common.DBConnectionPool;

/**
 * @author masayoshi
 *
 * TODO 講義時間割の情報を管理
 *
 */
public class TimeTable {
	//(task_id,processor_id,period_id)の組の集合
	private LinkedList timetablerows;
	//コネクションプール
	private DBConnectionPool cmanager;
	//全講師の情報(リストの中身Object:Teacher)
	private ArrayList teachers;
	//全講義の情報(リストの中身Object:Lecture)
	private ArrayList lectures;
	//全時間の情報(リストの中身Object:DayPeriod)
	private OrderDayPeriods day_periods;
	
	/**
	 * @return day_periods を戻します。
	 */
	public OrderDayPeriods getDay_periods() {
		return day_periods;
	}
	
	//timetablerowsのコピーを返す
	public ArrayList getTimeTableRowsCopy(){
		ArrayList result = new ArrayList();
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			result.add(new TimeTableRow(t.getTask_id(),t.getPeriod_id(),t.getProcessor_id()));
		}
		return result;
	}
	
	//timetablerowsのコピーを返す
	//連続する講義はtask_series_num個
	public ArrayList getTaskSeriesTimeTableRowsCopy(){
		ArrayList result = new ArrayList();
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			result.add(new TimeTableRow(t.getTask_id(),t.getPeriod_id(),t.getProcessor_id()));
			Lecture l = searchLecture(t.getTask_id());
			if(l.getTask_series_num()>1){
				int next_period_id = t.getPeriod_id();
				for(int num=1;num<l.getTask_series_num();num++){
					next_period_id = day_periods.getNextPeriods(next_period_id);
					result.add(new TimeTableRow(t.getTask_id(),next_period_id,t.getProcessor_id()));
				}
			}
		}
		return result;
	}
	
	//period_idにtask_idを担当しているprocessor_idのリストを返す
	public ArrayList getProcessors(int task_id, int period_id){
		ArrayList result = new ArrayList();
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			if((task_id==t.getTask_id())&&(period_id==t.getPeriod_id())){
				result.add(new Integer(t.getProcessor_id()));
			}
		}
		return result;
	}
	
	//現在の時間割の講義とその担当人数と開講時間のリストを取得
	public ArrayList getTimeTableTasksNum(){
		ArrayList result = new ArrayList();
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			int index = result.indexOf(new TaskPeriodNum(t.getTask_id(),t.getPeriod_id()));
			if(index==-1){
				result.add(new TaskPeriodNum(t.getTask_id(),t.getPeriod_id()));
			}else{
				 ((TaskPeriodNum)result.get(index)).updateNum();
			}
		}
		return result;
	}
	//現在の時間割の講義task_idの担当人数を取得
	public int getTimeTableTasks(int task_id){
		int count = 0;
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			if(t.getTask_id()==task_id){
				count++;
			}
		}
		return count;
	}
	//現在の時間割を講義とその開講時間のリストとして返す
	public ArrayList getTaskPeriods(){
		ArrayList result = new ArrayList();
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			if(!result.contains(new TaskPeriod(t.getTask_id(),t.getPeriod_id()))){
				result.add(new TaskPeriod(t.getTask_id(),t.getPeriod_id()));
			}
		}
		return result;
	}