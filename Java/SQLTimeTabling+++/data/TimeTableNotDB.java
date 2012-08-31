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
 * TODO この生成された型コメントのテンプレートを変更するには次へジャンプ:
 * ウィンドウ - 設定 - Java - コード・スタイル - コード・テンプレート
 */
public class TimeTableNotDB {
	//(task_id,processor_id,period_id)の組の集合
	private LinkedList timetablerows;
	private DBConnectionPool cmanager;
	//全講師
	private ArrayList teachers;
	//全講義
	private ArrayList lectures;
	
	public ArrayList getTimeTableRowsCopy(){
		ArrayList result = new ArrayList();
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			result.add(new TimeTableRow(t.getTask_id(),t.getPeriod_id(),t.getProcessor_id()));
		}
		return result;
	}
	
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
	
	public TimeTableNotDB(DBConnectionPool cmanager, ArrayList teachers, ArrayList lectures){
		this.cmanager = cmanager;
		this.teachers = teachers;
		this.lectures = lectures;
		this.timetablerows = new LinkedList();
	}
	
	public TimeTableNotDB(String host,String db,String user, String pass) throws Exception{
		cmanager = new DBConnectionPool("jdbc:mysql://"+host+"/"+db+"?useUnicode=true&characterEncoding=sjis",user,pass);
		init();
		this.timetablerows = new LinkedList();
	}
	
	//初期化
	private void init() throws Exception{
		teachers = new ArrayList();
		lectures = new ArrayList();
		Connection con = null;
		ResultSet rs = null;
		ResultSet rs2 = null;
		Statement smt = null;
		Statement smt2 = null;
		try {
			//MySQLサーバ接続
			con = cmanager.getConnection();
		} catch (SQLException e) {
			System.err.println("Couldn't get connection: " + e);
			throw e;
		}
		//Statementインターフェースの生成
		smt = con.createStatement();
		smt2 = con.createStatement();
		try {
			//このプログラム上で一時的に作るテーブルがDB上にすでに存在している場合そのテーブルを削除する
			String sql = "drop table if exists t";
			smt.executeUpdate(sql);
			sql = "drop table if exists t_timetable";
			smt.executeUpdate(sql);
			sql = "drop table if exists t_task_relation";
			smt.executeUpdate(sql);
			sql = "drop table if exists t_processor_relation";
			smt.executeUpdate(sql);
			//講師の初期化
			//属性が入力�