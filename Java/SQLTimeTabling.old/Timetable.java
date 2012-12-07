import java.io.*;
import java.sql.*;
import java.util.*;

import data.*;

/*
 * 作成日: 2006/12/11
 *
 * TODO この生成されたファイルのテンプレートを変更するには次へジャンプ:
 * ウィンドウ - 設定 - Java - コード・スタイル - コード・テンプレート
 */

/**
 * @author masayoshi
 *
 * TODO この生成された型コメントのテンプレートを変更するには次へジャンプ:
 * ウィンドウ - 設定 - Java - コード・スタイル - コード・テンプレート
 */
public class Timetable {
	//全講師
	private ArrayList teachers;
	//全講義
	private ArrayList lectures;
	private ArrayList v_sqls;
	private ArrayList t_v_sqls;
	private DBConnectionPool cmanager;
	
	public Timetable(String host,String db,String user, String pass, String file) throws Exception {
		cmanager = new DBConnectionPool("jdbc:mysql://"+host+"/"+db+"?useUnicode=true&characterEncoding=sjis",user,pass);
		//init(file);
		teachers = new ArrayList();
		lectures = new ArrayList();
		getWeightsFromDB();
		v_sqls = new ArrayList();
		t_v_sqls = new ArrayList();
		getRequestsFromFile(file);
	}
	
	public int local_search() throws Exception {
		int i;
		for (i=1; i<=50; i++) {
			System.out.println(i+"th iteration");
			if (//!search_move()&&
				!search_change() && 
				!search_swap() && 
				!search_add() &&
				!search_leave() ) {
				System.out.println("No more. ");
				break;
			}
		}
		return i;
	}
	
	//初期化
	private void init(String file) throws Exception {
		//teachers = new ArrayList();
		//lectures = new ArrayList();
		Connection con = null;
		ResultSet rs = null;
		ResultSet rs2 = null;
		Statement smt = null;
		Statement smt2 = null;
		String sql;
		
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
		//講師の初期化
		try {
			sql = "select * from processors a, processor_properties b where a.processor_id = b.processor_id order by a.processor_id";
 	        rs = smt.executeQuery(sql);
 	        while(rs.next()){
 	 	        Teacher t = new Teacher();
 	 	        ArrayList pds = new ArrayList();
                t.setProcessor_id(rs.getInt("processor_id"));
                //processor_idが担当な時間を取得
                String sql2 = "select period_id , preferred_level_proc from processor_schedules where processor_id = '" + t.getProcessor_id() + "' order by period_id";
                rs2 = smt2.executeQuery(sql2);
                while(rs2.next()){
                	PeriodDesire pd = new PeriodDesire();
                	pd.setPeriod_id(rs2.getInt("period_id"));
                	pds.add(pd);
                }
                t.setPeriods(pds);
                teachers.add(t);
 	        }
 	        System.out.println("Finished the initialization of lecturers. ");
 	        //講義の初期化
 	        sql = "select * from tasks a, task_properties b where a.task_id = b.task_id order by a.task_id";
 	       rs = smt.executeQuery(sql);
	        while(rs.next()){
	 	        Lecture l = new Lecture();
 	 	        ArrayList pds = new ArrayList();
 	 	        l.setTask_id(rs.getInt("task_id"));
 	 	        l.setRequired_processors_lb(rs.getInt("required_processors_lb"));
 	 	        l.setRequired_processors_ub(rs.getInt("required_processors_ub"));
 	 	        l.setQualification_id(rs.getInt("qualification_id"));
                //task_idが開講可能な時間を取得
                String sql2 = "select period_id , preferred_level_task from task_opportunities where task_id = '" + l.getTask_id() + "' order by period_id";
                rs2 = smt2.executeQuery(sql2);
                while(rs2.next()){
                	PeriodDesire pd = new PeriodDesire();
                	pd.setPeriod_id(rs2.getInt("period_id"));
                	pds.add(pd);
                }
                l.setPeriods(pds);
                lectures.add(l);
 	        