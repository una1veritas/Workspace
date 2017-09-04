import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.Random;
import java.util.StringTokenizer;

import org.apache.commons.lang.time.StopWatch;

import common.DBConnectionPool;

import data.CombEnum;
import data.DayPeriod;
import data.Lecture;
import data.OrderDayPeriods;
import data.PeriodDesire;
import data.TaskPeriod;
import data.TaskPeriodNum;
import data.Teacher;
import data.TimeTable;
import data.TimeTableRow;
import data.ValuateSQL;
import data.ValuateSQLs;
import data.ValuateSQLsPenalty;

/*
 * 作成日: 2006/12/11
 *
 * TODO この生成されたファイルのテンプレートを変更するには次へジャンプ:
 * ウィンドウ - 設定 - Java - コード・スタイル - コード・テンプレート
 */

/**
 * @author masayoshi
 *
 * TODO 時間割作成
 *
 */
public class TimeTableSchedulingTaskSeries {
	//全講師(リストの中身Object:Teacher)
	private ArrayList teachers;
	//全講義(リストの中身Object:Lecture)
	private ArrayList lectures;
	//全時間の情報(リストの中身Object:DayPeriod)
	private OrderDayPeriods day_periods;
	//SQLによる要望評価文
	private ArrayList v_sqls;
	//データベースのコネクションプール
	private DBConnectionPool cmanager;
	//講義時間割り
	private TimeTable timetable;
	//評価の方法 1:ファイル 2:DB
	private int valuateType;
	
	//実験用ストップウォッチ
	private StopWatch vsw;
	private boolean SWflag = true;
	private boolean debug = false;
	
	//コンストラクタ(プロジェクトが存在しない)
	public TimeTableSchedulingTaskSeries(String host,String db,String user, String pass, String file, int readType) throws Exception{
		cmanager = new DBConnectionPool("jdbc:mysql://"+host+"/"+db+"?useUnicode=true&characterEncoding=sjis",user,pass);
		init(file,readType);
		timetable = new TimeTable(cmanager,teachers,lectures);
		if(debug)vsw = new StopWatch();
	}
	//コンストラクタ
	public TimeTableSchedulingTaskSeries(String host,String db,String user, String pass, String file, int readType,int project_id,int mode) throws Exception{
		cmanager = new DBConnectionPool("jdbc:mysql://"+host+"/"+db+"?useUnicode=true&characterEncoding=sjis",user,pass);
		if(mode==4){
			initAddProject(file,readType,project_id);
			timetable = new TimeTable(cmanager,teachers,lectures);
		}else if(mode==5){
			initTaskSeriesAddProject(file,readType,project_id);
			timetable = new TimeTable(cmanager,teachers,lectures,day_periods);
		}
		if(debug) vsw = new StopWatch();
	}
	
	//初期化
	private void init(String file,int readType) throws Exception{
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
			sql = "drop table if exists t_task_relation";
			smt.executeUpdate(sql);
			sql = "drop table if exists t_processor_relation";
			smt.executeUpdate(sql);
			//講師の初期化
			//属性が入力されている全講師を取得
			sql = "select a.processor_id from processors a, processor_properties b where a.processor_id = b.processor_id order by a.processor_id";
			rs = smt.executeQuery(sql);
			while(rs.next()){
				Teacher t = new Teacher();
				ArrayList pds = new ArrayList();
				ArrayList qs = new ArrayList();
				t.setProcessor_id(rs.getInt("processor_id"));
				//processor_idが担当可能な時間を取得
				String sql2 = "select period_id , preferred_level_proc from processor_schedules where processor_id = '" + t.getProcessor_id() + "' order by period_id";
				rs2 = smt2.executeQuery(sql2);
				while(rs2.next()){
					PeriodDesire pd = new PeriodDesire();
					pd.setPeriod_id(rs2.getInt("period_id"));
					pds.add(pd);
				}
				t.setPeriods(pds);
				//processor_idが担当可能な講義の種類を取得
				sql2 = "select qualification_id from processor_qualification where processor_id = '" + t.getProcessor_id() + "' order by qualification_id";
				rs2 = smt2.executeQuery(sql2);
				while(rs2.next()){
					qs.add(new Integer(rs2.getInt("qualification_id")));
				}
				t.setQualifications(qs);
				teachers.add(t);
			}
			if(debug) System.out.println("講師の初期化終了");
			//講義の初期化
			//属性が入力されている全講義の取得
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
			}
			if(debug) System.out.println("講義の初期化終了");
		}catch(SQLException e) {
			System.err.println(e);
			throw e;
		}finally{
			//ResultSetインターフェースの破棄
			if(rs!=null){
				rs.close();
				rs = null;
			}
			//ResultSetインターフェースの破棄
			if(rs2!=null){
				rs2.close();
				rs2 = null;
			}
			// Statementインターフェースの破棄
			if(smt!=null){
				smt.close();
				smt = null;
			}
			// Statementインターフェースの破棄
			if(smt2!=null){
				smt2.close();
				smt2 = null;
			}
			//MySQLサーバ切断
			if(con!=null){
				cmanager.freeConnection(con);
				con = null;
			}
		}if(readType==1){
			readerFileValuateNumSQL(file);valuateType = 1;
		}else if(readType==2){
			readerFileValuateSQL(file);valuateType = 1;
		}else{
			valuateType = 2;readerDBValuateSQL(file);
		}
	}
	
	//初期化(プロジェクト管理しているデータベース使用)
	private void initAddProject(String file,int readType,int project_id) throws Exception{
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
			sql = "drop table if exists t_task_relation";
			smt.executeUpdate(sql);
			sql = "drop table if exists t_processor_relation";
			smt.executeUpdate(sql);
			//講師の初期化
			//属性が入力されている全講師を取得
			sql = "select a.processor_id from processors a, processor_properties b where a.processor_id = b.processor_id and a.project_id ="+project_id+" order by a.processor_id";
			rs = smt.executeQuery(sql);
			while(rs.next()){
				Teacher t = new Teacher();
				ArrayList pds = new ArrayList();
				ArrayList qs = new ArrayList();
				t.setProcessor_id(rs.getInt("processor_id"));
				//processor_idが担当可能な時間を取得
				String sql2 = "select period_id , preferred_level_proc from processor_schedules where processor_id = '" + t.getProcessor_id() + "' order by period_id";
				rs2 = smt2.executeQuery(sql2);
				while(rs2.next()){
					PeriodDesire pd = new PeriodDesire();
					pd.setPeriod_id(rs2.getInt("period_id"));
					pds.add(pd);
				}
				t.setPeriods(pds);
				//processor_idが担当可能な講義の種類を取得
				sql2 = "select qualification_id from processor_qualification where processor_id = '" + t.getProcessor_id() + "' order by qualification_id";
				rs2 = smt2.executeQuery(sql2);
				while(rs2.next()){
					qs.add(new Integer(rs2.getInt("qualification_id")));
				}
				t.setQualifications(qs);
				teachers.add(t);
			}
			if(debug) System.out.println("講師の初期化終了");
			//講義の初期化
			//属性が入力されている全講義の取得
			sql = "select * from tasks a, task_properties b where a.task_id = b.task_id and a.project_id = "+project_id+" order by a.task_id";
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
			}
			if(debug) System.out.println("講義の初期化終了");
		}catch(SQLException e) {
			System.err.println(e);
			throw e;
		}finally{
			//ResultSetインターフェースの破棄
			if(rs!=null){
				rs.close();
				rs = null;
			}
			//ResultSetインターフェースの破棄
			if(rs2!=null){
				rs2.close();
				rs2 = null;
			}
			// Statementインターフェースの破棄
			if(smt!=null){
				smt.close();
				smt = null;
			}
			// Statementインターフェースの破棄
			if(smt2!=null){
				smt2.close();
				smt2 = null;
			}
			//MySQLサーバ切断
			if(con!=null){
				cmanager.freeConnection(con);
				con = null;
			}
		}
		if(readType==1){
			readerFileValuateNumSQL(file);valuateType = 1;
		}else if(readType==2){
			readerFileValuateSQL(file);valuateType = 1;
		}else{
			valuateType = 2;readerDBValuateSQL(file);
		}
	}
	//初期化(プロジェクト管理しているデータベース使用)
	//連続する講義に対応
	private void initTaskSeriesAddProject(String file,int readType,int project_id) throws Exception{
		teachers = new ArrayList();
		lectures = new ArrayList();
		ArrayList list = new ArrayList();
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
			sql = "drop table if exists t_task_relation";
			smt.executeUpdate(sql);
			sql = "drop table if exists t_processor_relation";
			smt.executeUpdate(sql);
			//講師の情報の初期化
			//属性が入力されている講師のprocessor_idを取得
			sql = "select a.processor_id from processors a, processor_properties b where a.processor_id = b.processor_id and a.project_id ="+project_id+" order by a.processor_id";
			rs = smt.executeQuery(sql);
			while(rs.next()){
				Teacher t = new Teacher();
				ArrayList pds = new ArrayList();
				ArrayList qs = new ArrayList();
				t.setProcessor_id(rs.getInt("processor_id"));
				//processor_idが担当可能な時間を取得
				String sql2 = "select period_id , preferred_level_proc from processor_schedules where processor_id = '" + t.getProcessor_id() + "' order by period_id";
				rs2 = smt2.executeQuery(sql2);
				while(rs2.next()){
					PeriodDesire pd = new PeriodDesire();
					pd.setPeriod_id(rs2.getInt("period_id"));
					pds.add(pd);
				}
				t.setPeriods(pds);
				//processor_idが担当可能な講義の種類を取得
				sql2 = "select qualification_id from processor_qualification where processor_id = '" + t.getProcessor_id() + "' order by qualification_id";
				rs2 = smt2.executeQuery(sql2);
				while(rs2.next()){
					qs.add(new Integer(rs2.getInt("qualification_id")));
				}
				t.setQualifications(qs);
				teachers.add(t);
			}
			if(debug) System.out.println("講師の初期化終了");
			//講義の情報の初期化
			//属性が入力されている講義の情報の取得
			sql = "select * from tasks a, task_properties b where a.task_id = b.task_id and a.project_id = "+project_id+" order by a.task_id";
			rs = smt.executeQuery(sql);
			while(rs.next()){
				Lecture l = new Lecture();
				ArrayList pds = new ArrayList();
				l.setTask_id(rs.getInt("task_id"));
				l.setRequired_processors_lb(rs.getInt("required_processors_lb"));
				l.setRequired_processors_ub(rs.getInt("required_processors_ub"));
				l.setQualification_id(rs.getInt("qualification_id"));
				//連続する講義
				l.setTask_series_num(rs.getInt("task_series_num"));
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
			}
			if(debug){
				System.out.println("講義の初期化終了");
				/*
				Iterator i = lectures.iterator();
				while(i.hasNext()){
					Lecture l = (Lecture)i.next();
					System.out.println(l.toString());
				}
				*/
				
			}
			//時間の初期化
			sql = "select pp.day_id , pp.period_id from period_properties pp, days d, hours h where pp.day_id = d.day_id and pp.hour_id = h.hour_id and pp.project_id = "+project_id+" order by d.number,d.day_id,h.number,h.hour_id";
			rs = smt.executeQuery(sql);
			while(rs.next()){
				DayPeriod dp = new DayPeriod(rs.getInt("day_id"),rs.getInt("period_id"));
				list.add(dp);
			}
			day_periods = new OrderDayPeriods(list);
			if(debug) System.out.println("時間の初期化終了");
		}catch(SQLException e) {
			System.err.println(e);
			throw e;
		}finally{
			//ResultSetインターフェースの破棄
			if(rs!=null){
				rs.close();
				rs = null;
			}
			//ResultSetインターフェースの破棄
			if(rs2!=null){
				rs2.close();
				rs2 = null;
			}
			// Statementインターフェースの破棄
			if(smt!=null){
				smt.close();
				smt = null;
			}
			// Statementインターフェースの破棄
			if(smt2!=null){
				smt2.close();
				smt2 = null;
			}
			//MySQLサーバ切断
			if(con!=null){
				cmanager.freeConnection(con);
				con = null;
			}
		}
		if(readType==1){
			readerFileValuateNumSQL(file);valuateType = 1;
		}else if(readType==2){
			readerFileValuateSQL(file);valuateType = 1;
		}else{
			valuateType = 2;readerDBValuateSQL(file);
		}
	}
	
	//ファイルから評価SQLを取得する(読み込むＳＱＬに番号)
	//お勧めしない
	private void readerFileValuateNumSQL(String file) throws Exception{
		//ファイルから評価SQLを取得する
		BufferedReader reader = null;
		v_sqls = new ArrayList();
		try {
			reader = new BufferedReader(new FileReader(file));
			String line;
			while ((line = reader.readLine()) != null) {
				ValuateSQL v_sql = null;
				StringTokenizer strToken = new StringTokenizer(line, ";");
				if(strToken.hasMoreTokens()) {
					int type = Integer.parseInt(strToken.nextToken().toString());
					v_sql = new ValuateSQL();
					v_sql.setType(type);
				}
				if(strToken.hasMoreTokens()) {
					String sql = strToken.nextToken().toString();
					v_sql.setSql(sql);
				}
				if(v_sql!=null)v_sqls.add(v_sql);
			}
			reader.close();
		} catch (FileNotFoundException e) {
			System.out.println("ファイルが見つかりません");
			throw e;
		}
	}
	
	//	ファイルから評価SQLを取得する
	private void readerFileValuateSQL(String file) throws Exception{
		BufferedReader reader = null;
		ArrayList create_list = new ArrayList();
		v_sqls = new ArrayList();
		try {
			reader = new BufferedReader(new FileReader(file));
			String line;
			while ((line = reader.readLine()) != null) {
				if(!line.equals("")){
					String[] str = line.split(" ");
					ValuateSQL v_sql = new ValuateSQL();
					if(str[0].equals("select")){
						v_sql.setType(2);
						v_sql.setSql(line);
						v_sqls.add(v_sql);
					}else if(str[0].equals("create")){
						v_sql.setType(1);
						v_sql.setSql(line);
						v_sqls.add(v_sql);
						for(int i =1;i<str.length;i++){
							if(str[i].equals("table")){
								create_list.add(str[i+1]);
								break;
							}
						}
					}else if(str[0].equals("delete")){
						for(int i =1;i<str.length;i++){
							if(str[i].equals("from")){
								if(create_list.contains(str[i+1])){
									v_sql.setType(1);
									v_sql.setSql(line);
									v_sqls.add(v_sql);
								}
								break;
							}
						}
					}else if(str[0].equals("drop")){
						for(int i =1;i<str.length;i++){
							if(str[i].equals("table")){
								if(create_list.contains(str[i+1])){
									v_sql.setType(1);
									v_sql.setSql(line);
									v_sqls.add(v_sql);
								}
								break;
							}
						}
					}else{
						v_sql.setType(1);
						v_sql.setSql(line);
						v_sqls.add(v_sql);
					}
				}
			}
		} catch (FileNotFoundException e) {
			System.out.println("ファイルが見つかりません");
			throw e;
		} catch (IOException e) {
		} finally {
			if (reader != null) {
				reader.close();
			}
		}
		/*
		Iterator j = v_sqls.iterator();
		while(j.hasNext()){
			ValuateSQL temp= (ValuateSQL)j.next();
			System.out.println(temp.getType()+":"+temp.getSql());
		}
		*/
	}

	//データベースから評価SQLを取得する
	private void readerDBValuateSQL(String table) throws Exception{
		v_sqls = new ArrayList();
		ArrayList create_list = new ArrayList();
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		try{
			//MySQLサーバ接続
			con = cmanager.getConnection();
			//Statementインターフェースの生成
			smt = con.createStatement();
			String sql = "select * from "+ table + " where use_flag = 1 order by sql_id";
			rs = smt.executeQuery(sql);
			while(rs.next()){
				ValuateSQLs temp = new ValuateSQLs();
				temp.setWeight(rs.getInt("weight"));
				String lines = rs.getString("valuate_sql");
				StringTokenizer strToken = new StringTokenizer(lines, ";");
				while(strToken.hasMoreTokens()) {
					String line = strToken.nextToken().toString();
					if(!line.equals("")){
						String[] str = line.split(" ");
						ValuateSQL v_sql = new ValuateSQL();
						if(str[0].equals("select")){
							v_sql.setType(2);
							v_sql.setSql(line);
							temp.addValuateSQL(v_sql);
						}else if(str[0].equals("create")){
							v_sql.setType(1);
							v_sql.setSql(line);
							temp.addValuateSQL(v_sql);
							for(int i =1;i<str.length;i++){
								if(str[i].equals("table")){
									create_list.add(str[i+1]);
									break;
								}
							}
						}else if(str[0].equals("delete")){
							for(int i =1;i<str.length;i++){
								if(str[i].equals("from")){
									if(create_list.contains(str[i+1])){
										v_sql.setType(1);
										v_sql.setSql(line);
										temp.addValuateSQL(v_sql);
									}
									break;
								}
							}
						}else if(str[0].equals("drop")){
							for(int i =1;i<str.length;i++){
								if(str[i].equals("table")){
									if(create_list.contains(str[i+1])){
										v_sql.setType(1);
										v_sql.setSql(line);
										temp.addValuateSQL(v_sql);
									}
									break;
								}
							}
						}else{
							v_sql.setType(1);
							v_sql.setSql(line);
							temp.addValuateSQL(v_sql);
						}
					}
				}
				if(temp.getV_sqls().size()!=0)v_sqls.add(temp);
			}
		}catch(SQLException e) {
			System.out.println(e);
			throw e;
		}catch(Exception e){
			e.printStackTrace();
		}finally{
			//ResultSetインターフェースの破棄
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statementインターフェースの破棄
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQLサーバ切断
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		if(debug){
			Iterator j = v_sqls.iterator();
			while(j.hasNext()){
				ValuateSQLs temp= (ValuateSQLs)j.next();
				System.out.println("weight:"+temp.getWeight());
				Iterator j2 = temp.getV_sqls().iterator();
				while(j2.hasNext()){
					ValuateSQL temp2= (ValuateSQL)j2.next();
					System.out.println(temp2.getType()+":"+temp2.getSql());
				}
			}
			System.out.println("create_list");
			Iterator iterator = create_list.iterator();
			while(iterator.hasNext()){
				System.out.println("table:"+((String)iterator.next()));
			}
		}
	}

	//解の評価値をファイルに残す
	public void fileTimeTableSQLDBLog(String file) throws Exception{
		BufferedWriter bw = null;
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		ArrayList v_sqls_ps = null;
		ValuateSQLsPenalty v_sqls_p = null;
		int penalties = 0;
		ValuateSQLs t = null;
		ValuateSQL t2 = null;
		if(debug){
			if(SWflag){
				SWflag = false;
				vsw.start();
			}
			else vsw.resume();
		}
		try{
			v_sqls_ps = new ArrayList();
			bw = new BufferedWriter(new FileWriter(file));
			bw.write("count_offence:"+timetable.countOffence());
			bw.newLine();
			//MySQLサーバ接続
			con = cmanager.getConnection();
			//Statementインターフェースの生成
			smt = con.createStatement();
			Iterator i = v_sqls.iterator();
			while(i.hasNext()){
				v_sqls_p = new ValuateSQLsPenalty();
				t = (ValuateSQLs)i.next();
				v_sqls_p.setV_sqls(t);
				Iterator i2 = t.getV_sqls().iterator();
				while(i2.hasNext()){
					t2 = (ValuateSQL)i2.next();
					switch(t2.getType()){
					case 1:
						smt.executeUpdate(t2.getSql());
						break;
					case 2:
						rs = smt.executeQuery(t2.getSql());
						if(rs.next()){
							int num = rs.getInt("penalties");
							v_sqls_p.addPenalty(num*t.getWeight());
							penalties = penalties + num*t.getWeight();
						}
						break;
					case 3:
						rs = smt.executeQuery(t2.getSql());
						break;
					default:
					}
				}
				v_sqls_ps.add(v_sqls_p);
			}
			bw.write("valuate:"+penalties);
			bw.newLine();
			bw.newLine();
			Iterator i3 = v_sqls_ps.iterator();
			while(i3.hasNext()){
				bw.write("評価SQL");
				bw.newLine();
				v_sqls_p = (ValuateSQLsPenalty)i3.next();
				Iterator i4 = v_sqls_p.getV_sqls().getV_sqls().iterator();
				while(i4.hasNext()){
					t2 = (ValuateSQL)i4.next();
					bw.write(t2.getSql());
					bw.newLine();
				}
				bw.write("評価値:"+v_sqls_p.getPenalty());
				bw.newLine();
				bw.newLine();
			}
		}catch(Exception e) {
			if ( t2!=null)System.out.println("\"" + t2.getSql() + "\"" + e);
			else System.out.println(e);				
			throw e;
		}finally{
			if (bw != null) {
				bw.close();
			}
			//ResultSetインターフェースの破棄
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statementインターフェースの破棄
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQLサーバ切断
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		if(debug) vsw.suspend();
	}

	//コネクションの解放
	public void close() throws Exception{
		cmanager.release();
	}

	/**
	 * @return timetable を戻します。
	 */
	public TimeTable getTimetable() {
		return timetable;
	}

	//初期解作成(変更予定)
	public void initsol() throws Exception{
		ResultSet rs = null;
		Connection con = null;
		Statement smt = null;
		Statement smt2 = null;
		//シードを与えて、Randomクラスのインスタンスを生成する。
		Random rand = new Random(Calendar.getInstance().getTimeInMillis());
		try{
			//MySQLサーバ接続
			con = cmanager.getConnection();
			//Statementインターフェースの生成
			smt = con.createStatement();
			smt2 = con.createStatement();
			String sql = "create table t_task_relation as select a.task_id, b.period_id, a.qualification_id, a.required_processors_lb, a.required_processors_ub from task_properties a, task_opportunities b where a.task_id = b.task_id order by task_id, period_id";
			smt.executeUpdate(sql);
			sql = "create table t_processor_relation as select a.processor_id, c.period_id, b.qualification_id from processor_properties a, processor_qualification b, processor_schedules c where a.processor_id = b.processor_id and a.processor_id = c.processor_id order by processor_id, period_id";
			smt.executeUpdate(sql);
			//時間割の初期化
			sql = "delete from timetablesql";
			smt.executeUpdate(sql);
			while(true){
				//講義とその講義開講可能時間をランダムに選ぶ
				sql = "select * from t_task_relation order by rand() limit 0,1";
				rs = smt.executeQuery(sql);
				int task_id,period_id,qualification_id,required_processors_lb,required_processors_ub;
				int z;
				if(rs.next()) {
					task_id = rs.getInt("task_id");
					period_id = rs.getInt("period_id");
					qualification_id = rs.getInt("qualification_id");
					required_processors_lb = rs.getInt("required_processors_lb");
					required_processors_ub = rs.getInt("required_processors_ub");
					z = rand.nextInt(required_processors_ub-required_processors_lb+1)+required_processors_lb;
				}else{
					break;
				}
				rs.close();
				boolean change = false;
				sql="select processor_id from t_processor_relation where period_id='"+period_id+"' and qualification_id ='"+ qualification_id +"' order by rand()";
				rs = smt.executeQuery(sql);
				for(int i=0;rs.next()&&i<z;i++){
					int processor_id = rs.getInt("processor_id");
					sql="insert into timetablesql(task_id,period_id,processor_id) values('" + task_id + "','" +period_id + "','" + processor_id +"')";
					smt2.executeUpdate(sql);
					sql = "delete from t_processor_relation where processor_id ='" + processor_id+"' and period_id = '" + period_id+"'";
					smt2.executeUpdate(sql);
					change = true;
				}
				rs.close();
				if(change){
					//この操作により各講義は、各講義固有の開講可能時間のうち一つに開講される。
					sql="delete from t_task_relation where task_id ='" + task_id+"'";
					smt.executeUpdate(sql);
				}else{
					sql="delete from t_task_relation where task_id ='" + task_id+"' and period_id ='"+period_id+"'";
					smt.executeUpdate(sql);
				}
			}
			sql = "drop table t_task_relation";
			smt.executeUpdate(sql);
			sql = "drop table t_processor_relation";
			smt.executeUpdate(sql);
		}catch(SQLException e) {
			System.err.println("理由：" + e.toString());
			throw e;
		}finally{
			//ResultSetインターフェースの破棄
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statementインターフェースの破棄
			if(smt!=null){
				smt.close();
				smt = null;
			}
			// Statementインターフェースの破棄
			if(smt2!=null){
				smt2.close();
				smt2 = null;
			}
			//MySQLサーバ切断
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		timetable.loadTimeTable();
	}

	//初期解作成(変更予定)
	public void initsolAddProject(int project_id) throws Exception{
		ResultSet rs = null;
		Connection con = null;
		Statement smt = null;
		Statement smt2 = null;
		//シードを与えて、Randomクラスのインスタンスを生成する。
		Random rand = new Random(Calendar.getInstance().getTimeInMillis());
		try{
			//MySQLサーバ接続
			con = cmanager.getConnection();
			//Statementインターフェースの生成
			smt = con.createStatement();
			smt2 = con.createStatement();
			String sql = "create table t_task_relation as select a.task_id, b.period_id, a.qualification_id, a.required_processors_lb, a.required_processors_ub from task_properties a, task_opportunities b where a.task_id = b.task_id and a.project_id ="+project_id+" order by task_id, period_id";
			smt.executeUpdate(sql);
			sql = "create table t_processor_relation as select a.processor_id, c.period_id, b.qualification_id from processor_properties a, processor_qualification b, processor_schedules c where a.processor_id = b.processor_id and a.processor_id = c.processor_id  and a.project_id ="+project_id+" order by processor_id, period_id";
			smt.executeUpdate(sql);
			//時間割の初期化
			sql = "delete from timetablesql";
			smt.executeUpdate(sql);
			while(true){
				//講義とその講義開講可能時間をランダムに選ぶ
				sql = "select * from t_task_relation order by rand() limit 0,1";
				rs = smt.executeQuery(sql);
				int task_id,period_id,qualification_id,required_processors_lb,required_processors_ub;
				int z;
				if(rs.next()) {
					task_id = rs.getInt("task_id");
					period_id = rs.getInt("period_id");
					qualification_id = rs.getInt("qualification_id");
					required_processors_lb = rs.getInt("required_processors_lb");
					required_processors_ub = rs.getInt("required_processors_ub");
					z = rand.nextInt(required_processors_ub-required_processors_lb+1)+required_processors_lb;
				}else{
					break;
				}
				rs.close();
				boolean change = false;
				sql="select processor_id from t_processor_relation where period_id='"+period_id+"' and qualification_id ='"+ qualification_id +"' order by rand()";
				rs = smt.executeQuery(sql);
				for(int i=0;rs.next()&&i<z;i++){
					int processor_id = rs.getInt("processor_id");
					sql="insert into timetablesql(task_id,period_id,processor_id) values('" + task_id + "','" +period_id + "','" + processor_id +"')";
					smt2.executeUpdate(sql);
					sql = "delete from t_processor_relation where processor_id ='" + processor_id+"' and period_id = '" + period_id+"'";
					smt2.executeUpdate(sql);
					change = true;
				}
				rs.close();
				if(change){
					//この操作により各講義は、各講義固有の開講可能時間のうち一つに開講される。
					sql="delete from t_task_relation where task_id ='" + task_id+"'";
					smt.executeUpdate(sql);
				}else{
					sql="delete from t_task_relation where task_id ='" + task_id+"' and period_id ='"+period_id+"'";
					smt.executeUpdate(sql);
				}
			}
			sql = "drop table t_task_relation";
			smt.executeUpdate(sql);
			sql = "drop table t_processor_relation";
			smt.executeUpdate(sql);
		}catch(SQLException e) {
			System.err.println("理由：" + e.toString());
			throw e;
		}finally{
			//ResultSetインターフェースの破棄
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statementインターフェースの破棄
			if(smt!=null){
				smt.close();
				smt = null;
			}
			// Statementインターフェースの破棄
			if(smt2!=null){
				smt2.close();
				smt2 = null;
			}
			//MySQLサーバ切断
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		timetable.loadTimeTable();
	}

	//初期解作成(変更予定)
	//連続する講義対応
	public void initsolTaskSeriesAddProject(int project_id) throws Exception{
		timetable.clearTimeTable();
		boolean debug2=false;
		Connection con = null;
		Statement smt = null;
		Statement smt2 = null;
		ResultSet rs = null;
		//シードを与えて、Randomクラスのインスタンスを生成する。
		Random rand = new Random(Calendar.getInstance().getTimeInMillis());
		try{
			//MySQLサーバ接続
			con = cmanager.getConnection();
			//Statementインターフェースの生成
			smt = con.createStatement();
			smt2 = con.createStatement();
			String sql = "create table t_task_relation as select a.task_id, b.period_id, a.qualification_id, a.required_processors_lb,a.required_processors_ub,a.task_series_num from task_properties a, task_opportunities b where a.task_id = b.task_id and a.project_id ="+project_id+" order by task_id, period_id";
			smt.executeUpdate(sql);
			sql = "create table t_processor_relation as select a.processor_id, c.period_id, b.qualification_id from processor_properties a, processor_qualification b, processor_schedules c where a.processor_id = b.processor_id and a.processor_id = c.processor_id  and a.project_id ="+project_id+" order by processor_id, period_id";
			smt.executeUpdate(sql);
			//時間割の初期化
			sql = "delete from timetablesql";
			smt.executeUpdate(sql);
			while(true){
				//講義とその講義開講可能時間をランダムに選ぶ
				sql = "select * from t_task_relation order by rand() limit 0,1";
				rs = smt.executeQuery(sql);
				int task_id,period_id,qualification_id,required_processors_lb,required_processors_ub;
				int task_series_num;
				int z;
				if(rs.next()) {
					task_id = rs.getInt("task_id");
					period_id = rs.getInt("period_id");
					qualification_id = rs.getInt("qualification_id");
					required_processors_lb = rs.getInt("required_processors_lb");
					required_processors_ub = rs.getInt("required_processors_ub");
					task_series_num = rs.getInt("task_series_num");
					z = rand.nextInt(required_processors_ub-required_processors_lb+1)+required_processors_lb;
					if(debug2)System.out.println("task_id:"+task_id);
				}else{
					break;
				}
				rs.close();
				if(task_series_num>1){
					Lecture lecture = searchLecture(task_id);
					boolean change = true;
					ArrayList t_periods = new ArrayList();
					int  n_period_id = period_id;
					for(int j = 1;j<task_series_num&change;j++){
						n_period_id = day_periods.getNextPeriods(n_period_id);
						if(!lecture.isPeriod(n_period_id)){
							change = false;
							break;
						}
						t_periods.add(new Integer(n_period_id));
					}
					if(change){
						ArrayList t_teachers = new ArrayList();
						sql="select processor_id from t_processor_relation where period_id='"+period_id+"' and qualification_id ='"+ qualification_id +"' order by rand()";
						rs = smt.executeQuery(sql);
						int p_num = 0;
						for(int i=0;rs.next()&&p_num<z;i++){
							int processor_id = rs.getInt("processor_id");
							boolean breakflag = true;
							Iterator iterator = t_periods.iterator();
							while(iterator.hasNext()&&breakflag){
								int n_period = ((Integer)iterator.next()).intValue();
								sql="select processor_id from t_processor_relation where period_id='"+n_period_id+"' and processor_id ='"+ processor_id +"'";
								rs = smt2.executeQuery(sql);
								if(rs.next()){
									if(!iterator.hasNext()){
										p_num++;
										t_teachers.add(new Integer(processor_id));
									}
								}else{
									breakflag = false;
								}
							}
						}
						if(p_num>0){
							Iterator iterator = t_teachers.iterator();
							while(iterator.hasNext()){
								int  t_processor_id= ((Integer)iterator.next()).intValue();
								Iterator iterator2 = t_periods.iterator();
								sql="insert into timetablesql(task_id,period_id,processor_id) values('" + task_id + "','" +period_id + "','" + t_processor_id +"')";
								smt.executeUpdate(sql);
								timetable.insertTimeTableRow(task_id,period_id,t_processor_id);
								if(debug2)System.out.println("insert ("+task_id+","+period_id+","+t_processor_id+")");
								sql = "delete from t_processor_relation where processor_id ='" + t_processor_id+"' and period_id = '" + period_id+"'";
								smt.executeUpdate(sql);
								while(iterator2.hasNext()){
									int t_period_id= ((Integer)iterator2.next()).intValue();
									if(debug2)System.out.println("insert ("+task_id+","+t_period_id+","+t_processor_id+")");
									String sql2="insert into timetablesql(task_id,period_id,processor_id) values('" + task_id + "','" +t_period_id + "','" + t_processor_id +"')";
									smt2.executeUpdate(sql2);
									sql2 = "delete from t_processor_relation where processor_id ='" + t_processor_id+"' and period_id = '" + t_period_id+"'";
									smt2.executeUpdate(sql2);
								}
							}
							if(debug2)System.out.println("delete task_id:"+task_id);
							//この操作により各講義は、各講義固有の開講可能時間のうち一つに開講される。
							sql="delete from t_task_relation where task_id ='" + task_id+"'";
							smt.executeUpdate(sql);
						}else{
							if(debug2)System.out.println("delete task_id:"+task_id + "period_id:"+period_id);
							sql="delete from t_task_relation where task_id ='" + task_id+"' and period_id ='"+period_id+"'";
							smt.executeUpdate(sql);
						}
					}else{
						if(debug2)System.out.println("delete task_id:"+task_id + "period_id:"+period_id);
						sql="delete from t_task_relation where task_id ='" + task_id+"' and period_id ='"+period_id+"'";
						smt.executeUpdate(sql);
					}
				}else{
					boolean change = false;
					sql="select processor_id from t_processor_relation where period_id='"+period_id+"' and qualification_id ='"+ qualification_id +"' order by rand()";
					rs = smt.executeQuery(sql);
					for(int i=0;rs.next()&&i<z;i++){
						int processor_id = rs.getInt("processor_id");
						sql="insert into timetablesql(task_id,period_id,processor_id) values('" + task_id + "','" +period_id + "','" + processor_id +"')";
						smt2.executeUpdate(sql);
						timetable.insertTimeTableRow(task_id,period_id,processor_id);
						if(debug2)System.out.println("insert ("+task_id+","+period_id+","+processor_id+")");
						String sql2 = "delete from t_processor_relation where processor_id ='" + processor_id+"' and period_id = '" + period_id+"'";
						smt2.executeUpdate(sql2);
						change = true;
					}
					rs.close();
					if(change){
						if(debug2)System.out.println("delete task_id:"+task_id);
						//この操作により各講義は、各講義固有の開講可能時間のうち一つに開講される。
						sql="delete from t_task_relation where task_id ='" + task_id+"'";
						smt.executeUpdate(sql);
					}else{
						if(debug2)System.out.println("delete task_id:"+task_id + "period_id:"+period_id);
						sql="delete from t_task_relation where task_id ='" + task_id+"' and period_id ='"+period_id+"'";
						smt.executeUpdate(sql);
					}
				}
			}
			sql = "drop table t_task_relation";
			smt.executeUpdate(sql);
			sql = "drop table t_processor_relation";
			smt.executeUpdate(sql);
		}catch(SQLException e) {
			System.err.println("理由：" + e.toString());
			throw e;
		}finally{
			//ResultSetインターフェースの破棄
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statementインターフェースの破棄
			if(smt!=null){
				smt.close();
				smt = null;
			}
			// Statementインターフェースの破棄
			if(smt2!=null){
				smt2.close();
				smt2 = null;
			}
			//MySQLサーバ切断
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		//timetable.initTimetable();
	}

	//初期解作成
	public void initsolNotRandom(int seed) throws Exception{
		ResultSet rs = null;
		Connection con = null;
		Statement smt = null;
		Statement smt2 = null;
		//Randomクラスのインスタンスを生成する。
		Random rand = new Random();
		try{
			//MySQLサーバ接続
			con = cmanager.getConnection();
			//Statementインターフェースの生成
			smt = con.createStatement();
			smt2 = con.createStatement();
			String sql = "create table t_task_relation as select a.task_id, b.period_id, a.qualification_id, a.required_processors_lb, a.required_processors_ub, c.day_id, b.preferred_level_task from task_properties a, task_opportunities b, period_properties c where a.task_id = b.task_id and b.period_id = c.period_id order by task_id, period_id";
			smt.executeUpdate(sql);
			sql = "create table t_processor_relation as select a.processor_id, c.period_id, b.qualification_id, a.employment, a.total_periods_lb, a.total_periods_ub, a.total_days_lb, a.total_days_ub, a.wage_level, d.day_id, c.preferred_level_proc from processor_properties a, processor_qualification b, processor_schedules c, period_properties d where a.processor_id = b.processor_id and c.period_id = d.period_id and a.processor_id = c.processor_id order by processor_id, period_id";
			smt.executeUpdate(sql);
			//時間割の初期化
			sql = "delete from timetablesql";
			smt.executeUpdate(sql);
			while(true){
				//講義とその講義開講可能時間をランダムに選ぶ
				sql = "select * from t_task_relation order by rand("+seed+") limit 0,1";
				rs = smt.executeQuery(sql);
				int task_id;
				int period_id;
				int qualification_id;
				int required_processors_lb;
				int required_processors_ub;
				int z;
				if(rs.next()) {
					task_id = rs.getInt("task_id");
					period_id = rs.getInt("period_id");
					qualification_id = rs.getInt("qualification_id");
					required_processors_lb = rs.getInt("required_processors_lb");
					required_processors_ub = rs.getInt("required_processors_ub");
					z = rand.nextInt(required_processors_ub-required_processors_lb+1)+required_processors_lb;
				}else{
					break;
				}
				rs.close();
				boolean change = false;
				sql="select processor_id from t_processor_relation where period_id='"+period_id+"' and qualification_id ='"+ qualification_id +"' order by rand("+seed+")";
				rs = smt.executeQuery(sql);
				for(int i=0;rs.next()&&i<z;i++){
					int processor_id = rs.getInt("processor_id");
					sql="insert into timetablesql(task_id,period_id,processor_id) values('" + task_id + "','" +period_id + "','" + processor_id +"')";
					smt2.executeUpdate(sql);
					String sql2 = "delete from t_processor_relation where processor_id ='" + processor_id+"' and period_id = '" + period_id+"'";
					smt2.executeUpdate(sql2);
					change = true;
					
				}
				rs.close();
				if(change){
					//この操作により各講義は、各講義固有の開講可能時間のうち一つに開講される。
					sql="delete from t_task_relation where task_id ='" + task_id+"'";
					smt.executeUpdate(sql);
				}else{
					sql="delete from t_task_relation where task_id ='" + task_id+"' and period_id ='"+period_id+"'";
					smt.executeUpdate(sql);
				}
			}
			sql = "drop table t_task_relation";
			smt.executeUpdate(sql);
			sql = "drop table t_processor_relation";
			smt.executeUpdate(sql);
		}catch(SQLException e) {
			System.err.println("理由：" + e.toString());
			throw e;
		}finally{
			//ResultSetインターフェースの破棄
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statementインターフェースの破棄
			if(smt!=null){
				smt.close();
				smt = null;
			}
			// Statementインターフェースの破棄
			if(smt2!=null){
				smt2.close();
				smt2 = null;
			}
			//MySQLサーバ切断
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		//実験用
		timetable.loadTimeTable();
	}

//	初期解作成(変更予定)
	//連続する講義対応
	public void initsolNotRandomTaskSeriesAddProject(int seed,int project_id) throws Exception{
		timetable.clearTimeTable();
		boolean debug2=false;
		ResultSet rs = null;
		Connection con = null;
		Statement smt = null;
		Statement smt2 = null;
		//シードを与えて、Randomクラスのインスタンスを生成する。
		Random rand = new Random();
		try{
			//MySQLサーバ接続
			con = cmanager.getConnection();
			//Statementインターフェースの生成
			smt = con.createStatement();
			smt2 = con.createStatement();
			String sql = "create table t_task_relation as select a.task_id, b.period_id, a.qualification_id, a.required_processors_lb,a.required_processors_ub,a.task_series_num from task_properties a, task_opportunities b where a.task_id = b.task_id and a.project_id ="+project_id+" order by task_id, period_id";
			smt.executeUpdate(sql);
			sql = "create table t_processor_relation as select a.processor_id, c.period_id, b.qualification_id from processor_properties a, processor_qualification b, processor_schedules c where a.processor_id = b.processor_id and a.processor_id = c.processor_id  and a.project_id ="+project_id+" order by processor_id, period_id";
			smt.executeUpdate(sql);
			//時間割の初期化
			sql = "delete from timetablesql";
			smt.executeUpdate(sql);
			while(true){
				//講義とその講義開講可能時間をランダムに選ぶ
				sql = "select * from t_task_relation order by rand("+seed+") limit 0,1";
				rs = smt.executeQuery(sql);
				int task_id,period_id,qualification_id,required_processors_lb,required_processors_ub;
				int task_series_num;
				int z;
				if(rs.next()) {
					task_id = rs.getInt("task_id");
					period_id = rs.getInt("period_id");
					qualification_id = rs.getInt("qualification_id");
					required_processors_lb = rs.getInt("required_processors_lb");
					required_processors_ub = rs.getInt("required_processors_ub");
					task_series_num = rs.getInt("task_series_num");
					z = rand.nextInt(required_processors_ub-required_processors_lb+1)+required_processors_lb;
					if(debug2)System.out.println("task_id:"+task_id);
				}else{
					break;
				}
				rs.close();
				if(task_series_num>1){
					Lecture lecture = searchLecture(task_id);
					boolean change = true;
					ArrayList t_periods = new ArrayList();
					int  n_period_id = period_id;
					for(int j = 1;j<task_series_num&change;j++){
						n_period_id = day_periods.getNextPeriods(n_period_id);
						if(!lecture.isPeriod(n_period_id)){
							change = false;
							break;
						}
						t_periods.add(new Integer(n_period_id));
					}
					if(change){
						ArrayList t_teachers = new ArrayList();
						sql="select processor_id from t_processor_relation where period_id='"+period_id+"' and qualification_id ='"+ qualification_id +"' order by rand("+seed+")";
						rs = smt.executeQuery(sql);
						int p_num = 0;
						for(int i=0;rs.next()&&p_num<z;i++){
							int processor_id = rs.getInt("processor_id");
							boolean breakflag = true;
							Iterator iterator = t_periods.iterator();
							while(iterator.hasNext()&&breakflag){
								int n_period = ((Integer)iterator.next()).intValue();
								sql="select processor_id from t_processor_relation where period_id='"+n_period_id+"' and processor_id ='"+ processor_id +"'";
								rs = smt2.executeQuery(sql);
								if(rs.next()){
									if(!iterator.hasNext()){
										p_num++;
										t_teachers.add(new Integer(processor_id));
									}
								}else{
									breakflag = false;
								}
							}
						}
						if(p_num>0){
							Iterator iterator = t_teachers.iterator();
							while(iterator.hasNext()){
								int  t_processor_id= ((Integer)iterator.next()).intValue();
								Iterator iterator2 = t_periods.iterator();
								sql="insert into timetablesql(task_id,period_id,processor_id) values('" + task_id + "','" +period_id + "','" + t_processor_id +"')";
								smt.executeUpdate(sql);
								timetable.insertTimeTableRow(task_id,period_id,t_processor_id);
								if(debug2)System.out.println("insert ("+task_id+","+period_id+","+t_processor_id+")");
								sql = "delete from t_processor_relation where processor_id ='" + t_processor_id+"' and period_id = '" + period_id+"'";
								smt.executeUpdate(sql);
								while(iterator2.hasNext()){
									int t_period_id= ((Integer)iterator2.next()).intValue();
									if(debug2)System.out.println("insert ("+task_id+","+t_period_id+","+t_processor_id+")");
									String sql2="insert into timetablesql(task_id,period_id,processor_id) values('" + task_id + "','" +t_period_id + "','" + t_processor_id +"')";
									smt2.executeUpdate(sql2);
									sql2 = "delete from t_processor_relation where processor_id ='" + t_processor_id+"' and period_id = '" + t_period_id+"'";
									smt2.executeUpdate(sql2);
								}
							}
							if(debug2)System.out.println("delete task_id:"+task_id);
							//この操作により各講義は、各講義固有の開講可能時間のうち一つに開講される。
							sql="delete from t_task_relation where task_id ='" + task_id+"'";
							smt.executeUpdate(sql);
						}else{
							if(debug2)System.out.println("delete task_id:"+task_id + "period_id:"+period_id);
							sql="delete from t_task_relation where task_id ='" + task_id+"' and period_id ='"+period_id+"'";
							smt.executeUpdate(sql);
						}
					}else{
						if(debug2)System.out.println("delete task_id:"+task_id + "period_id:"+period_id);
						sql="delete from t_task_relation where task_id ='" + task_id+"' and period_id ='"+period_id+"'";
						smt.executeUpdate(sql);
					}
				}else{
					boolean change = false;
					sql="select processor_id from t_processor_relation where period_id='"+period_id+"' and qualification_id ='"+ qualification_id +"' order by rand()";
					rs = smt.executeQuery(sql);
					for(int i=0;rs.next()&&i<z;i++){
						int processor_id = rs.getInt("processor_id");
						sql="insert into timetablesql(task_id,period_id,processor_id) values('" + task_id + "','" +period_id + "','" + processor_id +"')";
						smt2.executeUpdate(sql);
						timetable.insertTimeTableRow(task_id,period_id,processor_id);
						if(debug2)System.out.println("insert ("+task_id+","+period_id+","+processor_id+")");
						String sql2 = "delete from t_processor_relation where processor_id ='" + processor_id+"' and period_id = '" + period_id+"'";
						smt2.executeUpdate(sql2);
						change = true;
					}
					rs.close();
					if(change){
						if(debug2)System.out.println("delete task_id:"+task_id);
						//この操作により各講義は、各講義固有の開講可能時間のうち一つに開講される。
						sql="delete from t_task_relation where task_id ='" + task_id+"'";
						smt.executeUpdate(sql);
					}else{
						if(debug2)System.out.println("delete task_id:"+task_id + "period_id:"+period_id);
						sql="delete from t_task_relation where task_id ='" + task_id+"' and period_id ='"+period_id+"'";
						smt.executeUpdate(sql);
					}
				}
			}
			sql = "drop table t_task_relation";
			smt.executeUpdate(sql);
			sql = "drop table t_processor_relation";
			smt.executeUpdate(sql);
		}catch(SQLException e) {
			System.err.println("理由：" + e.toString());
			throw e;
		}finally{
			//ResultSetインターフェースの破棄
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statementインターフェースの破棄
			if(smt!=null){
				smt.close();
				smt = null;
			}
			// Statementインターフェースの破棄
			if(smt2!=null){
				smt2.close();
				smt2 = null;
			}
			//MySQLサーバ切断
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		//timetable.initTimetable();
	}

	//局所探索(改善１,２)(プログラム中のテーブルを初期化しない)(操作用テーブルを使用しない)
	public int local_search() throws Exception
	{
		//if(debug) SWflag = true;
		int i;
		for (i=1; i<=50; i++)
		{
			if(debug) System.out.println(i+"回目");
			if (!search_add2()&&!search_move()&&!search_change()&&!search_swap()&&!search_add()&&!search_leave()){
				if(debug) System.out.println("これ以上無し");
				break;
			}
			if(debug){
				System.out.println("count_offence:"+count_offence());
				System.out.println("valuate:"+valuateTimeTableSQL());
			}
		}
		if(debug) System.out.println("評価にかかった時間:"+vsw);
		return i;
	}

	//局所探索(改善１,２)
	//(プログラム中のテーブルを初期化しない)
	//(操作用テーブルを使用しない)
	//連続する時間割に対応
	public int task_series_local_search() throws Exception
	{
		//if(debug) SWflag = true;
		int i;
		for (i=1; i<=50; i++)
		{
			if(debug) System.out.println(i+"回目");
			if(!search_move_task_series()&&!search_change_task_series()&&!search_swap_task_series()&&!search_add_task_series()&&!search_leave_task_series()){
			//if (!search_change_task_series()&&!search_swap_task_series()&&!search_add_task_series()&&!search_leave_task_series()){
				if(debug) System.out.println("これ以上無し");
				break;
			}
			if(debug){
				//System.out.println("count_offence:"+task_series_count_offence());
				//System.out.println("count_offenceDetail:"+task_series_count_offenceDetail());
				//System.out.println("valuate:"+valuateTimeTableSQL());
				//System.out.println("valuate:"+valuateTimeTableSQLDBDetail());
			}
		}
		if(debug) System.out.println("評価にかかった時間:"+vsw);
		//printInsertTimeTable();
		return i;
	}
	
	//(プログラム中のテーブルを初期化しない)
	//近傍操作move:講義を行う時間を変更する(すべて調べる)(メソッド使用)
	//近傍moveを探索し、改善解があればその中で最良のものに移動しtrueを返す。なければfalseを返す。
	public boolean search_move() throws Exception{
		boolean change = false;
		int min_val,min_off,new_val,new_off;
		ArrayList new_teacher_ids = null;
		int new_task_id = 0,new_period_id = 0,del_period_id = 0;
		//値の評価
		min_val = valuateTimeTableSQL();
		min_off = timetable.countOffence();
		//現在時間割に入っている講義を取得
		ArrayList tps = timetable.getTaskPeriods();
		Iterator i_tps = tps.iterator();
		while(i_tps.hasNext()) {			
			TaskPeriod tp = (TaskPeriod)i_tps.next();
			int task_id = tp.getTask_id();
			int period_id = tp.getPeriod_id();
			int qualification_id;
			int required_processors_lb;
			int required_processors_ub;
			Lecture l = searchLecture(task_id);
			qualification_id = l.getQualification_id();
			required_processors_lb = l.getRequired_processors_lb();
			required_processors_ub = l.getRequired_processors_ub();
			//task_idの開講可能時間の取得
			Iterator i = l.getPeriods().iterator();
			while(i.hasNext()){
				PeriodDesire lpd = (PeriodDesire)i.next();
				int period_id2 = lpd.getPeriod_id();
				//講義の開講時間が現在開講されている時間でない
				if(period_id!=period_id2){
					//task_idを担当している講師を入れる
					//period_idにtask_idを担当している講師を取得
					ArrayList t_teacher3 = timetable.getProcessors(task_id,period_id);
					//task_idを除く
					deleteTaskTimeTable(task_id,period_id);
					timetable.deleteTaskTimeTable(task_id,period_id);
					//新しくtask_idを担当する講師
					ArrayList t_teachers2 = new ArrayList();
					Iterator i2 = teachers.iterator();
					while(i2.hasNext()){
						Teacher t = (Teacher)i2.next();
						//講師tが講義を担当できるか
						if(t.isQualification(qualification_id)){
							if(t.isPeriod(period_id2)){
								int processor_id = t.getProcessor_id();
								//講師は同じ時間に複数の講義をしないことを調べる
								if(timetable.isNotTasks(processor_id,period_id2)){
									t_teachers2.add(new Integer(processor_id));
									//System.out.println("追加可能:"+processor_id);
								}
							}
						}
					}
					//講義.担当人数下限 <= z <= 講義.担当人数上限
					for(int z=required_processors_lb;z<=required_processors_ub;z++){
						//period_id2に担当可能な講師を取得
						ArrayList t_teachers = null;
						Enumeration e = new CombEnum(t_teachers2.toArray(), z);
						while(e.hasMoreElements()) {
							t_teachers = new ArrayList();
							Object[] a = (Object[])e.nextElement();
							for(int num2=0; num2<a.length; num2++){
								int processor_id = ((Integer)a[num2]).intValue();
								//講師の追加
								insertTimeTableRow(task_id,period_id2,processor_id);
								timetable.insertTimeTableRow(task_id,period_id2,processor_id);
								t_teachers.add(new Integer(processor_id));
							}
							//値の評価
							new_off = timetable.countOffence();
							new_val = valuateTimeTableSQL();
							if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
							{
								//変化した物を入れておく
								new_teacher_ids = t_teachers;
								new_task_id = task_id;
								new_period_id = period_id2;
								del_period_id = period_id;
								min_off = new_off;
								min_val = new_val;
								//if(debug)System.out.println(""+min_val);
								change = true;
							}
							//追加した講義の削除
							deleteTaskTimeTable(task_id);
							timetable.deleteTaskTimeTable(task_id);
						}
					}
					//除いたtask_idの追加
					Iterator i4 = t_teacher3.iterator();
					while(i4.hasNext()){
						int processor_id = ((Integer)i4.next()).intValue();
						insertTimeTableRow(task_id,period_id,processor_id);
						timetable.insertTimeTableRow(task_id,period_id,processor_id);
					}
				}
			}
		}
		//解が改善された時
		if(change){
			//変更の表示
			if(debug)System.out.println("move:task_id:"+new_task_id+"がperiod_id:"+del_period_id+"からperiod_id:"+new_period_id+"に移動");
			deleteTaskTimeTable(new_task_id);
			timetable.deleteTaskTimeTable(new_task_id);
			Iterator i = new_teacher_ids.iterator();
			int processor_id;
			while(i.hasNext()){
				processor_id = ((Integer)i.next()).intValue();
				if(debug)System.out.println("担当講師:"+processor_id);
				insertTimeTableRow(new_task_id,new_period_id,processor_id);
				timetable.insertTimeTableRow(new_task_id,new_period_id,processor_id);
			}
			if(debug)printInsertTimeTable();
		}
		return change;
	}

	//(プログラム中のテーブルを初期化しない)
	//近傍操作move:講義を行う時間を変更する(すべて調べる)(メソッド使用)
	//近傍moveを探索し、改善解があればその中で最良のものに移動しtrueを返す。なければfalseを返す。
	public boolean search_move_task_series() throws Exception{
		if(debug)System.out.println("move start");
		boolean change = false;
		int min_val,min_off,new_val,new_off;
		ArrayList new_teacher_ids = null;
		ArrayList new_n_period_ids = null;
		int new_task_id = 0,new_period_id = 0,new_task_series_num=1,del_period_id = 0;
		//値の評価
		min_val = valuateTimeTableSQL();
		min_off = timetable.taskSeriesCountOffence();
		//現在時間割に入っている講義を取得
		ArrayList tps = timetable.getTaskPeriods();
		Iterator i_tps = tps.iterator();
		while(i_tps.hasNext()) {			
			TaskPeriod tp = (TaskPeriod)i_tps.next();
			int task_id = tp.getTask_id();
			int period_id = tp.getPeriod_id();
			int qualification_id,required_processors_lb,required_processors_ub,task_series_num;
			Lecture l = searchLecture(task_id);
			qualification_id = l.getQualification_id();
			required_processors_lb = l.getRequired_processors_lb();
			required_processors_ub = l.getRequired_processors_ub();
			task_series_num = l.getTask_series_num();
			if(task_series_num==1){
				//task_idの開講可能時間の取得
				Iterator i = l.getPeriods().iterator();
				while(i.hasNext()){
					PeriodDesire lpd = (PeriodDesire)i.next();
					int period_id2 = lpd.getPeriod_id();
					//講義の開講時間が現在開講されている時間でない
					if(period_id!=period_id2){
						//task_idを担当している講師を入れる
						//period_idにtask_idを担当している講師を取得
						ArrayList t_teacher3 = timetable.getProcessors(task_id,period_id);
						//task_idを除く
						deleteTaskTimeTable(task_id,period_id);
						timetable.deleteTaskTimeTable(task_id,period_id);
						//新しくtask_idを担当する講師
						ArrayList t_teachers2 = new ArrayList();
						Iterator i2 = teachers.iterator();
						while(i2.hasNext()){
							Teacher t = (Teacher)i2.next();
							//講師tが講義を担当できるか
							if(t.isQualification(qualification_id)){
								if(t.isPeriod(period_id2)){
									int processor_id = t.getProcessor_id();
									//講師は同じ時間に複数の講義をしないことを調べる
									if(timetable.isNotTasks(processor_id,period_id2)){
										t_teachers2.add(new Integer(processor_id));
										//System.out.println("追加可能:"+processor_id);
									}
								}
							}
						}
						//講義.担当人数下限 <= z <= 講義.担当人数上限
						for(int z=required_processors_lb;z<=required_processors_ub;z++){
							//period_id2に担当可能な講師を取得
							ArrayList t_teachers = null;
							Enumeration e = new CombEnum(t_teachers2.toArray(), z);
							while(e.hasMoreElements()) {
								t_teachers = new ArrayList();
								Object[] a = (Object[])e.nextElement();
								for(int num2=0; num2<a.length; num2++){
									int processor_id = ((Integer)a[num2]).intValue();
									//講師の追加
									insertTimeTableRow(task_id,period_id2,processor_id);
									timetable.insertTimeTableRow(task_id,period_id2,processor_id);
									t_teachers.add(new Integer(processor_id));
								}
								//値の評価
								new_off = timetable.taskSeriesCountOffence();
								new_val = valuateTimeTableSQL();
								if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
								{
									//変化した物を入れておく
									new_task_series_num = task_series_num;
									new_teacher_ids = t_teachers;
									new_task_id = task_id;
									new_period_id = period_id2;
									del_period_id = period_id;
									min_off = new_off;
									min_val = new_val;
									//if(debug)System.out.println(""+min_val);
									change = true;
								}
								//追加した講義の削除
								deleteTaskTimeTable(task_id);
								timetable.deleteTaskTimeTable(task_id);
							}
						}
						//除いたtask_idの追加
						Iterator i4 = t_teacher3.iterator();
						while(i4.hasNext()){
							int processor_id = ((Integer)i4.next()).intValue();
							insertTimeTableRow(task_id,period_id,processor_id);
							timetable.insertTimeTableRow(task_id,period_id,processor_id);
						}
					}
				}
			}else{
				//task_idの開講可能時間の取得
				Iterator i = l.getPeriods().iterator();
				while(i.hasNext()){
					PeriodDesire lpd = (PeriodDesire)i.next();
					int period_id2 = lpd.getPeriod_id();
					//講義の開講時間が現在開講されている時間でない
					if(period_id!=period_id2){
						ArrayList t_periods = new ArrayList();
						boolean break_flag = false;
						int next_period_id = period_id2;
						for(int num=1;num<task_series_num&&!break_flag;num++){
							next_period_id = day_periods.getNextPeriods(next_period_id);
							if(!l.isPeriod(next_period_id)){
								break_flag = true;break;
							}
							t_periods.add(new Integer(next_period_id));
						}
						if(break_flag)break;
						//task_idを担当している講師を入れる
						//period_idにtask_idを担当している講師を取得
						ArrayList t_teacher3 = timetable.getProcessors(task_id,period_id);
						//task_idを除く
						//deleteTaskTimeTable(task_id,period_id);
						deleteTaskTimeTable(task_id);
						timetable.deleteTaskTimeTable(task_id,period_id);
						//新しくtask_idを担当する講師
						ArrayList t_teachers2 = new ArrayList();
						Iterator i2 = teachers.iterator();
						while(i2.hasNext()){
							Teacher t = (Teacher)i2.next();
							//講師tが講義を担当できるか
							if(t.isQualification(qualification_id)){
								if(t.isPeriod(period_id2)){
									//講師は同じ時間に複数の講義をしないことを調べる
									if(timetable.isNotTasks(t.getProcessor_id(),period_id2)){
										next_period_id=period_id2;
										break_flag = false;
										for(int num=1;num<task_series_num&&!break_flag;num++){
											next_period_id = day_periods.getNextPeriods(next_period_id);
											if(!t.isPeriod(next_period_id)){
												break_flag = true;break;
											}
											if(!timetable.isNotTasks(t.getProcessor_id(),next_period_id)){
												break_flag = true;break;
											}
										}
										if(break_flag)break;
										t_teachers2.add(new Integer(t.getProcessor_id()));
										//System.out.println("追加可能:"+processor_id);
									}
								}
							}
						}
						//講義.担当人数下限 <= z <= 講義.担当人数上限
						for(int z=required_processors_lb;z<=required_processors_ub;z++){
							//period_id2に担当可能な講師を取得
							ArrayList t_teachers = null;
							Enumeration e = new CombEnum(t_teachers2.toArray(), z);
							while(e.hasMoreElements()) {
								t_teachers = new ArrayList();
								Object[] a = (Object[])e.nextElement();
								for(int num2=0; num2<a.length; num2++){
									int processor_id = ((Integer)a[num2]).intValue();
									//講師の追加
									insertTimeTableRow(task_id,period_id2,processor_id);
									timetable.insertTimeTableRow(task_id,period_id2,processor_id);
									t_teachers.add(new Integer(processor_id));
									Iterator iterator = t_periods.iterator();
									while(iterator.hasNext()){
										int n_period = ((Integer)iterator.next()).intValue();
										insertTimeTableRow(task_id,n_period,processor_id);
									}
								}
								//値の評価
								new_off = timetable.taskSeriesCountOffence();
								new_val = valuateTimeTableSQL();
								if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
								{
									//変化した物を入れておく
									new_task_series_num = task_series_num;
									new_teacher_ids = t_teachers;
									new_n_period_ids = t_periods;
									new_task_id = task_id;
									new_period_id = period_id2;
									del_period_id = period_id;
									min_off = new_off;
									min_val = new_val;
									//if(debug)System.out.println(""+min_val);
									change = true;
								}
								//追加した講義の削除
								deleteTaskTimeTable(task_id);
								timetable.deleteTaskTimeTable(task_id);
							}
						}
						//除いたtask_idの追加
						Iterator i4 = t_teacher3.iterator();
						while(i4.hasNext()){
							int processor_id = ((Integer)i4.next()).intValue();
							insertTimeTableRow(task_id,period_id,processor_id);
							timetable.insertTimeTableRow(task_id,period_id,processor_id);
							int n_period = period_id;
							for(int num=1;num<task_series_num;num++){
								n_period = day_periods.getNextPeriods(n_period);
								insertTimeTableRow(task_id,n_period,processor_id);
							}
						}
					}
				}
				
			}
		}
		//解が改善された時
		if(change){
			//変更の表示
			if(debug)System.out.println("move:task_id:"+new_task_id+"がperiod_id:"+del_period_id+"からperiod_id:"+new_period_id+"に移動");
			deleteTaskTimeTable(new_task_id);
			timetable.deleteTaskTimeTable(new_task_id);
			Iterator i = new_teacher_ids.iterator();
			int processor_id;
			if(new_task_series_num==1){
				while(i.hasNext()){
					processor_id = ((Integer)i.next()).intValue();
					if(debug)System.out.println("担当講師:"+processor_id);
					insertTimeTableRow(new_task_id,new_period_id,processor_id);
					timetable.insertTimeTableRow(new_task_id,new_period_id,processor_id);
				}
			}else{
				while(i.hasNext()){
					processor_id = ((Integer)i.next()).intValue();
					if(debug)System.out.println("担当講師:"+processor_id);
					insertTimeTableRow(new_task_id,new_period_id,processor_id);
					timetable.insertTimeTableRow(new_task_id,new_period_id,processor_id);
					Iterator iterator = new_n_period_ids.iterator();
					int n_period;
					while(iterator.hasNext()){
						n_period = ((Integer)iterator.next()).intValue();
						insertTimeTableRow(new_task_id,n_period,processor_id);
					}
				}
			}
			//if(debug)printInsertTimeTable();
		}
		return change;
	}

	//(プログラム中のテーブルを初期化しない)
	//近傍操作move:講義を行う時間を変更する(すべて調べる)(メソッド使用)
	//近傍moveを探索し、改善解があればその中で最良のものに移動しtrueを返す。なければfalseを返す。
	public boolean search_move_task_series2() throws Exception{
		boolean change = false;
		int min_val,min_off,new_val,new_off;
		ArrayList new_teacher_ids = null;
		ArrayList new_n_period_ids = null;
		int new_task_id = 0,new_period_id = 0,new_task_series_num=1,del_period_id = 0;
		//値の評価
		min_val = valuateTimeTableSQL();
		min_off = timetable.taskSeriesCountOffence();
		//現在時間割に入っている講義を取得
		ArrayList tps = timetable.getTaskPeriods();
		Iterator i_tps = tps.iterator();
		while(i_tps.hasNext()) {			
			TaskPeriod tp = (TaskPeriod)i_tps.next();
			int task_id = tp.getTask_id();
			int period_id = tp.getPeriod_id();
			int qualification_id,required_processors_lb,required_processors_ub,task_series_num;
			Lecture l = searchLecture(task_id);
			qualification_id = l.getQualification_id();
			required_processors_lb = l.getRequired_processors_lb();
			required_processors_ub = l.getRequired_processors_ub();
			task_series_num = l.getTask_series_num();
			if(task_series_num==1){
				//task_idの開講可能時間の取得
				Iterator i = l.getPeriods().iterator();
				while(i.hasNext()){
					PeriodDesire lpd = (PeriodDesire)i.next();
					int period_id2 = lpd.getPeriod_id();
					//講義の開講時間が現在開講されている時間でない
					if(period_id!=period_id2){
						//task_idを担当している講師を入れる
						//period_idにtask_idを担当している講師を取得
						ArrayList t_teacher3 = timetable.getProcessors(task_id,period_id);
						//task_idを除く
						deleteTaskTimeTable(task_id,period_id);
						timetable.deleteTaskTimeTable(task_id,period_id);
						//新しくtask_idを担当する講師
						ArrayList t_teachers2 = new ArrayList();
						Iterator i2 = teachers.iterator();
						while(i2.hasNext()){
							Teacher t = (Teacher)i2.next();
							//講師tが講義を担当できるか
							if(t.isQualification(qualification_id)){
								if(t.isPeriod(period_id2)){
									int processor_id = t.getProcessor_id();
									//講師は同じ時間に複数の講義をしないことを調べる
									if(timetable.isNotTasks(processor_id,period_id2)){
										t_teachers2.add(new Integer(processor_id));
										//System.out.println("追加可能:"+processor_id);
									}
								}
							}
						}
						//講義.担当人数下限 <= z <= 講義.担当人数上限
						for(int z=required_processors_lb;z<=required_processors_ub;z++){
							//period_id2に担当可能な講師を取得
							ArrayList t_teachers = null;
							Enumeration e = new CombEnum(t_teachers2.toArray(), z);
							while(e.hasMoreElements()) {
								t_teachers = new ArrayList();
								Object[] a = (Object[])e.nextElement();
								for(int num2=0; num2<a.length; num2++){
									int processor_id = ((Integer)a[num2]).intValue();
									//講師の追加
									insertTimeTableRow(task_id,period_id2,processor_id);
									timetable.insertTimeTableRow(task_id,period_id2,processor_id);
									t_teachers.add(new Integer(processor_id));
								}
								//値の評価
								new_off = timetable.taskSeriesCountOffence();
								new_val = valuateTimeTableSQL();
								if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
								{
									//変化した物を入れておく
									new_task_series_num = task_series_num;
									new_teacher_ids = t_teachers;
									new_task_id = task_id;
									new_period_id = period_id2;
									del_period_id = period_id;
									min_off = new_off;
									min_val = new_val;
									//if(debug)System.out.println(""+min_val);
									change = true;
								}
								//追加した講義の削除
								deleteTaskTimeTable(task_id);
								timetable.deleteTaskTimeTable(task_id);
							}
						}
						//除いたtask_idの追加
						Iterator i4 = t_teacher3.iterator();
						while(i4.hasNext()){
							int processor_id = ((Integer)i4.next()).intValue();
							insertTimeTableRow(task_id,period_id,processor_id);
							timetable.insertTimeTableRow(task_id,period_id,processor_id);
						}
					}
				}
			}else{
				//task_idの開講可能時間の取得
				Iterator i = l.getPeriods().iterator();
				while(i.hasNext()){
					PeriodDesire lpd = (PeriodDesire)i.next();
					int period_id2 = lpd.getPeriod_id();
					//講義の開講時間が現在開講されている時間でない
					if(period_id!=period_id2){
						ArrayList t_periods = new ArrayList();
						boolean break_flag = false;
						int next_period_id = period_id2;
						for(int num=1;num<task_series_num&&!break_flag;num++){
							next_period_id = day_periods.getNextPeriods(next_period_id);
							if(!l.isPeriod(next_period_id)){
								break_flag = true;break;
							}
							t_periods.add(new Integer(next_period_id));
						}
						if(break_flag)break;
						//task_idを担当している講師を入れる
						//period_idにtask_idを担当している講師を取得
						ArrayList t_teacher3 = timetable.getProcessors(task_id,period_id);
						//task_idを除く
						//deleteTaskTimeTable(task_id,period_id);
						deleteTaskTimeTable(task_id);
						timetable.deleteTaskTimeTable(task_id,period_id);
						//新しくtask_idを担当する講師
						ArrayList t_teachers2 = new ArrayList();
						Iterator i2 = teachers.iterator();
						while(i2.hasNext()){
							Teacher t = (Teacher)i2.next();
							//講師tが講義を担当できるか
							if(t.isQualification(qualification_id)){
								if(t.isPeriod(period_id2)){
									//講師は同じ時間に複数の講義をしないことを調べる
									if(timetable.isNotTasks(t.getProcessor_id(),period_id2)){
										next_period_id=period_id2;
										break_flag = false;
										for(int num=1;num<task_series_num&&!break_flag;num++){
											next_period_id = day_periods.getNextPeriods(next_period_id);
											if(!t.isPeriod(next_period_id)){
												break_flag = true;break;
											}
											if(!timetable.isNotTasks(t.getProcessor_id(),next_period_id)){
												break_flag = true;break;
											}
										}
										if(break_flag)break;
										t_teachers2.add(new Integer(t.getProcessor_id()));
										//System.out.println("追加可能:"+processor_id);
									}
								}
							}
						}
						//講義.担当人数下限 <= z <= 講義.担当人数上限
						for(int z=required_processors_lb;z<=required_processors_ub;z++){
							//period_id2に担当可能な講師を取得
							ArrayList t_teachers = null;
							Enumeration e = new CombEnum(t_teachers2.toArray(), z);
							while(e.hasMoreElements()) {
								t_teachers = new ArrayList();
								Object[] a = (Object[])e.nextElement();
								for(int num2=0; num2<a.length; num2++){
									int processor_id = ((Integer)a[num2]).intValue();
									//講師の追加
									insertTimeTableRow(task_id,period_id2,processor_id);
									timetable.insertTimeTableRow(task_id,period_id2,processor_id);
									t_teachers.add(new Integer(processor_id));
									Iterator iterator = t_periods.iterator();
									while(iterator.hasNext()){
										int n_period = ((Integer)iterator.next()).intValue();
										insertTimeTableRow(task_id,n_period,processor_id);
									}
								}
								//値の評価
								new_off = timetable.taskSeriesCountOffence();
								new_val = valuateTimeTableSQL();
								if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
								{
									//変化した物を入れておく
									new_task_series_num = task_series_num;
									new_teacher_ids = t_teachers;
									new_n_period_ids = t_periods;
									new_task_id = task_id;
									new_period_id = period_id2;
									del_period_id = period_id;
									min_off = new_off;
									min_val = new_val;
									//if(debug)System.out.println(""+min_val);
									change = true;
								}
								//追加した講義の削除
								deleteTaskTimeTable(task_id);
								timetable.deleteTaskTimeTable(task_id);
							}
						}
						//除いたtask_idの追加
						Iterator i4 = t_teacher3.iterator();
						while(i4.hasNext()){
							int processor_id = ((Integer)i4.next()).intValue();
							insertTimeTableRow(task_id,period_id,processor_id);
							timetable.insertTimeTableRow(task_id,period_id,processor_id);
							int n_period = period_id;
							for(int num=1;num<task_series_num;num++){
								n_period = day_periods.getNextPeriods(n_period);
								insertTimeTableRow(task_id,n_period,processor_id);
							}
						}
					}
				}
				
			}
		}
		//解が改善された時
		if(change){
			//変更の表示
			if(debug)System.out.println("move:task_id:"+new_task_id+"がperiod_id:"+del_period_id+"からperiod_id:"+new_period_id+"に移動");
			deleteTaskTimeTable(new_task_id);
			timetable.deleteTaskTimeTable(new_task_id);
			Iterator i = new_teacher_ids.iterator();
			int processor_id;
			if(new_task_series_num==1){
				while(i.hasNext()){
					processor_id = ((Integer)i.next()).intValue();
					if(debug)System.out.println("担当講師:"+processor_id);
					insertTimeTableRow(new_task_id,new_period_id,processor_id);
					timetable.insertTimeTableRow(new_task_id,new_period_id,processor_id);
				}
			}else{
				while(i.hasNext()){
					processor_id = ((Integer)i.next()).intValue();
					if(debug)System.out.println("担当講師:"+processor_id);
					insertTimeTableRow(new_task_id,new_period_id,processor_id);
					timetable.insertTimeTableRow(new_task_id,new_period_id,processor_id);
					Iterator iterator = new_n_period_ids.iterator();
					int n_period;
					while(iterator.hasNext()){
						n_period = ((Integer)iterator.next()).intValue();
						insertTimeTableRow(new_task_id,n_period,processor_id);
					}
				}
			}
			//if(debug)printInsertTimeTable();
		}
		return change;
	}

	//(プログラム中のテーブルを初期化しない)	
	//近傍操作Change:講義の担当講師を変更する(メソッド使用)
	//近傍changeを探索し、改善解があればその中で最良のものに移動しtrueを返す。なければfalseを返す。
	public boolean search_change() throws Exception{
		ArrayList t_timetable = null;
		boolean change = false;
		int min_val,min_off,new_val,new_off;
		int del_processor_id = 0,new_task_id = 0,new_processor_id = 0,new_period_id = 0;
		//値の評価
		min_val = valuateTimeTableSQL();
		min_off = timetable.countOffence();
		t_timetable = timetable.getTimeTableRowsCopy();
		//現在の時間割の取得
		Iterator i_t_timetable = t_timetable.iterator();
		int task_id, period_id, processor_id;
		int qualification_id = 0;
		while(i_t_timetable.hasNext()) {
			TimeTableRow ttr = (TimeTableRow)i_t_timetable.next();
			task_id = ttr.getTask_id();
			period_id = ttr.getPeriod_id();
			processor_id = ttr.getProcessor_id();
			//(task,period,processor)を取り除く
			deleteTimeTableRow(task_id,period_id,processor_id);
			timetable.deleteTimeTableRow(task_id, period_id, processor_id);
			//task_idの講義種類を調べる
			Lecture l = searchLecture(task_id);
			qualification_id = l.getQualification_id();
			//period_idに担当可能な講師に対して変更を加える
			Iterator i = teachers.iterator();
			while(i.hasNext()){
				Teacher t = (Teacher)i.next();
				//現在の講師と同じ講師でない
				if(t.getProcessor_id()!=processor_id){
					//講師tは講義の種類qualification_idを担当できる
					if(t.isQualification(qualification_id)){
						//講師tはperiod_idで担当できる
						if(t.isPeriod(period_id)){
							//System.out.println("processor_id:"+t.getProcessor_id());
							int processor_id2 = t.getProcessor_id();
							//講師は同じ時間に複数の講義をしない
							if(timetable.isNotTasks(processor_id2,period_id)){
								//(task_id,period_id,processor_id2)の追加
								insertTimeTableRow(task_id,period_id,processor_id2);
								timetable.insertTimeTableRow(task_id,period_id,processor_id2);
								//値の評価
								new_off = timetable.countOffence();
								new_val = valuateTimeTableSQL();
								if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
								{
									//変化した物を入れておく
									new_task_id = task_id;
									new_processor_id = processor_id2;
									del_processor_id = processor_id;
									new_period_id = period_id;
									min_off = new_off;
									min_val = new_val;
									change = true;
								}
								//追加した(task_id,period_id,processor_id2)の削除
								deleteTimeTableRow(task_id,period_id,processor_id2);
								timetable.deleteTimeTableRow(task_id,period_id,processor_id2);
							}
						}
					}
				}
			}
			//取り除いた(task_id,period_id,processor_id)の追加
			insertTimeTableRow(task_id,period_id,processor_id);
			timetable.insertTimeTableRow(task_id,period_id,processor_id);
		}
		//解が改善された時
		if(change){
			//今までの解を変更する
			deleteTimeTableRow(new_task_id,new_period_id,del_processor_id);
			timetable.deleteTimeTableRow(new_task_id,new_period_id,del_processor_id);
			insertTimeTableRow(new_task_id,new_period_id,new_processor_id);
			timetable.insertTimeTableRow(new_task_id,new_period_id,new_processor_id);
			//変更の表示
			if(debug){
				System.out.println("task_id:"+new_task_id+" period_id:"+new_period_id);
				System.out.println("change:processor_id:"+del_processor_id+"がprocessor_id:"+new_processor_id+"に変化");
			}
		}
		return change;
	}

	//(プログラム中のテーブルを初期化しない)	
	//近傍操作Change:講義の担当講師を変更する(メソッド使用)
	//近傍changeを探索し、改善解があればその中で最良のものに移動しtrueを返す。なければfalseを返す。
	public boolean search_change_task_series() throws Exception{
		if(debug)System.out.println("change start");
		//printInsertTimeTable();
		ArrayList t_timetable = null;
		boolean change = false;
		int min_val,min_off,new_val,new_off;
		int del_processor_id = 0,new_task_id = 0,new_task_series_num=1,new_processor_id = 0,new_period_id = 0;
		//値の評価
		min_val = valuateTimeTableSQL();
		min_off = timetable.taskSeriesCountOffence();
		t_timetable = timetable.getTimeTableRowsCopy();
		//現在の時間割の取得
		Iterator i_t_timetable = t_timetable.iterator();
		int task_id, period_id, processor_id,qualification_id,task_series_num;
		while(i_t_timetable.hasNext()) {
			TimeTableRow ttr = (TimeTableRow)i_t_timetable.next();
			task_id = ttr.getTask_id();
			period_id = ttr.getPeriod_id();
			processor_id = ttr.getProcessor_id();
			//task_idの講義種類を調べる
			Lecture l = searchLecture(task_id);
			qualification_id = l.getQualification_id();
			task_series_num = l.getTask_series_num();
			//(task,period,processor)を取り除く
			deleteTimeTableRow(task_id,period_id,processor_id);
			timetable.deleteTimeTableRow(task_id, period_id, processor_id);
			if(task_series_num>1){
				int n_period = period_id;
				for(int num=1;num<task_series_num;num++){
					n_period = day_periods.getNextPeriods(n_period);
					deleteTimeTableRow(task_id,n_period,processor_id);
				}
			}
			//period_idに担当可能な講師に対して変更を加える
			Iterator i = teachers.iterator();
			while(i.hasNext()){
				Teacher t = (Teacher)i.next();
				//現在の講師と同じ講師でない
				if(t.getProcessor_id()!=processor_id){
					//講師tは講義の種類qualification_idを担当できる
					if(t.isQualification(qualification_id)){
						//講師tはperiod_idで担当できる
						if(t.isPeriod(period_id)){
							//System.out.println("processor_id:"+t.getProcessor_id());
							int processor_id2 = t.getProcessor_id();
							//講師は同じ時間に複数の講義をしない
							if(timetable.isNotTasks(processor_id2,period_id)){
								boolean break_flag = false;
								int next_period_id = period_id;
								for(int num=1;num<task_series_num&&!break_flag;num++){
									next_period_id = day_periods.getNextPeriods(next_period_id);
									if(!t.isPeriod(next_period_id)){
										break_flag = true;break;
									}
									if(!timetable.isNotTasks(processor_id2,next_period_id)){
										break_flag = true;break;
									}
								}
								if(break_flag)break;
								//(task_id,period_id,processor_id2)の追加
								insertTimeTableRow(task_id,period_id,processor_id2);
								timetable.insertTimeTableRow(task_id,period_id,processor_id2);
								if(task_series_num>1){
									int n_period = period_id;
									for(int num=1;num<task_series_num;num++){
										n_period = day_periods.getNextPeriods(n_period);
										insertTimeTableRow(task_id,n_period,processor_id2);
									}
								}
								//値の評価
								new_off = timetable.taskSeriesCountOffence();
								new_val = valuateTimeTableSQL();
								if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
								{
									//変化した物を入れておく
									new_task_id = task_id;
									new_task_series_num=task_series_num;
									new_processor_id = processor_id2;
									del_processor_id = processor_id;
									new_period_id = period_id;
									min_off = new_off;
									min_val = new_val;
									change = true;
								}
								//追加した(task_id,period_id,processor_id2)の削除
								deleteTimeTableRow(task_id,period_id,processor_id2);
								timetable.deleteTimeTableRow(task_id,period_id,processor_id2);
								if(task_series_num>1){
									int n_period = period_id;
									for(int num=1;num<task_series_num;num++){
										n_period = day_periods.getNextPeriods(n_period);
										deleteTimeTableRow(task_id,n_period,processor_id2);
									}
								}
							}
						}
					}
				}
			}
			//取り除いた(task_id,period_id,processor_id)の追加
			insertTimeTableRow(task_id,period_id,processor_id);
			timetable.insertTimeTableRow(task_id,period_id,processor_id);
			if(task_series_num>1){
				int n_period = period_id;
				for(int num=1;num<task_series_num;num++){
					n_period = day_periods.getNextPeriods(n_period);
					insertTimeTableRow(task_id,n_period,processor_id);
				}
			}
		}
		//解が改善された時
		if(change){
			//今までの解を変更する
			deleteTimeTableRow(new_task_id,new_period_id,del_processor_id);
			timetable.deleteTimeTableRow(new_task_id,new_period_id,del_processor_id);
			insertTimeTableRow(new_task_id,new_period_id,new_processor_id);
			timetable.insertTimeTableRow(new_task_id,new_period_id,new_processor_id);
			if(new_task_series_num>1){
				int n_period = new_period_id;
				for(int num=1;num<new_task_series_num;num++){
					n_period = day_periods.getNextPeriods(n_period);
					deleteTimeTableRow(new_task_id,n_period,del_processor_id);
					insertTimeTableRow(new_task_id,n_period,new_processor_id);
				}
			}
			//変更の表示
			if(debug){
				System.out.println("task_id:"+new_task_id+" period_id:"+new_period_id);
				System.out.println("change:processor_id:"+del_processor_id+"がprocessor_id:"+new_processor_id+"に変化");
			}
		}
		return change;
	}
	
//(プログラム中のテーブルを初期化しない)	
	//近傍操作Swap:異なる2つの担当講師を入れ替える(メソッド使用)
	//近傍swapを探索し、改善解があればその中で最良のものに移動しtrueを返す。なければfalseを返す。
	public boolean search_swap() throws Exception{
		ArrayList t_timetable = null;
		boolean change = false;
		int min_val,min_off,new_val,new_off;
		int new_task_id1 = 0,new_processor_id1 = 0,new_period_id1 = 0,new_task_id2 = 0,new_processor_id2 = 0,new_period_id2 = 0;
		int next = 0;
		//値の評価
		min_val = valuateTimeTableSQL();
		min_off = timetable.countOffence();
		t_timetable = timetable.getTimeTableRowsCopy();
		//現在の時間割の取得
		Iterator i_t_timetable = t_timetable.iterator();
		while(i_t_timetable.hasNext()) {
			TimeTableRow ttr = (TimeTableRow)i_t_timetable.next();
			int task_id1 = ttr.getTask_id();
			int period_id1 = ttr.getPeriod_id();
			int processor_id1 = ttr.getProcessor_id();
			for(int a = next;a<t_timetable.size();a++){
				TimeTableRow ttr2 = (TimeTableRow)t_timetable.get(a);
				int task_id2 = ttr2.getTask_id();
				int period_id2 = ttr2.getPeriod_id();
				int processor_id2 = ttr2.getProcessor_id();
				int qualification_id1=0,qualification_id2=0;
				//異なる講義かどうか
				if(task_id1!=task_id2){
					//(task_id1,period_id1,processor_id1)と(task_id2,period_id2,processor_id2)を取り除く
					deleteTimeTableRow(task_id1,period_id1,processor_id1);
					deleteTimeTableRow(task_id2,period_id2,processor_id2);
					timetable.deleteTimeTableRow(task_id1,period_id1,processor_id1);
					timetable.deleteTimeTableRow(task_id2,period_id2,processor_id2);
					//task_id1とtask_id2の種類を調べる
					Lecture l1 = searchLecture(task_id1);
					qualification_id1 = l1.getQualification_id();
					Lecture l2 = searchLecture(task_id2);
					qualification_id2 = l2.getQualification_id();
					//(task_id1,period_id1,processor_id2)が有効かどうか
					//processor_id2がperiod_id1に可能かどうか？
					//講師tがprocessor_id2である
					Teacher t = searchTeacher(processor_id2);
					if(t.isQualification(qualification_id1)){
						//processor_id2がperiod_id1に可能
						if(t.isPeriod(period_id1)){
							//講師は同じ時間に複数の講義をしない
							if(timetable.isNotTasks(processor_id2,period_id1)){
								//(task_id2,period_id2,processor_id1)が有効かどうか
								//processor_id1がperiod_id2に担当可能かどうか？
								//講師t2がprocessor_id1である
								Teacher t2 = searchTeacher(processor_id1);
								if(t2.isQualification(qualification_id2)){
									//processor_id1がperiod_id2に可能
									if(t2.isPeriod(period_id2)){
										//講師は同じ時間に複数の講義をしない
										if(timetable.isNotTasks(processor_id1,period_id2)){
											//(task_id1,period_id1,processor_id2)の追加
											insertTimeTableRow(task_id1,period_id1,processor_id2);
											//(task_id2,period_id2,processor_id1)の追加
											insertTimeTableRow(task_id2,period_id2,processor_id1);
											timetable.insertTimeTableRow(task_id1,period_id1,processor_id2);
											timetable.insertTimeTableRow(task_id2,period_id2,processor_id1);
											new_off = timetable.countOffence();
											new_val = valuateTimeTableSQL();
											if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
											{
												//変化した物を入れておく
												new_period_id1=period_id1;
												new_period_id2=period_id2;
												new_processor_id1=processor_id1;
												new_processor_id2=processor_id2;
												new_task_id1=task_id1;
												new_task_id2=task_id2;
												min_off = new_off;
												min_val = new_val;
												change = true;
											}
											//追加した(task_id1,period_id1,processor_id2)の削除
											deleteTimeTableRow(task_id1,period_id1,processor_id2);
											//追加した(task_id2,period_id2,processor_id1)の削除
											deleteTimeTableRow(task_id2,period_id2,processor_id1);
											timetable.deleteTimeTableRow(task_id1,period_id1,processor_id2);
											timetable.deleteTimeTableRow(task_id2,period_id2,processor_id1);
											}
									}
								}
							}
						}
					}
					//取り除いた(task_id1,period_id1,processor_id1)と(task_id2,period_id2,processor_id2)を追加する
					insertTimeTableRow(task_id1,period_id1,processor_id1);
					insertTimeTableRow(task_id2,period_id2,processor_id2);
					timetable.insertTimeTableRow(task_id1,period_id1,processor_id1);
					timetable.insertTimeTableRow(task_id2,period_id2,processor_id2);
				}
			}
			next++;
		}
		//解が改善された時
		if(change){
			updateProcessorTimeTable(new_processor_id2,new_processor_id1,new_task_id1,new_period_id1);
			timetable.updateProcessorTimeTable(new_processor_id2,new_processor_id1,new_task_id1,new_period_id1);
			updateProcessorTimeTable(new_processor_id1,new_processor_id2,new_task_id2,new_period_id2);
			timetable.updateProcessorTimeTable(new_processor_id1,new_processor_id2,new_task_id2,new_period_id2);
			//変更の表示
			if(debug){
				System.out.println("(task_id,processor_id,period_id)");
				System.out.println("swap:("+new_task_id1+","+new_processor_id1+","+new_period_id1+")と("+new_task_id2+","+new_processor_id2+","+new_period_id2+")");
			}
			
		}
		return change;
	}

	//(プログラム中のテーブルを初期化しない)	
	//近傍操作Swap:異なる2つの担当講師を入れ替える(メソッド使用)
	//近傍swapを探索し、改善解があればその中で最良のものに移動しtrueを返す。なければfalseを返す。
	public boolean search_swap_task_series() throws Exception{
		if(debug)System.out.println("swap start");
		//printInsertTimeTable();
		ArrayList t_timetable = null;
		boolean change = false;
		int min_val,min_off,new_val,new_off;
		int new_task_id1 = 0,new_processor_id1 = 0,new_period_id1 = 0,new_task_id2 = 0,new_processor_id2 = 0,new_period_id2 = 0,new_task_series_num1=1,new_task_series_num2=1;
		int next = 0;
		//値の評価
		min_val = valuateTimeTableSQL();
		min_off = timetable.taskSeriesCountOffence();
		t_timetable = timetable.getTimeTableRowsCopy();
		//現在の時間割の取得
		Iterator i_t_timetable = t_timetable.iterator();
		while(i_t_timetable.hasNext()) {
			TimeTableRow ttr = (TimeTableRow)i_t_timetable.next();
			int task_id1 = ttr.getTask_id();
			int period_id1 = ttr.getPeriod_id();
			int processor_id1 = ttr.getProcessor_id();
			for(int a = next;a<t_timetable.size();a++){
				TimeTableRow ttr2 = (TimeTableRow)t_timetable.get(a);
				int task_id2 = ttr2.getTask_id();
				int period_id2 = ttr2.getPeriod_id();
				int processor_id2 = ttr2.getProcessor_id();
				int qualification_id1,qualification_id2,task_series_num1,task_series_num2;
				//異なる講義かどうか
				if(task_id1!=task_id2){
					//task_id1とtask_id2の種類を調べる
					Lecture l1 = searchLecture(task_id1);
					qualification_id1 = l1.getQualification_id();
					task_series_num1 = l1.getTask_series_num();
					Lecture l2 = searchLecture(task_id2);
					qualification_id2 = l2.getQualification_id();
					task_series_num2 = l2.getTask_series_num();
					//(task_id1,period_id1,processor_id1)と(task_id2,period_id2,processor_id2)を取り除く
					deleteTimeTableRow(task_id1,period_id1,processor_id1);
					deleteTimeTableRow(task_id2,period_id2,processor_id2);
					timetable.deleteTimeTableRow(task_id1,period_id1,processor_id1);
					timetable.deleteTimeTableRow(task_id2,period_id2,processor_id2);
					if(task_series_num1>1){
						int n_period = period_id1;
						for(int num=1;num<task_series_num1;num++){
							n_period = day_periods.getNextPeriods(n_period);
							deleteTimeTableRow(task_id1,n_period,processor_id1);
							//System.out.println("delete task_id1:"+task_id1);
						}
					}
					if(task_series_num2>1){
						int n_period = period_id2;
						for(int num=1;num<task_series_num2;num++){
							n_period = day_periods.getNextPeriods(n_period);
							deleteTimeTableRow(task_id2,n_period,processor_id2);
							//System.out.println("delete task_id2:"+task_id2);
						}
					}
					//(task_id1,period_id1,processor_id2)が有効かどうか
					//processor_id2がperiod_id1に可能かどうか？
					//講師tがprocessor_id2である
					Teacher t = searchTeacher(processor_id2);
					if(t.isQualification(qualification_id1)){
						//processor_id2がperiod_id1に可能
						if(t.isPeriod(period_id1)){
							//講師は同じ時間に複数の講義をしない
							if(timetable.isNotTasks(processor_id2,period_id1)){
								boolean break_flag = false;
								int next_period_id = period_id1;
								for(int num=1;num<task_series_num1&&!break_flag;num++){
									next_period_id = day_periods.getNextPeriods(next_period_id);
									if(!t.isPeriod(next_period_id)){
										break_flag = true;break;
									}
									if(!timetable.isNotTasks(processor_id2,next_period_id)){
										break_flag = true;break;
									}
								}
								if(break_flag){
									//取り除いた(task_id1,period_id1,processor_id1)と(task_id2,period_id2,processor_id2)を追加する
									insertTimeTableRow(task_id1,period_id1,processor_id1);
									insertTimeTableRow(task_id2,period_id2,processor_id2);
									timetable.insertTimeTableRow(task_id1,period_id1,processor_id1);
									timetable.insertTimeTableRow(task_id2,period_id2,processor_id2);
									if(task_series_num1>1){
										int n_period = period_id1;
										for(int num=1;num<task_series_num1;num++){
											n_period = day_periods.getNextPeriods(n_period);
											insertTimeTableRow(task_id1,n_period,processor_id1);
											//System.out.println("insert task_id1:"+task_id1);
										}
									}
									if(task_series_num2>1){
										int n_period = period_id2;
										for(int num=1;num<task_series_num2;num++){
											n_period = day_periods.getNextPeriods(n_period);
											insertTimeTableRow(task_id2,n_period,processor_id2);
											//System.out.println("insert task_id2:"+task_id2);
										}
									}
									break;
								}
								//(task_id2,period_id2,processor_id1)が有効かどうか
								//processor_id1がperiod_id2に担当可能かどうか？
								//講師t2がprocessor_id1である
								Teacher t2 = searchTeacher(processor_id1);
								if(t2.isQualification(qualification_id2)){
									//processor_id1がperiod_id2に可能
									if(t2.isPeriod(period_id2)){
										//講師は同じ時間に複数の講義をしない
										if(timetable.isNotTasks(processor_id1,period_id2)){
											break_flag = false;
											next_period_id = period_id2;
											for(int num=1;num<task_series_num2&&!break_flag;num++){
												next_period_id = day_periods.getNextPeriods(next_period_id);
												if(!t.isPeriod(next_period_id)){
													break_flag = true;break;
												}
												if(!timetable.isNotTasks(processor_id1,next_period_id)){
													break_flag = true;break;
												}
											}
											if(break_flag){
												//取り除いた(task_id1,period_id1,processor_id1)と(task_id2,period_id2,processor_id2)を追加する
												insertTimeTableRow(task_id1,period_id1,processor_id1);
												insertTimeTableRow(task_id2,period_id2,processor_id2);
												timetable.insertTimeTableRow(task_id1,period_id1,processor_id1);
												timetable.insertTimeTableRow(task_id2,period_id2,processor_id2);
												if(task_series_num1>1){
													int n_period = period_id1;
													for(int num=1;num<task_series_num1;num++){
														n_period = day_periods.getNextPeriods(n_period);
														insertTimeTableRow(task_id1,n_period,processor_id1);
														//System.out.println("insert task_id1:"+task_id1);
													}
												}
												if(task_series_num2>1){
													int n_period = period_id2;
													for(int num=1;num<task_series_num2;num++){
														n_period = day_periods.getNextPeriods(n_period);
														insertTimeTableRow(task_id2,n_period,processor_id2);
														//System.out.println("insert task_id2:"+task_id2);
													}
												}
												break;
											}
											//(task_id1,period_id1,processor_id2)の追加
											insertTimeTableRow(task_id1,period_id1,processor_id2);
											//(task_id2,period_id2,processor_id1)の追加
											insertTimeTableRow(task_id2,period_id2,processor_id1);
											timetable.insertTimeTableRow(task_id1,period_id1,processor_id2);
											timetable.insertTimeTableRow(task_id2,period_id2,processor_id1);
											if(task_series_num1>1){
												int n_period = period_id1;
												for(int num=1;num<task_series_num1;num++){
													n_period = day_periods.getNextPeriods(n_period);
													insertTimeTableRow(task_id1,n_period,processor_id2);
												}
											}
											if(task_series_num2>1){
												int n_period = period_id2;
												for(int num=1;num<task_series_num2;num++){
													n_period = day_periods.getNextPeriods(n_period);
													insertTimeTableRow(task_id2,n_period,processor_id1);
												}
											}
											new_off = timetable.taskSeriesCountOffence();
											new_val = valuateTimeTableSQL();
											if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
											{
												//変化した物を入れておく
												new_period_id1=period_id1;
												new_period_id2=period_id2;
												new_processor_id1=processor_id1;
												new_processor_id2=processor_id2;
												new_task_series_num1=task_series_num1;
												new_task_series_num2=task_series_num2;
												new_task_id1=task_id1;
												new_task_id2=task_id2;
												min_off = new_off;
												min_val = new_val;
												change = true;
											}
											//追加した(task_id1,period_id1,processor_id2)の削除
											deleteTimeTableRow(task_id1,period_id1,processor_id2);
											//追加した(task_id2,period_id2,processor_id1)の削除
											deleteTimeTableRow(task_id2,period_id2,processor_id1);
											timetable.deleteTimeTableRow(task_id1,period_id1,processor_id2);
											timetable.deleteTimeTableRow(task_id2,period_id2,processor_id1);
											if(task_series_num1>1){
												int n_period = period_id1;
												for(int num=1;num<task_series_num1;num++){
													n_period = day_periods.getNextPeriods(n_period);
													deleteTimeTableRow(task_id1,n_period,processor_id2);
												}
											}
											if(task_series_num2>1){
												int n_period = period_id2;
												for(int num=1;num<task_series_num2;num++){
													n_period = day_periods.getNextPeriods(n_period);
													deleteTimeTableRow(task_id2,n_period,processor_id1);
												}
											}
										}
									}
								}
							}
						}
					}
					//取り除いた(task_id1,period_id1,processor_id1)と(task_id2,period_id2,processor_id2)を追加する
					insertTimeTableRow(task_id1,period_id1,processor_id1);
					insertTimeTableRow(task_id2,period_id2,processor_id2);
					timetable.insertTimeTableRow(task_id1,period_id1,processor_id1);
					timetable.insertTimeTableRow(task_id2,period_id2,processor_id2);
					if(task_series_num1>1){
						int n_period = period_id1;
						for(int num=1;num<task_series_num1;num++){
							n_period = day_periods.getNextPeriods(n_period);
							insertTimeTableRow(task_id1,n_period,processor_id1);
							//System.out.println("insert task_id1:"+task_id1);
						}
					}
					if(task_series_num2>1){
						int n_period = period_id2;
						for(int num=1;num<task_series_num2;num++){
							n_period = day_periods.getNextPeriods(n_period);
							insertTimeTableRow(task_id2,n_period,processor_id2);
							//System.out.println("insert task_id2:"+task_id2);
						}
					}
				}
			}
			next++;
		}
		//解が改善された時
		if(change){
			updateProcessorTimeTable(new_processor_id2,new_processor_id1,new_task_id1,new_period_id1);
			timetable.updateProcessorTimeTable(new_processor_id2,new_processor_id1,new_task_id1,new_period_id1);
			updateProcessorTimeTable(new_processor_id1,new_processor_id2,new_task_id2,new_period_id2);
			timetable.updateProcessorTimeTable(new_processor_id1,new_processor_id2,new_task_id2,new_period_id2);
			if(new_task_series_num1>1){
				int n_period = new_period_id1;
				for(int num=1;num<new_task_series_num1;num++){
					n_period = day_periods.getNextPeriods(n_period);
					updateProcessorTimeTable(new_processor_id2,new_processor_id1,new_task_id1,n_period);
				}
			}
			if(new_task_series_num2>1){
				int n_period = new_period_id2;
				for(int num=1;num<new_task_series_num2;num++){
					n_period = day_periods.getNextPeriods(n_period);
					updateProcessorTimeTable(new_processor_id1,new_processor_id2,new_task_id2,n_period);
				}
			}
			//変更の表示
			if(debug){
				System.out.println("(task_id,processor_id,period_id)");
				System.out.println("swap:("+new_task_id1+","+new_processor_id1+","+new_period_id1+")と("+new_task_id2+","+new_processor_id2+","+new_period_id2+")");
			}
			if(debug)printTimeTable();
		}
		return change;
	}
	
	//(プログラム中のテーブルを初期化しない)	
	//近傍操作add:講義の担当講師を追加する(メソッド使用)
	//近傍addを探索し、改善解があればその中で最良のものに移動しtrueを返す。なければfalseを返す。
	public boolean search_add() throws Exception{
		boolean change = false;
		ArrayList new_teacher_ids = null;
		int min_val,min_off,new_val,new_off;
		int new_task_id = 0,new_period_id = 0;
		//値の評価
		min_val = valuateTimeTableSQL();
		min_off = timetable.countOffence();
		int task_id;
		int qualification_id = 0;
		int period_id;
		int num;
		int low;
		//現在の時間割の講義とその担当人数と開講時間を取得
		ArrayList list = timetable.getTimeTableTasksNum();
		Iterator i_list = list.iterator();
		while(i_list.hasNext()){
			TaskPeriodNum tpn = (TaskPeriodNum)i_list.next();
			task_id = tpn.getTask_id();
			period_id = tpn.getPeriod_id();
			num = tpn.getNum();
			//task_idである講義の取得
			Lecture l = searchLecture(task_id);
			qualification_id = l.getQualification_id();
			//period_idに担当可能な講師のidを入れておく
			ArrayList t_teachers2 = null;
			//task_idの担当人数上限に足りない人数の取得
			low = l.getRequired_processors_ub()-num;
			//System.out.println("low:"+low);
			//もし講義の担当人数上限に足りていないならその講義を担当できる全講師のprocessor_idを取得
			if(low>0){
				t_teachers2 = new ArrayList();
				Iterator i3 = teachers.iterator();
				while(i3.hasNext()){
					Teacher t = (Teacher)i3.next();
					if(t.isQualification(qualification_id)){
						//tがperiod_idで担当可能
						if(t.isPeriod(period_id)){
							int processor_id = t.getProcessor_id();
							//講師は同じ時間に複数の講義をしない
							if(timetable.isNotTasks(processor_id,period_id)){
								t_teachers2.add(new Integer(processor_id));	
								//System.out.println("追加可能:"+processor_id);
							}
						}
					}
				}
			}
			//担当人数の増加
			for(int j = 0;j<low;j++){
				//System.out.println("task_id:"+task_id);
				ArrayList t_teachers = null;
				Enumeration e = new CombEnum(t_teachers2.toArray(), j+1);
				//次の追加する講師の組み合わせはあるか
				while(e.hasMoreElements()) {
					t_teachers = new ArrayList();
					Object[] a = (Object[])e.nextElement();
					//System.out.println("追加する人数:"+(j+1));
					for(int num2=0; num2<a.length; num2++){
						int processor_id = ((Integer)a[num2]).intValue();
						//講師の追加
						insertTimeTableRow(task_id,period_id,processor_id);
						timetable.insertTimeTableRow(task_id,period_id,processor_id);
						t_teachers.add(new Integer(processor_id));
					}
					//値の評価
					new_off = timetable.countOffence();
					new_val = valuateTimeTableSQL();
					if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
					{
						//変化した物を入れておく
						new_teacher_ids = t_teachers;
						new_task_id = task_id;
						new_period_id = period_id;
						min_off = new_off;
						min_val = new_val;
						change = true;
					}
					//追加した講師の削除
					Iterator i5 = t_teachers.iterator();
					while(i5.hasNext()){
						Integer t = (Integer)i5.next();
						deleteTimeTableRow(task_id,period_id,t.intValue());
						timetable.deleteTimeTableRow(task_id,period_id,t.intValue());
					}
				}
			}
		}
		//解が改善された時
		if(change){
			//変更の表示
			if(debug) System.out.println("(task_id,processor_id,period_id)");
			Iterator i = new_teacher_ids.iterator();
			int processor_id;
			while(i.hasNext()){
				processor_id = ((Integer)i.next()).intValue();
				if(debug) System.out.println("add:("+new_task_id+","+processor_id+","+new_period_id+")");
				insertTimeTableRow(new_task_id,new_period_id,processor_id);
				timetable.insertTimeTableRow(new_task_id,new_period_id,processor_id);
			}
		}
		return change;		
	}

	//(プログラム中のテーブルを初期化しない)	
	//近傍操作add:講義の担当講師を追加する(メソッド使用)
	//近傍addを探索し、改善解があればその中で最良のものに移動しtrueを返す。なければfalseを返す。
	public boolean search_add_task_series() throws Exception{
		if(debug)System.out.println("add start");
		//printInsertTimeTable();
		boolean change = false;
		ArrayList new_teacher_ids = null;
		int min_val,min_off,new_val,new_off;
		int new_task_id = 0,new_period_id = 0,new_task_series_num=1;
		//値の評価
		min_val = valuateTimeTableSQL();
		min_off = timetable.taskSeriesCountOffence();
		int task_id,qualification_id,period_id,task_num,low,task_series_num;
		//現在の時間割の講義とその担当人数と開講時間を取得
		ArrayList list = timetable.getTimeTableTasksNum();
		Iterator i_list = list.iterator();
		while(i_list.hasNext()){
			TaskPeriodNum tpn = (TaskPeriodNum)i_list.next();
			task_id = tpn.getTask_id();
			period_id = tpn.getPeriod_id();
			task_num = tpn.getNum();
			//task_idである講義の取得
			Lecture l = searchLecture(task_id);
			qualification_id = l.getQualification_id();
			task_series_num = l.getTask_series_num();
			//period_idに担当可能な講師のidを入れておく
			ArrayList t_teachers2 = null;
			//task_idの担当人数上限に足りない人数の取得
			low = l.getRequired_processors_ub()-task_num;
			//System.out.println("low:"+low);
			//もし講義の担当人数上限に足りていないならその講義を担当できる全講師のprocessor_idを取得
			if(low>0){
				t_teachers2 = new ArrayList();
				Iterator i3 = teachers.iterator();
				while(i3.hasNext()){
					Teacher t = (Teacher)i3.next();
					if(t.isQualification(qualification_id)){
						//tがperiod_idで担当可能
						if(t.isPeriod(period_id)){
							int processor_id = t.getProcessor_id();
							//講師は同じ時間に複数の講義をしない
							if(timetable.isNotTasks(processor_id,period_id)){
								int next_period_id=period_id;
								boolean break_flag = false;
								for(int num=1;num<task_series_num&&!break_flag;num++){
									next_period_id = day_periods.getNextPeriods(next_period_id);
									if(!t.isPeriod(next_period_id)){
										break_flag = true;break;
									}
									if(!timetable.isNotTasks(processor_id,next_period_id)){
										break_flag = true;break;
									}
								}
								if(break_flag)break;
								t_teachers2.add(new Integer(processor_id));	
								//System.out.println("追加可能:"+processor_id);
							}
						}
					}
				}
			}
			//担当人数の増加
			for(int j = 0;j<low;j++){
				//System.out.println("task_id:"+task_id);
				ArrayList t_teachers = null;
				Enumeration e = new CombEnum(t_teachers2.toArray(), j+1);
				//次の追加する講師の組み合わせはあるか
				while(e.hasMoreElements()) {
					t_teachers = new ArrayList();
					Object[] a = (Object[])e.nextElement();
					//System.out.println("追加する人数:"+(j+1));
					for(int num2=0; num2<a.length; num2++){
						int processor_id = ((Integer)a[num2]).intValue();
						//講師の追加
						insertTimeTableRow(task_id,period_id,processor_id);
						timetable.insertTimeTableRow(task_id,period_id,processor_id);
						if(task_series_num>1){
							int n_period = period_id;
							for(int num=1;num<task_series_num;num++){
								n_period = day_periods.getNextPeriods(n_period);
								insertTimeTableRow(task_id,n_period,processor_id);
							}
						}
						t_teachers.add(new Integer(processor_id));
					}
					//値の評価
					new_off = timetable.taskSeriesCountOffence();
					new_val = valuateTimeTableSQL();
					if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
					{
						//変化した物を入れておく
						new_task_series_num=task_series_num;
						new_teacher_ids = t_teachers;
						new_task_id = task_id;
						new_period_id = period_id;
						min_off = new_off;
						min_val = new_val;
						change = true;
					}
					//追加した講師の削除
					Iterator i5 = t_teachers.iterator();
					while(i5.hasNext()){
						Integer t = (Integer)i5.next();
						deleteTimeTableRow(task_id,period_id,t.intValue());
						timetable.deleteTimeTableRow(task_id,period_id,t.intValue());
						if(task_series_num>1){
							int n_period = period_id;
							for(int num=1;num<task_series_num;num++){
								n_period = day_periods.getNextPeriods(n_period);
								deleteTimeTableRow(task_id,n_period,t.intValue());
							}
						}
					}
				}
			}
		}
		//解が改善された時
		if(change){
			//変更の表示
			if(debug) System.out.println("(task_id,processor_id,period_id)");
			Iterator i = new_teacher_ids.iterator();
			int processor_id;
			while(i.hasNext()){
				processor_id = ((Integer)i.next()).intValue();
				if(debug) System.out.println("add:("+new_task_id+","+processor_id+","+new_period_id+")");
				insertTimeTableRow(new_task_id,new_period_id,processor_id);
				timetable.insertTimeTableRow(new_task_id,new_period_id,processor_id);
				if(new_task_series_num>1){
					int n_period = new_period_id;
					for(int num=1;num<new_task_series_num;num++){
						n_period = day_periods.getNextPeriods(n_period);
						insertTimeTableRow(new_task_id,n_period,processor_id);
					}
				}
			}
			if(debug)printTimeTable();
		}
		return change;		
	}

	//(プログラム中のテーブルを初期化しない)
	//近傍操作add:講義が開講されてない時追加(メソッド使用)
	//近傍addを探索し、改善解があればその中で最良のものに移動しtrueを返す。なければfalseを返す。
	public boolean search_add2() throws Exception{
		boolean change = false;
		ArrayList new_teacher_ids = null;
		int min_val,min_off,new_val,new_off;
		int new_task_id = 0,new_period_id = 0;
		//値の評価
		min_val = valuateTimeTableSQL();
		min_off = timetable.countOffence();
		int task_id,qualification_id,period_id,num,low;
		Iterator i_list = lectures.iterator();
		while(i_list.hasNext()){
			Lecture l = (Lecture)i_list.next();
			num = timetable.getTimeTableTasks(l.getTask_id());
			if(num==0){
				task_id = l.getTask_id();
				qualification_id = l.getQualification_id();
				//task_idの担当人数上限に足りない人数の取得
				low = l.getRequired_processors_ub()-num;
				//task_idの開講可能時間の取得
				Iterator i = l.getPeriods().iterator();
				while(i.hasNext()){
					PeriodDesire lpd = (PeriodDesire)i.next();
					period_id = lpd.getPeriod_id();
					//period_idに担当可能な講師のidを入れておく
					ArrayList t_teachers2 = new ArrayList();
					Iterator i3 = teachers.iterator();
					while(i3.hasNext()){
						Teacher t = (Teacher)i3.next();
						if(t.isQualification(qualification_id)){
							//tがperiod_idで担当可能
							if(t.isPeriod(period_id)){
								int processor_id = t.getProcessor_id();
								//講師は同じ時間に複数の講義をしない
								if(timetable.isNotTasks(processor_id,period_id)){
									t_teachers2.add(new Integer(processor_id));	
									//System.out.println("追加可能:"+processor_id);
								}
							}
						}
					}
					//担当人数の増加
					for(int j = 0;j<low;j++){
						//System.out.println("task_id:"+task_id);
						ArrayList t_teachers = null;
						Enumeration e = new CombEnum(t_teachers2.toArray(), j+1);
						//次の追加する講師の組み合わせはあるか
						while(e.hasMoreElements()) {
							t_teachers = new ArrayList();
							Object[] a = (Object[])e.nextElement();
							//System.out.println("追加する人数:"+(j+1));
							for(int num2=0; num2<a.length; num2++){
								int processor_id = ((Integer)a[num2]).intValue();
								//講師の追加
								insertTimeTableRow(task_id,period_id,processor_id);
								timetable.insertTimeTableRow(task_id,period_id,processor_id);
								t_teachers.add(new Integer(processor_id));
							}
							//値の評価
							new_off = timetable.countOffence();
							new_val = valuateTimeTableSQL();
							if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
							{
								//変化した物を入れておく
								new_teacher_ids = t_teachers;
								new_task_id = task_id;
								new_period_id = period_id;
								min_off = new_off;
								min_val = new_val;
								change = true;
							}
							//追加した講師の削除
							Iterator i5 = t_teachers.iterator();
							while(i5.hasNext()){
								Integer t = (Integer)i5.next();
								deleteTimeTableRow(task_id,period_id,t.intValue());
								timetable.deleteTimeTableRow(task_id,period_id,t.intValue());
							}
						}
					}
				}
			}
		}
		//解が改善された時
		if(change){
			//変更の表示
			if(debug) System.out.println("(task_id,processor_id,period_id)");
			Iterator i = new_teacher_ids.iterator();
			int processor_id;
			while(i.hasNext()){
				processor_id = ((Integer)i.next()).intValue();
				if(debug) System.out.println("add:("+new_task_id+","+processor_id+","+new_period_id+")");
				insertTimeTableRow(new_task_id,new_period_id,processor_id);
				timetable.insertTimeTableRow(new_task_id,new_period_id,processor_id);
			}
		}
		return change;		
	}

	//(プログラム中のテーブルを初期化しない)
	//近傍操作add:講義が開講されてない時追加(メソッド使用)
	//近傍addを探索し、改善解があればその中で最良のものに移動しtrueを返す。なければfalseを返す。
	public boolean search_add2_task_series() throws Exception{
		boolean change = false;
		ArrayList new_teacher_ids = null;
		int min_val,min_off,new_val,new_off;
		int new_task_id = 0,new_period_id = 0,new_task_series_num=1;
		//値の評価
		min_val = valuateTimeTableSQL();
		min_off = timetable.taskSeriesCountOffence();
		int task_id,qualification_id,period_id,task_num,low,task_series_num;
		Iterator i_list = lectures.iterator();
		while(i_list.hasNext()){
			Lecture l = (Lecture)i_list.next();
			task_num = timetable.getTimeTableTasks(l.getTask_id());
			if(task_num==0){
				task_id = l.getTask_id();
				qualification_id = l.getQualification_id();
				task_series_num = l.getTask_series_num();
				//task_idの担当人数上限に足りない人数の取得
				low = l.getRequired_processors_ub()-task_num;
				//task_idの開講可能時間の取得
				Iterator i = l.getPeriods().iterator();
				while(i.hasNext()){
					PeriodDesire lpd = (PeriodDesire)i.next();
					period_id = lpd.getPeriod_id();
					//period_idに担当可能な講師のidを入れておく
					ArrayList t_teachers2 = new ArrayList();
					Iterator i3 = teachers.iterator();
					while(i3.hasNext()){
						Teacher t = (Teacher)i3.next();
						if(t.isQualification(qualification_id)){
							//tがperiod_idで担当可能
							if(t.isPeriod(period_id)){
								int processor_id = t.getProcessor_id();
								//講師は同じ時間に複数の講義をしない
								if(timetable.isNotTasks(processor_id,period_id)){
									int next_period_id=period_id;
									boolean break_flag = false;
									for(int num=1;num<task_series_num&&!break_flag;num++){
										next_period_id = day_periods.getNextPeriods(next_period_id);
										if(!t.isPeriod(next_period_id)){
											break_flag = true;break;
										}
										if(!timetable.isNotTasks(processor_id,next_period_id)){
											break_flag = true;break;
										}
									}
									if(break_flag)break;
									t_teachers2.add(new Integer(processor_id));	
									//System.out.println("追加可能:"+processor_id);
								}
							}
						}
					}
					//担当人数の増加
					for(int j = 0;j<low;j++){
						//System.out.println("task_id:"+task_id);
						ArrayList t_teachers = null;
						Enumeration e = new CombEnum(t_teachers2.toArray(), j+1);
						//次の追加する講師の組み合わせはあるか
						while(e.hasMoreElements()) {
							t_teachers = new ArrayList();
							Object[] a = (Object[])e.nextElement();
							//System.out.println("追加する人数:"+(j+1));
							for(int num2=0; num2<a.length; num2++){
								int processor_id = ((Integer)a[num2]).intValue();
								//講師の追加
								insertTimeTableRow(task_id,period_id,processor_id);
								timetable.insertTimeTableRow(task_id,period_id,processor_id);
								if(task_series_num>1){
									int n_period = period_id;
									for(int num=1;num<task_series_num;num++){
										n_period = day_periods.getNextPeriods(n_period);
										insertTimeTableRow(task_id,n_period,processor_id);
									}
								}
								t_teachers.add(new Integer(processor_id));
							}
							//値の評価
							new_off = timetable.taskSeriesCountOffence();
							new_val = valuateTimeTableSQL();
							if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
							{
								//変化した物を入れておく
								new_task_series_num=task_series_num;
								new_teacher_ids = t_teachers;
								new_task_id = task_id;
								new_period_id = period_id;
								min_off = new_off;
								min_val = new_val;
								change = true;
							}
							//追加した講師の削除
							Iterator i5 = t_teachers.iterator();
							while(i5.hasNext()){
								Integer t = (Integer)i5.next();
								deleteTimeTableRow(task_id,period_id,t.intValue());
								timetable.deleteTimeTableRow(task_id,period_id,t.intValue());
								if(task_series_num>1){
									int n_period = period_id;
									for(int num=1;num<task_series_num;num++){
										n_period = day_periods.getNextPeriods(n_period);
										deleteTimeTableRow(task_id,n_period,t.intValue());
									}
								}
							}
						}
					}
				}
			}
		}
		//解が改善された時
		if(change){
			//変更の表示
			if(debug) System.out.println("(task_id,processor_id,period_id)");
			Iterator i = new_teacher_ids.iterator();
			int processor_id;
			while(i.hasNext()){
				processor_id = ((Integer)i.next()).intValue();
				if(debug) System.out.println("add:("+new_task_id+","+processor_id+","+new_period_id+")");
				insertTimeTableRow(new_task_id,new_period_id,processor_id);
				timetable.insertTimeTableRow(new_task_id,new_period_id,processor_id);
				if(new_task_series_num>1){
					int n_period = new_period_id;
					for(int num=1;num<new_task_series_num;num++){
						n_period = day_periods.getNextPeriods(n_period);
						insertTimeTableRow(new_task_id,n_period,processor_id);
					}
				}
			}
		}
		return change;		
	}
	//(プログラム中のテーブルを初期化しない)	
	//近傍操作leave:講義の担当講師を減らす(メソッド使用)
	//近傍leaveを探索し、改善解があればその中で最良のものに移動しtrueを返す。なければfalseを返す。
	public boolean search_leave() throws Exception{
		boolean change = false;
		ArrayList new_teacher_ids = null;
		int min_val,min_off,new_val,new_off;
		int new_task_id = 0,new_period_id = 0;
		//値の評価
		min_val = valuateTimeTableSQL();
		min_off = timetable.countOffence();
		//現在の時間割の講義とその担当人数と開講時間を取得
		ArrayList list = timetable.getTimeTableTasksNum();
		Iterator i_list = list.iterator();
		int task_id;
		int period_id;
		int num;
		while(i_list.hasNext()){
			TaskPeriodNum tpn = (TaskPeriodNum)i_list.next();
			task_id = tpn.getTask_id();
			period_id = tpn.getPeriod_id();
			num = tpn.getNum();
			//task_idである講義の取得
			Lecture l = searchLecture(task_id);
			//task_idの担当人数下限以上の人数の取得
			int up = num-l.getRequired_processors_lb();
			ArrayList t_teachers2 = null;
			//もし講義の担当人数下限以上ならばその講義を担当している全講師のprocessor_idを取得
			if(up > 0){
				//period_idにtask_idを担当する講師を取得
				t_teachers2 = timetable.getProcessors(task_id,period_id);
			}
			//担当人数の減少
			for(int j=0;j<up;j++){
				//削除したprocessor_idを保存する
				ArrayList t_teachers = null;
				Enumeration e = new CombEnum(t_teachers2.toArray(), j+1);
				//次の削除する講師の組み合わせはあるか
				while(e.hasMoreElements()) {
					t_teachers = new ArrayList();
					Object[] a = (Object[])e.nextElement();
					//System.out.println("削除する人数:"+(j+1));
					for(int num2=0; num2<a.length; num2++){
						int processor_id = ((Integer)a[num2]).intValue();
						//講師の削除
						deleteTimeTableRow(task_id,period_id,processor_id);
						timetable.deleteTimeTableRow(task_id,period_id,processor_id);
						t_teachers.add(new Integer(processor_id));
					}
					//値の評価
					new_off = timetable.countOffence();
					new_val = valuateTimeTableSQL();
					if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
					{
						//変化した物を入れておく
						new_teacher_ids = t_teachers;
						new_task_id = task_id;
						new_period_id = period_id;
						min_off = new_off;
						min_val = new_val;
						change = true;
					}
					//削除した講師を追加
					Iterator i5 = t_teachers.iterator();
					while(i5.hasNext()){
						int processor_id = ((Integer)i5.next()).intValue();
						insertTimeTableRow(task_id,period_id,processor_id);
						timetable.insertTimeTableRow(task_id,period_id,processor_id);
					}
				}
			}
		}
		//解が改善された時
		if(change){
			//変更の表示
			if(debug) System.out.println("(task_id,processor_id,period_id)");
			Iterator i = new_teacher_ids.iterator();
			int processor_id;
			while(i.hasNext()){
				processor_id = ((Integer)i.next()).intValue();
				if(debug) System.out.println("leave:("+new_task_id+","+processor_id+","+new_period_id+")");
				deleteTimeTableRow(new_task_id,new_period_id,processor_id);
				timetable.deleteTimeTableRow(new_task_id,new_period_id,processor_id);
			}
		}
		return change;		
	}

	//(プログラム中のテーブルを初期化しない)	
	//近傍操作leave:講義の担当講師を減らす(メソッド使用)
	//近傍leaveを探索し、改善解があればその中で最良のものに移動しtrueを返す。なければfalseを返す。
	public boolean search_leave_task_series() throws Exception{
		if(debug)System.out.println("leave start");
		//printInsertTimeTable();
		boolean change = false;
		ArrayList new_teacher_ids = null;
		int min_val,min_off,new_val,new_off;
		int new_task_id = 0,new_period_id = 0,new_task_series_num=1;
		//値の評価
		min_val = valuateTimeTableSQL();
		min_off = timetable.taskSeriesCountOffence();
		//現在の時間割の講義とその担当人数と開講時間を取得
		ArrayList list = timetable.getTimeTableTasksNum();
		Iterator i_list = list.iterator();
		int task_id,period_id,task_num,task_series_num;
		while(i_list.hasNext()){
			TaskPeriodNum tpn = (TaskPeriodNum)i_list.next();
			task_id = tpn.getTask_id();
			period_id = tpn.getPeriod_id();
			task_num = tpn.getNum();
			//task_idである講義の取得
			Lecture l = searchLecture(task_id);
			//task_idの担当人数下限以上の人数の取得
			int up = task_num-l.getRequired_processors_lb();
			task_series_num=l.getTask_series_num();
			ArrayList t_teachers2 = null;
			//もし講義の担当人数下限以上ならばその講義を担当している全講師のprocessor_idを取得
			if(up > 0){
				//period_idにtask_idを担当する講師を取得
				t_teachers2 = timetable.getProcessors(task_id,period_id);
			}
			//担当人数の減少
			for(int j=0;j<up;j++){
				//削除したprocessor_idを保存する
				ArrayList t_teachers = null;
				Enumeration e = new CombEnum(t_teachers2.toArray(), j+1);
				//次の削除する講師の組み合わせはあるか
				while(e.hasMoreElements()) {
					t_teachers = new ArrayList();
					Object[] a = (Object[])e.nextElement();
					//System.out.println("削除する人数:"+(j+1));
					for(int num2=0; num2<a.length; num2++){
						int processor_id = ((Integer)a[num2]).intValue();
						//講師の削除
						deleteTimeTableRow(task_id,period_id,processor_id);
						timetable.deleteTimeTableRow(task_id,period_id,processor_id);
						if(task_series_num>1){
							int n_period = period_id;
							for(int num=1;num<task_series_num;num++){
								n_period = day_periods.getNextPeriods(n_period);
								deleteTimeTableRow(task_id,n_period,processor_id);
							}
						}
						t_teachers.add(new Integer(processor_id));
					}
					//値の評価
					new_off = timetable.taskSeriesCountOffence();
					new_val = valuateTimeTableSQL();
					if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
					{
						//変化した物を入れておく
						new_teacher_ids = t_teachers;
						new_task_series_num=task_series_num;
						new_task_id = task_id;
						new_period_id = period_id;
						min_off = new_off;
						min_val = new_val;
						change = true;
					}
					//削除した講師を追加
					Iterator i5 = t_teachers.iterator();
					while(i5.hasNext()){
						int processor_id = ((Integer)i5.next()).intValue();
						insertTimeTableRow(task_id,period_id,processor_id);
						timetable.insertTimeTableRow(task_id,period_id,processor_id);
						if(task_series_num>1){
							int n_period = period_id;
							for(int num=1;num<task_series_num;num++){
								n_period = day_periods.getNextPeriods(n_period);
								insertTimeTableRow(task_id,n_period,processor_id);
							}
						}
					}
				}
			}
		}
		//解が改善された時
		if(change){
			//変更の表示
			if(debug) System.out.println("(task_id,processor_id,period_id)");
			Iterator i = new_teacher_ids.iterator();
			int processor_id;
			while(i.hasNext()){
				processor_id = ((Integer)i.next()).intValue();
				if(debug) System.out.println("leave:("+new_task_id+","+processor_id+","+new_period_id+")");
				deleteTimeTableRow(new_task_id,new_period_id,processor_id);
				timetable.deleteTimeTableRow(new_task_id,new_period_id,processor_id);
				if(new_task_series_num>1){
					int n_period = new_period_id;
					for(int num=1;num<new_task_series_num;num++){
						n_period = day_periods.getNextPeriods(n_period);
						deleteTimeTableRow(new_task_id,n_period,processor_id);
					}
				}
			}
		}
		return change;		
	}
	
	//解の評価値を要望別に数えて合計を返す
	public int valuateTimeTableSQL() throws Exception{
		if(valuateType==1) return valuateTimeTableSQLFile();
		else return valuateTimeTableSQLDB();
	}
	
	//解の評価値を要望別に数えて合計を返す(timetablesqlに対して行う)
	public int valuateTimeTableSQLFile() throws Exception{
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		int penalties = 0;
		ValuateSQL t = null;
		if(debug){
			if(SWflag){
				SWflag = false;
				vsw.start();
			}
			else vsw.resume();
		}
		try{
			//MySQLサーバ接続
			con = cmanager.getConnection();
			//Statementインターフェースの生成
			smt = con.createStatement();
			Iterator i = v_sqls.iterator();
			while(i.hasNext()){
				t = (ValuateSQL)i.next();
				switch(t.getType()){
				case 1:
					smt.executeUpdate(t.getSql());
					break;
				case 2:
					rs = smt.executeQuery(t.getSql());
					if(rs.next()){
						int num = rs.getInt("penalties");
						penalties = penalties + num;
					}
					break;
				case 3:
					rs = smt.executeQuery(t.getSql());
					break;
				default:
				}
			}
		}catch(SQLException e) {
			if ( t != null)
				System.out.println("\"" + t.getSql() + "\"" + e);
			else 
				System.out.println(e);				
			throw e;
		}finally{
			//ResultSetインターフェースの破棄
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statementインターフェースの破棄
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQLサーバ切断
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		if(debug)vsw.suspend();
		return penalties;
	}
	
	//解の評価値を要望別に数えて合計を返す(timetablesqlに対して行う)
	public int valuateTimeTableSQLDB() throws Exception{
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		int penalties = 0;
		ValuateSQLs t = null;
		ValuateSQL t2 = null;
		if(debug){
			if(SWflag){
				SWflag = false;
				vsw.start();
			}else{
				vsw.resume();
			}
		}
		try{
			//MySQLサーバ接続
			con = cmanager.getConnection();
			//Statementインターフェースの生成
			smt = con.createStatement();
			Iterator i = v_sqls.iterator();
			while(i.hasNext()){
				t = (ValuateSQLs)i.next();
				Iterator i2 = t.getV_sqls().iterator();
				while(i2.hasNext()){
					t2 = (ValuateSQL)i2.next();
					switch(t2.getType()){
					case 1:
						smt.executeUpdate(t2.getSql());
						break;
					case 2:
						rs = smt.executeQuery(t2.getSql());
						if(rs.next()){
							int num = rs.getInt("penalties");
							penalties = penalties + num*t.getWeight();
						}
						break;
					case 3:
						rs = smt.executeQuery(t2.getSql());
						break;
					default:
					}
				}
			}
		}catch(SQLException e) {
				if ( t != null)
					System.out.println("\"" + t2.getSql() + "\"" + e);
				else 
					System.out.println(e);				
				throw e;
		}finally{
			//ResultSetインターフェースの破棄
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statementインターフェースの破棄
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQLサーバ切断
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		if(debug) vsw.suspend();
		return penalties;
	}

	//解の評価値を要望別に数えて合計を返す(timetablesqlに対して行う)
	public int valuateTimeTableSQLDBDetail() throws Exception{
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		ArrayList v_sqls_ps = null;
		ValuateSQLsPenalty v_sqls_p = null;
		int penalties = 0;
		ValuateSQLs t = null;
		ValuateSQL t2 = null;
		if(debug){
			if(SWflag){
				SWflag = false;
				vsw.start();
			}
			else vsw.resume();
		}
		try{
			v_sqls_ps = new ArrayList();
			//MySQLサーバ接続
			con = cmanager.getConnection();
			//Statementインターフェースの生成
			smt = con.createStatement();
			Iterator i = v_sqls.iterator();
			while(i.hasNext()){
				v_sqls_p = new ValuateSQLsPenalty();
				t = (ValuateSQLs)i.next();
				v_sqls_p.setV_sqls(t);
				Iterator i2 = t.getV_sqls().iterator();
				while(i2.hasNext()){
					t2 = (ValuateSQL)i2.next();
					switch(t2.getType()){
					case 1:
						smt.executeUpdate(t2.getSql());
						break;
					case 2:
						rs = smt.executeQuery(t2.getSql());
						if(rs.next()){
							int num = rs.getInt("penalties");
							v_sqls_p.addPenalty(num*t.getWeight());
							penalties = penalties + num*t.getWeight();
						}
						break;
					case 3:
						rs = smt.executeQuery(t2.getSql());
						break;
					default:
					}
				}
				v_sqls_ps.add(v_sqls_p);
			}
			Iterator i3 = v_sqls_ps.iterator();
			while(i3.hasNext()){	
				v_sqls_p = (ValuateSQLsPenalty)i3.next();
				if(v_sqls_p.getPenalty()!=0){
					System.out.println("評価SQL");
					Iterator i4 = v_sqls_p.getV_sqls().getV_sqls().iterator();
					while(i4.hasNext()){
						t2 = (ValuateSQL)i4.next();
						System.out.println(t2.getSql());
					}
					System.out.println("評価値:"+v_sqls_p.getPenalty());
					System.out.println();
				}
			}
		}catch(Exception e) {
				if ( t2 != null)
					System.out.println("\"" + t2.getSql() + "\"" + e);
				else 
					System.out.println(e);				
				throw e;
		}finally{
			//ResultSetインターフェースの破棄
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statementインターフェースの破棄
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQLサーバ切断
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		if(debug) vsw.suspend();
		return penalties;
	}

	//解の制約違反を種類別に数えて合計を返す
	public int count_offence(){
		return timetable.countOffence();
	}
	//解の制約違反を種類別に数えて合計を返す
	//連続する講義に対応
	public int task_series_count_offence(){
		return timetable.taskSeriesCountOffence();
	}
	//解の制約違反を種類別に数えて合計を返す(詳細情報表示)
	public int count_offenceDetail(){
		return timetable.countOffenceDetail();
	}
	//解の制約違反を種類別に数えて合計を返す(詳細情報表示)
	//連続する講義に対応
	public int task_series_count_offenceDetail(){
		return timetable.taskSeriesCountOffenceDetail();
	}
	
	//テーブルtable(フィールドproject_id追加)をtimetablesqlに読み込む
	public void loadTimeTableAddProject(String table,int project_id) throws Exception{
		Connection con = null;
		Statement smt = null;
		try{
			con = cmanager.getConnection();
			smt = con.createStatement();
			//削除
			String sql = "delete from timetablesql";
			smt.executeUpdate(sql);
			sql = "insert into timetablesql(task_id,period_id,processor_id) select task_id , period_id , processor_id from "+ table+ " where project_id="+project_id;
			smt.executeUpdate(sql);
		}catch(SQLException e) {
			System.out.println(e);throw e;
		}finally{
			// Statementインターフェースの破棄
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQLサーバ切断
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}

	//テーブルtableをtimetablesqlに読み込む
	public void loadTimeTable(String table) throws Exception{
		Connection con = null;
		Statement smt = null;
		try{
			//MySQLサーバ接続
			con = cmanager.getConnection();
			//Statementインターフェースの生成
			smt = con.createStatement();
			//こDB上にすでに存在している場合そのテーブルを削除する
			String sql = "drop table if exists timetablesql";
			smt.executeUpdate(sql);
			sql = "create table timetablesql as select * from "+table;
			smt.executeUpdate(sql);
		}catch(SQLException e) {
			System.out.println(e);				
			throw e;
		}finally{
			// Statementインターフェースの破棄
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQLサーバ切断
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}
	
	//DB上にtimetablesqlのデータをtable名で保存する
	//プロジェクト管理してない時
	public void saveTimeTable(String table) throws Exception{
		Connection con = null;
		Statement smt = null;
		try{
			//MySQLサーバ接続
			con = cmanager.getConnection();
			//Statementインターフェースの生成
			smt = con.createStatement();
			//DB上にすでに存在している場合そのテーブルを削除する
			String sql = "drop table if exists " + table;
			smt.executeUpdate(sql);
			sql = "create table " + table + " as select * from timetablesql";
			smt.executeUpdate(sql);
		}catch(SQLException e) {
			System.out.println(e);				
			throw e;
		}finally{
			// Statementインターフェースの破棄
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQLサーバ切断
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}

	//バグが存在
	//DB上にtimetablesqlのデータをtable(フィールド:projectu_id追加)で保存する
	public void saveTimeTableAddProject(String table,int project_id) throws Exception{
		Connection con = null;
		PreparedStatement psmt = null;
		Statement smt = null;
		ResultSet rs = null;
		try{
			con = cmanager.getConnection();
			smt = con.createStatement();
			//DB上にすでに存在している場合,削除する
			String sql = "delete from " + table + " where project_id="+project_id;
			smt.executeUpdate(sql);
			sql = "select * from timetablesql";
			rs = smt.executeQuery(sql);
			psmt = con.prepareStatement("insert into "+table+"(task_id,period_id,processor_id,project_id) values( ? , ? , ? ,?)");
			while(rs.next()){
				psmt.setInt(1,rs.getInt("task_id"));
				psmt.setInt(2,rs.getInt("period_id"));
				psmt.setInt(3,rs.getInt("processor_id"));
				psmt.setInt(4,project_id);
				psmt.executeUpdate();
				
			}
		}catch(SQLException e) {
			System.out.println(e);				
			throw e;
		}finally{
			if(rs!=null){
				rs.close();
				rs = null;
			}
			if(psmt!=null){
				psmt.close();
				psmt = null;
			}
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//コネクションプールへ返却
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}
	//DB上にtimetablesqlのデータをtable(フィールド:projectu_id追加)で保存する
	public void saveTimeTableAddProject(String table,int project_id,int number,Timestamp date) throws Exception{
		Connection con = null;
		PreparedStatement psmt = null;
		Statement smt = null;
		ResultSet rs = null;
		try{
			con = cmanager.getConnection();
			smt = con.createStatement();
			String sql = "select * from timetablesql";
			rs = smt.executeQuery(sql);
			psmt = con.prepareStatement("insert into "+table+"(task_id,period_id,processor_id,project_id,number,create_date) values( ? , ? , ? , ? , ?, ?)");
			while(rs.next()){
				psmt.setInt(1,rs.getInt("task_id"));
				psmt.setInt(2,rs.getInt("period_id"));
				psmt.setInt(3,rs.getInt("processor_id"));
				psmt.setInt(4,project_id);
				psmt.setInt(5,number);
				psmt.setTimestamp(6,date);
				psmt.executeUpdate();
				
			}
		}catch(SQLException e) {
			System.out.println(e);				
			throw e;
		}finally{
			if(rs!=null){
				rs.close();
				rs = null;
			}
			if(psmt!=null){
				psmt.close();
				psmt = null;
			}
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//コネクションプールへ返却
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}
	//DB上のtable名のテーブルの要素でproject_idであるものを削除
	public void deleteTimetable(String table,int project_id) throws Exception{
		Connection con = null;
		Statement smt = null;
		try{
			con = cmanager.getConnection();
			smt = con.createStatement();
			//DB上にすでに存在している場合そのテーブルを削除する
			String sql = "delete from "+table+ " where project_id = "+project_id;
			smt.executeUpdate(sql);
		}catch(SQLException e) {
			System.out.println(e);				
			throw e;
		}finally{
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//コネクションプールへ返却
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}
	//リスト:lecturesからtask_idである講義を取得する
	private Lecture searchLecture(int task_id){
		Iterator i = lectures.iterator();
		while(i.hasNext()){
			Lecture l = (Lecture)i.next();
			//講義lがtask_idである
			if(l.getTask_id()==task_id){
				return l;
			}
		}
		return null;
	}
	
	//リスト:teachersからprocessor_idである講師を取得する
	private Teacher searchTeacher(int processor_id){
		Iterator i = teachers.iterator();
		while(i.hasNext()){
			Teacher t = (Teacher)i.next();
			//講師tがprocessor_idである
			if(t.getProcessor_id()==processor_id){
				return t;
			}
		}
		return null;
	}
	
	//(task_id,period_id,processor_id)を時間割timetablesqlに追加
	private void insertTimeTableRow(int task_id,int period_id,int processor_id) throws Exception{
		Connection con = null;
		Statement smt = null;
		try{
			con = cmanager.getConnection();
			smt = con.createStatement();
			//追加
			String sql="insert into timetablesql(task_id,period_id,processor_id) values('" + task_id + "','" +period_id + "','" + processor_id+"')";
			smt.executeUpdate(sql);
		}catch(SQLException e) {
			System.out.println(e);				
			throw e;
		}finally{
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//コネクションプールへ返却
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}
	//(task_id,period_id,processor_id)を時間割timetablesqlから削除
	private void deleteTimeTableRow(int task_id,int period_id, int processor_id) throws Exception{
		Connection con = null;
		Statement smt = null;
		try{
			con = cmanager.getConnection();
			smt = con.createStatement();
			//削除
			String sql = "delete from timetablesql where processor_id ='" + processor_id+"' and period_id = '" + period_id+"' and task_id = '"+task_id +"'";
			smt.executeUpdate(sql);
		}catch(SQLException e) {
			System.out.println(e);				
			throw e;
		}finally{
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//コネクションプールへ返却
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}
	
	//作業時間period_idの講義task_idを時間割timetablesqlから削除
	private void deleteTaskTimeTable(int task_id, int period_id) throws Exception{
		Connection con = null;
		Statement smt = null;
		try{
			con = cmanager.getConnection();
			smt = con.createStatement();
			//削除
			String sql = "delete from timetablesql where task_id = '"+task_id +"' and period_id ='"+ period_id+"'";
			smt.executeUpdate(sql);
		}catch(SQLException e) {
			System.out.println(e);				
			throw e;
		}finally{
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//コネクションプールへ返却
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}
	
	//講義task_idを時間割timetablesqlから削除
	private void deleteTaskTimeTable(int task_id) throws Exception{
		Connection con = null;
		Statement smt = null;
		try{
			con = cmanager.getConnection();
			smt = con.createStatement();
			//削除
			String sql = "delete from timetablesql where task_id = '"+task_id +"'";
			smt.executeUpdate(sql);
		}catch(SQLException e) {
			System.out.println(e);				
			throw e;
		}finally{
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//コネクションプールへ返却
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}
	//時間割timetablesqlの(old_task_id,old_period_id,old_processor_id)を講師new_processor_idに変更
	private void updateProcessorTimeTable(int new_processor_id,int old_processor_id, int old_task_id, int old_period_id) throws Exception{
		Connection con = null;
		Statement smt = null;
		try{
			con = cmanager.getConnection();
			smt = con.createStatement();
			String sql = "update timetablesql set processor_id ='"+new_processor_id+"' where processor_id = '"+old_processor_id+"' and task_id = '" + old_task_id + "' and period_id = '" + old_period_id + "'";
			smt.executeUpdate(sql);
		}catch(SQLException e) {
			System.out.println(e);				
			throw e;
		}finally{
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//コネクションプールへ返却
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}
	
	//現在の時間割の情報(プログラム中)を表示
	public void printTimeTable(){
		timetable.printTimeTable();
	}

	//insert文で現在の時間割の情報(プログラム中)を表示
	public void printInsertTimeTable(){
		timetable.printInsertTimeTable();
	}
	/**
	 * @return cmanager を戻します。
	 */
	public DBConnectionPool getCmanager() {
		return cmanager;
	}
	/**
	 * @param cmanager cmanager を設定。
	 */
	public void setCmanager(DBConnectionPool cmanager) {
		this.cmanager = cmanager;
	}
}
