package tts_notdb;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.Random;

import common.DBConnectionPool;

import data.CombEnum;
import data.PeriodDesire;
import data.TaskPeriod;
import data.TaskPeriodNum;
import data.TimeTableRow;
import data.notdb.LectureDetail;
import data.notdb.TeacherDetail;
import data.notdb.TimeTableNotDB;

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
public class TimeTableSchedulingNotDB {
	//全講師
	private ArrayList teachers;
	//全講義
	private ArrayList lectures;
	//データベースのコネクションプール
	private DBConnectionPool cmanager;
	//講義時間割り
	private TimeTableNotDB timetable;
	
	public TimeTableSchedulingNotDB(String host,String db,String user, String pass) throws Exception{
		cmanager = new DBConnectionPool("jdbc:mysql://"+host+"/"+db+"?useUnicode=true&characterEncoding=sjis",user,pass);
		init();
		timetable = new TimeTableNotDB(cmanager,teachers,lectures);
	}
	
	public int local_search4() throws Exception
	{
		//timetable.initTimetable();
		int i;
		for (i=1; i<=50; i++)
		{
			System.out.println(i+"回目");
			if (!search_move4()&&!search_change4()&&!search_swap4()&&!search_add4()&&!search_leave4()){
				System.out.println("これ以上無し");
				break;
			}
		}
		timetable.SaveTimetable();
		return i;
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
			//属性が入力されている全講師を取得
			//変更
			sql = "select a.processor_id ,b.employment, b.total_periods_lb , b.total_periods_ub,b.total_days_ub from processors a, processor_properties b where a.processor_id = b.processor_id order by a.processor_id";
			rs = smt.executeQuery(sql);
			while(rs.next()){
				TeacherDetail t = new TeacherDetail();
				ArrayList pds = new ArrayList();
				ArrayList qs = new ArrayList();
				t.setProcessor_id(rs.getInt("processor_id"));
				t.setEmployment(rs.getInt("employment"));
				t.setTotal_periods_lb(rs.getInt("total_periods_lb"));
				t.setTotal_periods_ub(rs.getInt("total_periods_ub"));
				t.setTotal_days_ub(rs.getInt("total_days_ub"));
				//processor_idが担当可能な時間を取得
				String sql2 = "select period_id , preferred_level_proc from processor_schedules where processor_id = '" + t.getProcessor_id() + "' order by period_id";
				rs2 = smt2.executeQuery(sql2);
				while(rs2.next()){
					PeriodDesire pd = new PeriodDesire();
					pd.setPeriod_id(rs2.getInt("period_id"));
					pd.setPreferred_level(rs2.getInt("preferred_level_proc"));
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
			System.out.println("講師の初期化終了");
			//講義の初期化
			//属性が入力されている全講義の取得
			sql = "select * from tasks a, task_properties b where a.task_id = b.task_id order by a.task_id";
			rs = smt.executeQuery(sql);
			while(rs.next()){
				LectureDetail l = new LectureDetail();
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
			System.out.println("講義の初期化終了");
		}catch(SQLException e) {
			System.err.println(e);
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
				con = null;
			}
		}
	}
	
	//コネクションの解放
	public void close() throws Exception{
		cmanager.release();
	}
	
	//初期解作成
	public void initsol() throws Exception{
		ResultSet rs = null;
		ResultSet rs2 = null;
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
			String sql = "create table t_task_relation as select a.task_id, b.period_id, a.qualification_id, a.required_processors_lb, a.required_processors_ub, c.day_id, b.preferred_level_task from task_properties a, task_opportunities b, period_properties c where a.task_id = b.task_id and b.period_id = c.period_id order by task_id, period_id";
			smt.executeUpdate(sql);
			sql = "create table t_processor_relation as select a.processor_id, c.period_id, b.qualification_id, a.employment, a.total_periods_lb, a.total_periods_ub, a.total_days_lb, a.total_days_ub, a.wage_level, d.day_id, c.preferred_level_proc from processor_properties a, processor_qualification b, processor_schedules c, period_properties d where a.processor_id = b.processor_id and c.period_id = d.period_id and a.processor_id = c.processor_id order by processor_id, period_id";
			smt.executeUpdate(sql);
			//時間割の初期化
			sql = "delete from timetableSQL";
			smt.executeUpdate(sql);
			while(true){
				//講義とその講義開講可能時間をランダムに選ぶ
				sql = "select * from t_task_relation order by rand() limit 0,1";
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
				//
				sql="select distinct processor_id,employment,total_periods_lb,total_periods_ub,total_days_lb,total_days_ub,wage_level,preferred_level_proc"
					+ " from t_processor_relation where period_id='"+period_id+"' and (qualification_id = 3 or qualification_id ='"+ qualification_id +"' ) order by rand()";
				rs2 = smt.executeQuery(sql);
				for(int i=0;rs2.next()&&i<z;i++){
					int processor_id = rs2.getInt("processor_id");
					sql="insert into timetableSQL(task_id,period_id,processor_id) values('" + task_id + "','" +period_id + "','" + processor_id +"')";
					smt2.executeUpdate(sql);
					String sql2 = "delete from t_processor_relation where processor_id ='" + processor_id+"' and period_id = '" + period_id+"'";
					smt2.executeUpdate(sql2);
				}
				rs2.close();
				//この操作により各講義は、各講義固有の開講可能時間のうち一つに開講される。
				sql="delete from t_task_relation where task_id ='" + task_id+"'";
				smt.executeUpdate(sql);
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
			}
		}
		//実験用
		timetable.initTimetable();
	}
	
	//初期解作成
	public void initsol2() throws Exception{
		ResultSet rs = null;
		ResultSet rs2 = null;
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
			String sql = "create table t_task_relation as select a.task_id, b.period_id, a.qualification_id, a.required_processors_lb, a.required_processors_ub, c.day_id, b.preferred_level_task from task_properties a, task_opportunities b, period_properties c where a.task_id = b.task_id and b.period_id = c.period_id order by task_id, period_id";
			smt.executeUpdate(sql);
			sql = "create table t_processor_relation as select a.processor_id, c.period_id, b.qualification_id, a.employment, a.total_periods_lb, a.total_periods_ub, a.total_days_lb, a.total_days_ub, a.wage_level, d.day_id, c.preferred_level_proc from processor_properties a, processor_qualification b, processor_schedules c, period_properties d where a.processor_id = b.processor_id and c.period_id = d.period_id and a.processor_id = c.processor_id order by processor_id, period_id";
			smt.executeUpdate(sql);
			//時間割の初期化
			sql = "delete from timetableSQL";
			smt.executeUpdate(sql);
			while(true){
				//講義とその講義開講可能時間をランダムに選ぶ
				sql = "select * from t_task_relation order by rand() limit 0,1";
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
				sql="select distinct processor_id,employment,total_periods_lb,total_periods_ub,total_days_lb,total_days_ub,wage_level,preferred_level_proc"
					+ " from t_processor_relation where period_id='"+period_id+"' and qualification_id ='"+ qualification_id +"' order by rand()";
				rs2 = smt.executeQuery(sql);
				for(int i=0;rs2.next()&&i<z;i++){
					int processor_id = rs2.getInt("processor_id");
					sql="insert into timetableSQL(task_id,period_id,processor_id) values('" + task_id + "','" +period_id + "','" + processor_id +"')";
					smt2.executeUpdate(sql);
					String sql2 = "delete from t_processor_relation where processor_id ='" + processor_id+"' and period_id = '" + period_id+"'";
					smt2.executeUpdate(sql2);
				}
				rs2.close();
				//この操作により各講義は、各講義固有の開講可能時間のうち一つに開講される。
				sql="delete from t_task_relation where task_id ='" + task_id+"'";
				smt.executeUpdate(sql);
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
			}
		}
		//実験用
		timetable.initTimetable();
	}

	//初期解作成
	public void initsolNotRandom(int count) throws Exception{
		ResultSet rs = null;
		ResultSet rs2 = null;
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
			String sql = "create table t_task_relation as select a.task_id, b.period_id, a.qualification_id, a.required_processors_lb, a.required_processors_ub, c.day_id, b.preferred_level_task from task_properties a, task_opportunities b, period_properties c where a.task_id = b.task_id and b.period_id = c.period_id order by task_id, period_id";
			smt.executeUpdate(sql);
			sql = "create table t_processor_relation as select a.processor_id, c.period_id, b.qualification_id, a.employment, a.total_periods_lb, a.total_periods_ub, a.total_days_lb, a.total_days_ub, a.wage_level, d.day_id, c.preferred_level_proc from processor_properties a, processor_qualification b, processor_schedules c, period_properties d where a.processor_id = b.processor_id and c.period_id = d.period_id and a.processor_id = c.processor_id order by processor_id, period_id";
			smt.executeUpdate(sql);
			//時間割の初期化
			sql = "delete from timetableSQL";
			smt.executeUpdate(sql);
			while(true){
				//講義とその講義開講可能時間をランダムに選ぶ
				sql = "select * from t_task_relation order by rand("+count+") limit 0,1";
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
				sql="select distinct processor_id,employment,total_periods_lb,total_periods_ub,total_days_lb,total_days_ub,wage_level,preferred_level_proc"
					+ " from t_processor_relation where period_id='"+period_id+"' and qualification_id ='"+ qualification_id +"' order by rand("+count+")";
				rs2 = smt.executeQuery(sql);
				for(int i=0;rs2.next()&&i<z;i++){
					int processor_id = rs2.getInt("processor_id");
					sql="insert into timetableSQL(task_id,period_id,processor_id) values('" + task_id + "','" +period_id + "','" + processor_id +"')";
					smt2.executeUpdate(sql);
					String sql2 = "delete from t_processor_relation where processor_id ='" + processor_id+"' and period_id = '" + period_id+"'";
					smt2.executeUpdate(sql2);
				}
				rs2.close();
				//この操作により各講義は、各講義固有の開講可能時間のうち一つに開講される。
				sql="delete from t_task_relation where task_id ='" + task_id+"'";
				smt.executeUpdate(sql);
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
			}
		}
		//実験用
		timetable.initTimetable();
	}
	
	//(プログラム中のテーブルを初期化しない)
	//近傍操作move:講義を行う時間を変更する(すべて調べる)(メソッド使用)
	//近傍moveを探索し、改善解があればその中で最良のものに移動しtrueを返す。なければfalseを返す。
	public boolean search_move4() throws Exception{
		boolean change = false;
		int min_val,min_off,new_val,new_off;
		ArrayList new_teacher_ids = null;
		int new_task_id = 0,new_period_id = 0,del_period_id = 0;
		//値の評価
		min_val = timetable.valuate();
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
			LectureDetail l = SearchLecture(task_id);
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
					timetable.deleteTaskTimeTable(task_id,period_id);
					//新しくtask_idを担当する講師
					ArrayList t_teachers2 = new ArrayList();
					Iterator i2 = teachers.iterator();
					while(i2.hasNext()){
						TeacherDetail t = (TeacherDetail)i2.next();
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
								timetable.insertTimeTableRow(task_id,period_id2,processor_id);
								t_teachers.add(new Integer(processor_id));
							}
							//値の評価
							new_off = timetable.countOffence();
							new_val = timetable.valuate();
							if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
							{
								//変化した物を入れておく
								new_teacher_ids = t_teachers;
								new_task_id = task_id;
								new_period_id = period_id2;
								del_period_id = period_id;
								min_off = new_off;
								min_val = new_val;
								change = true;
							}
							//追加した講義の削除
							timetable.deleteTaskTimeTable(task_id);
						}
					}
					//除いたtask_idの追加
					Iterator i4 = t_teacher3.iterator();
					while(i4.hasNext()){
						timetable.insertTimeTableRow(task_id,period_id,((Integer)i4.next()).intValue());
					}
				}
			}
		}
		//解が改善された時
		if(change){
			//変更の表示
			System.out.println("move:task_id:"+new_task_id+"がperiod_id:"+del_period_id+"からperiod_id:"+new_period_id+"に移動");
			timetable.deleteTaskTimeTable(new_task_id);
			Iterator i = new_teacher_ids.iterator();
			int processor_id;
			while(i.hasNext()){
				processor_id = ((Integer)i.next()).intValue();
				System.out.println("担当講師:"+processor_id);
				timetable.insertTimeTableRow(new_task_id,new_period_id,processor_id);
			}
		}
		return change;
	}

	//(プログラム中のテーブルを初期化しない)	
	//近傍操作Change:講義の担当講師を変更する(メソッド使用)
	//近傍changeを探索し、改善解があればその中で最良のものに移動しtrueを返す。なければfalseを返す。
	public boolean search_change4() throws Exception{
		ArrayList t_timetable = null;
		boolean change = false;
		int min_val,min_off,new_val,new_off;
		int del_processor_id = 0,new_task_id = 0,new_processor_id = 0,new_period_id = 0;
		//値の評価
		min_val = timetable.valuate();
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
			timetable.deleteTimeTableRow(task_id, period_id, processor_id);
			//task_idの講義種類を調べる
			LectureDetail l = SearchLecture(task_id);
			qualification_id = l.getQualification_id();
			//period_idに担当可能な講師に対して変更を加える
			Iterator i = teachers.iterator();
			while(i.hasNext()){
				TeacherDetail t = (TeacherDetail)i.next();
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
								timetable.insertTimeTableRow(task_id,period_id,processor_id2);
								//値の評価
								new_off = timetable.countOffence();
								new_val = timetable.valuate();
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
								timetable.deleteTimeTableRow(task_id,period_id,processor_id2);
							}
						}
					}
				}
			}
			//取り除いた(task_id,period_id,processor_id)の追加
			timetable.insertTimeTableRow(task_id,period_id,processor_id);
		}
		//解が改善された時
		if(change){
			//今までの解を変更する
			timetable.deleteTimeTableRow(new_task_id,new_period_id,del_processor_id);
			timetable.insertTimeTableRow(new_task_id,new_period_id,new_processor_id);
			//変更の表示
			System.out.println("task_id:"+new_task_id+" period_id:"+new_period_id);
			System.out.println("change:processor_id:"+del_processor_id+"がprocessor_id:"+new_processor_id+"に変化");
		}
		return change;
	}
	
	
    //(プログラム中のテーブルを初期化しない)	
	//近傍操作Swap:異なる2つの担当講師を入れ替える(メソッド使用)
	//近傍swapを探索し、改善解があればその中で最良のものに移動しtrueを返す。なければfalseを返す。
	public boolean search_swap4() throws Exception{
		ArrayList t_timetable = null;
		boolean change = false;
		int min_val,min_off,new_val,new_off;
		int new_task_id1 = 0,new_processor_id1 = 0,new_period_id1 = 0,new_task_id2 = 0,new_processor_id2 = 0,new_period_id2 = 0;
		int next = 0;
		//値の評価
		min_val = timetable.valuate();
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
					timetable.deleteTimeTableRow(task_id1,period_id1,processor_id1);
					timetable.deleteTimeTableRow(task_id2,period_id2,processor_id2);
					//task_id1とtask_id2の種類を調べる
					LectureDetail l1 = SearchLecture(task_id1);
					qualification_id1 = l1.getQualification_id();
					LectureDetail l2 = SearchLecture(task_id2);
					qualification_id2 = l2.getQualification_id();
					//(task_id1,period_id1,processor_id2)が有効かどうか
					//processor_id2がperiod_id1に可能かどうか？
					//講師tがprocessor_id2である
					TeacherDetail t = SearchTeacher(processor_id2);
					if(t.isQualification(qualification_id1)){
						//processor_id2がperiod_id1に可能
						if(t.isPeriod(period_id1)){
							//講師は同じ時間に複数の講義をしない
							if(timetable.isNotTasks(processor_id2,period_id1)){
								//(task_id2,period_id2,processor_id1)が有効かどうか
								//processor_id1がperiod_id2に担当可能かどうか？
								//講師t2がprocessor_id1である
								TeacherDetail t2 = SearchTeacher(processor_id1);
								if(t2.isQualification(qualification_id2)){
									//processor_id1がperiod_id2に可能
									if(t2.isPeriod(period_id2)){
										//講師は同じ時間に複数の講義をしない
										if(timetable.isNotTasks(processor_id1,period_id2)){
											//(task_id1,period_id1,processor_id2)の追加
											//(task_id2,period_id2,processor_id1)の追加
											timetable.insertTimeTableRow(task_id1,period_id1,processor_id2);
											timetable.insertTimeTableRow(task_id2,period_id2,processor_id1);
											new_off = timetable.countOffence();
											new_val = timetable.valuate();
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
											//追加した(task_id1,period_id1,processor_id2)の削
											//追加した(task_id2,period_id2,processor_id1)の削除
											timetable.deleteTimeTableRow(task_id1,period_id1,processor_id2);
											timetable.deleteTimeTableRow(task_id2,period_id2,processor_id1);
											}
									}
								}
							}
						}
					}
					//取り除いた(task_id1,period_id1,processor_id1)と(task_id2,period_id2,processor_id2)を追加する
					timetable.insertTimeTableRow(task_id1,period_id1,processor_id1);
					timetable.insertTimeTableRow(task_id2,period_id2,processor_id2);
				}
			}
			next++;
		}
		//解が改善された時
		if(change){
			timetable.updateProcessorTimeTable(new_processor_id2,new_processor_id1,new_task_id1,new_period_id1);
			timetable.updateProcessorTimeTable(new_processor_id1,new_processor_id2,new_task_id2,new_period_id2);
			//変更の表示
			System.out.println("(task_id,processor_id,period_id)");
			System.out.println("swap:("+new_task_id1+","+new_processor_id1+","+new_period_id1+")と("+new_task_id2+","+new_processor_id2+","+new_period_id2+")");
		}
		return change;
	}

	//(プログラム中のテーブルを初期化しない)	
	//近傍操作add:講義の担当講師を追加する(メソッド使用)
	//近傍addを探索し、改善解があればその中で最良のものに移動しtrueを返す。なければfalseを返す。
	public boolean search_add4() throws Exception{
		boolean change = false;
		ArrayList new_teacher_ids = null;
		int min_val,min_off,new_val,new_off;
		int new_task_id = 0,new_period_id = 0;
		//値の評価
		min_val = timetable.valuate();
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
			LectureDetail l = SearchLecture(task_id);
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
					TeacherDetail t = (TeacherDetail)i3.next();
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
						timetable.insertTimeTableRow(task_id,period_id,processor_id);
						t_teachers.add(new Integer(processor_id));
					}
					//値の評価
					new_off = timetable.countOffence();
					new_val = timetable.valuate();
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
						timetable.deleteTimeTableRow(task_id,period_id,t.intValue());
					}
				}
			}
		}
		//解が改善された時
		if(change){
			//変更の表示
			System.out.println("(task_id,processor_id,period_id)");
			Iterator i = new_teacher_ids.iterator();
			int processor_id;
			while(i.hasNext()){
				processor_id = ((Integer)i.next()).intValue();
				System.out.println("add:("+new_task_id+","+processor_id+","+new_period_id+")");
				timetable.insertTimeTableRow(new_task_id,new_period_id,processor_id);
			}
		}
		return change;		
	}

	//(プログラム中のテーブルを初期化しない)	
	//近傍操作leave:講義の担当講師を減らす(メソッド使用)
	//近傍leaveを探索し、改善解があればその中で最良のものに移動しtrueを返す。なければfalseを返す。
	public boolean search_leave4() throws Exception{
		boolean change = false;
		ArrayList new_teacher_ids = null;
		int min_val,min_off,new_val,new_off;
		int new_task_id = 0,new_period_id = 0;
		//値の評価
		min_val = timetable.valuate();
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
			LectureDetail l = SearchLecture(task_id);
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
						timetable.deleteTimeTableRow(task_id,period_id,processor_id);
						t_teachers.add(new Integer(processor_id));
					}
					//値の評価
					new_off = timetable.countOffence();
					new_val = timetable.valuate();
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
						timetable.insertTimeTableRow(task_id,period_id,((Integer)i5.next()).intValue());
					}
				}
			}
		}
		//解が改善された時
		if(change){
			//変更の表示
			System.out.println("(task_id,processor_id,period_id)");
			Iterator i = new_teacher_ids.iterator();
			int processor_id;
			while(i.hasNext()){
				processor_id = ((Integer)i.next()).intValue();
				System.out.println("leave:("+new_task_id+","+processor_id+","+new_period_id+")");
				timetable.deleteTimeTableRow(new_task_id,new_period_id,processor_id);
			}
		}
		return change;		
	}

	//解の評価値を要望別に数えて合計を返す
	public int valuate(String table) throws Exception{
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		int penalties = 0;
		int timetable = 0;
		try{
			//MySQLサーバ接続
			con = cmanager.getConnection();
			//Statementインターフェースの生成
			smt = con.createStatement();
			//時間割の元の個数を調べる
			String sql = "select count(*) as num from "+ table ;
			rs = smt.executeQuery(sql);
			if(rs.next()){
				timetable = rs.getInt("num");
			}
			//各講義の担当講師の人数の上限を上回るかどうか？
			sql = "create temporary table t select a.task_id,count(*),b.required_processors_ub,count(*)-CAST(b.required_processors_ub AS SIGNED) as num from "
				+table+" a, task_properties b where a.task_id = b.task_id group by a.task_id,b.required_processors_lb having count(*)-CAST(b.required_processors_ub AS SIGNED) > 0";
			smt.executeUpdate(sql);
			sql = "select sum(num) as penalties from t";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("penalties");
				penalties = num * 10000;
			}
			sql = "drop table t";
			smt.executeUpdate(sql);
			//各講義の担当講師の人数の下限を下回るかどうか？
			sql = "create temporary table t select a.task_id,count(*),b.required_processors_lb,CAST(b.required_processors_lb AS SIGNED)-count(*) as num from "
				+table+" a, task_properties b where a.task_id = b.task_id group by a.task_id,b.required_processors_ub having CAST(b.required_processors_lb AS SIGNED)-count(*) > 0";
			smt.executeUpdate(sql);
			sql = "select sum(num) as penalties from t";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("penalties");
				penalties = penalties + (num * 10000);
			}
			sql = "drop table t";
			smt.executeUpdate(sql);
			
			//各講師について、講義を担当する回数が上限を上回るかどうか？
			sql = "create temporary table t select a.processor_id,b.employment,count(*),b.total_periods_ub,(count(*) - CAST(b.total_periods_ub AS SIGNED)) as num from "
				+table+ " a, processor_properties b where a.processor_id = b.processor_id group by a.processor_id,b.total_periods_ub having count(*) - CAST(b.total_periods_ub AS SIGNED) > 0";
			smt.executeUpdate(sql);
			sql = "select sum(num*weight) as penalties from t,penalties p where (t.employment = 1 and p.name='R_WEEKOVER')or (t.employment = 2 and p.name='N_WEEKOVER')";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("penalties");
				penalties = penalties + num;
			}
			sql = "drop table t";
			smt.executeUpdate(sql);

			//各講師について、講義を担当する回数が下限を下回るかどうか？
			sql = "create temporary table t select a.processor_id,b.employment,count(*),b.total_periods_lb,(CAST(b.total_periods_lb AS SIGNED) - count(*)) as num from "
				+table+" a, processor_properties b where a.processor_id = b.processor_id group by a.processor_id,b.total_periods_lb having (CAST(b.total_periods_lb AS SIGNED) - count(*)) > 0";
			smt.executeUpdate(sql);
			sql = "select sum(num*weight) as penalties from t,penalties p where (t.employment = 1 and p.name='R_WEEKUNDER')or (t.employment = 2 and p.name='N_WEEKUNDER')";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("penalties");
				penalties = penalties +num;
			}
			sql = "drop table t";
			smt.executeUpdate(sql);

			//非常勤講師全員の講義担当回数の合計と上限値との差(上限値:40)
			sql = "select count(*) as num from " 
				+table+ " a,processor_properties b where b.employment = 2 and a.processor_id = b.processor_id";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("num");
				int temp = num - 40;
				if(temp > 0){
					penalties = penalties + ( temp * 1000);
				}
			}
			
			//各講師について、講義を担当する日数が上限を上回るか？
			sql = "create temporary table t select a.processor_id ,c.employment,c.total_days_ub , b.day_id , count(*) from "
				+table+" a , period_properties b , processor_properties c where a.period_id = b.period_id and a.processor_id = c.processor_id group by a.processor_id ,c.employment, c.total_days_ub , b.day_id";
			smt.executeUpdate(sql);
			sql = "create temporary table a select processor_id,employment,total_days_ub,count(*) as days , count(*) - CAST(total_days_ub AS SIGNED) as over from t group by processor_id,total_days_ub having count(*) - CAST(total_days_ub AS SIGNED) > 0";
			smt.executeUpdate(sql);
			sql = "select sum(over*weight) as penalties from a,penalties p where (a.employment = 1 and p.name='R_DAYSOVER')or (a.employment = 2 and p.name='N_DAYSOVER')";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("penalties");
				penalties = penalties +num;
			}
			sql = "drop table t";
			smt.executeUpdate(sql);
			sql = "drop table a";
			smt.executeUpdate(sql);
			
			//英語で行う各講義について、担当講師のうち英語を流暢に話せない物の人数
			//日本語で行う各講義について、担当講師のうち日本語を流暢に話せない物の人数
			sql = "select count(*) as num from "
				+table+" a , processor_qualification b , task_properties c where c.qualification_id = b.qualification_id and a.processor_id = b.processor_id and a.task_id = c.task_id";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("num");
				num = timetable - num;
				penalties = penalties + (num * 1000);
			}
			
			//各講師、各曜日について、担当する講義の間にあるある空き時間の数
			sql = "create table t as select a.task_id,a.period_id,a.processor_id,c.employment,b.day_id from "
				+table+" a,period_properties b ,processor_properties c where a.period_id = b.period_id and a.processor_id = c.processor_id";
			smt.executeUpdate(sql);			
			//空き時間1の時
			sql = "create temporary table c select a.processor_id,a.employment from t a , t b where a.period_id +1 != all(select period_id from t where a.processor_id = t.processor_id) and a.processor_id = b.processor_id and a.period_id = b.period_id - 2 and a.day_id = b.day_id";
			smt.executeUpdate(sql);
			sql = "select sum(weight) as penalties from c , penalties p where (c.employment = 1 and p.name='R_HOLE') or (c.employment = 2 and p.name='N_HOLE')";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("penalties");
				penalties = penalties + num;
			}
			sql = "drop table c";
			smt.executeUpdate(sql);
			//空き時間2の時
			sql = "create temporary table c select a.processor_id,a.employment from t a , t b where a.period_id+1 != all(select period_id from t where a.processor_id = t.processor_id)  and a.period_id+2 != all(select period_id from t where a.processor_id = t.processor_id) and a.processor_id = b.processor_id and a.period_id = b.period_id - 3 and a.day_id = b.day_id";
			smt.executeUpdate(sql);
			sql = "select sum(weight*2) as penalties from c , penalties p where (c.employment = 1 and p.name='R_HOLE') or (c.employment = 2 and p.name='N_HOLE')";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("penalties");
				penalties = penalties +num;
			}
			sql = "drop table c";
			smt.executeUpdate(sql);
			//空き時間3の時
			sql = "create temporary table c select a.processor_id,a.employment from t a , t b where  a.period_id+1 != all(select period_id from t where a.processor_id = t.processor_id)  and a.period_id+2 != all(select period_id from t where a.processor_id = t.processor_id) and a.period_id+3 != all(select period_id from t where a.processor_id = t.processor_id) and a.processor_id = b.processor_id and a.period_id = b.period_id - 4 and a.day_id = b.day_id";
			smt.executeUpdate(sql);
			sql = "select sum(weight*3) as penalties from c , penalties p where (c.employment = 1 and p.name='R_HOLE') or (c.employment = 2 and p.name='N_HOLE')";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("penalties");
				penalties = penalties +num;
			}
			sql = "drop table c";
			smt.executeUpdate(sql);
			sql = "drop table t";
			smt.executeUpdate(sql);
			
			//各講師について、講義を担当する講義時間における担当不満度
			sql = "create temporary table t select a.processor_id,sum(b.preferred_level_proc) as num from "
				+table+" a,processor_schedules b where a.period_id = b.period_id and a.processor_id = b.processor_id group by a.processor_id";
			smt.executeUpdate(sql);
			sql = "select sum(num) as penalties from t";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("penalties");
				penalties = penalties + num;
			}
			sql = "drop table t";
			smt.executeUpdate(sql);
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
		return penalties;
	}

	//解の評価値を要望別に数えて合計を返す
	public int valuate2(String table) throws Exception{
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		int penalties = 0;
		int timetable = 0;
		try{
			//MySQLサーバ接続
			con = cmanager.getConnection();
			//Statementインターフェースの生成
			smt = con.createStatement();
			//時間割の元の個数を調べる
			String sql = "select count(*) as num from "+ table ;
			rs = smt.executeQuery(sql);
			if(rs.next()){
				timetable = rs.getInt("num");
			}
			//各講義の担当講師の人数の上限を上回るかどうか？
			sql = "create temporary table t select a.task_id,count(*),b.required_processors_ub,count(*)-CAST(b.required_processors_ub AS SIGNED) as num from "
				+table+" a, task_properties b where a.task_id = b.task_id group by a.task_id,b.required_processors_lb having count(*)-CAST(b.required_processors_ub AS SIGNED) > 0";
			smt.executeUpdate(sql);
			sql = "select sum(num) as penalties from t";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("penalties");
				penalties = num * 10000;
			}
			sql = "drop table t";
			smt.executeUpdate(sql);
			//各講義の担当講師の人数の下限を下回るかどうか？
			sql = "create temporary table t select a.task_id,count(*),b.required_processors_lb,CAST(b.required_processors_lb AS SIGNED)-count(*) as num from "
				+table+" a, task_properties b where a.task_id = b.task_id group by a.task_id,b.required_processors_ub having CAST(b.required_processors_lb AS SIGNED)-count(*) > 0";
			smt.executeUpdate(sql);
			sql = "select sum(num) as penalties from t";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("penalties");
				penalties = penalties + (num * 10000);
			}
			sql = "drop table t";
			smt.executeUpdate(sql);
			
			//各講師について、講義を担当する回数が上限を上回るかどうか？
			sql = "create temporary table t select a.processor_id,b.employment,count(*),b.total_periods_ub,(count(*) - CAST(b.total_periods_ub AS SIGNED)) as num from "
				+table+ " a, processor_properties b where a.processor_id = b.processor_id group by a.processor_id,b.total_periods_ub having count(*) - CAST(b.total_periods_ub AS SIGNED) > 0";
			smt.executeUpdate(sql);
			sql = "select sum(num*weight) as penalties from t,penalties p where (t.employment = 1 and p.name='R_WEEKOVER')or (t.employment = 2 and p.name='N_WEEKOVER')";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("penalties");
				penalties = penalties + num;
			}
			sql = "drop table t";
			smt.executeUpdate(sql);

			//各講師について、講義を担当する回数が下限を下回るかどうか？
			sql = "select sum(num*weight) as penalties from (select a.processor_id,b.processor_id,b.employment,count(*),b.total_periods_lb,IF(a.processor_id is null,b.total_periods_lb, (CAST(b.total_periods_lb AS SIGNED) - count(*))) as num from timetableSQL a right join processor_properties b on a.processor_id = b.processor_id group by a.processor_id ,b.processor_id, b.total_periods_lb having ((CAST(b.total_periods_lb AS SIGNED) - count(*)) > 0) or a.processor_id IS NULL) as t,penalties p where (t.employment = 1 and p.name='R_WEEKUNDER') or (t.employment = 2 and p.name='N_WEEKUNDER')";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("penalties");
				penalties = penalties +num;
			}
			//非常勤講師全員の講義担当回数の合計と上限値との差(上限値:40)
			sql = "select count(*) as num from " 
				+table+ " a,processor_properties b where b.employment = 2 and a.processor_id = b.processor_id";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("num");
				int temp = num - 40;
				if(temp > 0){
					penalties = penalties + ( temp * 1000);
				}
			}
			
			//各講師について、講義を担当する日数が上限を上回るか？
			sql = "create temporary table t select a.processor_id ,c.employment,c.total_days_ub , b.day_id , count(*) from "
				+table+" a , period_properties b , processor_properties c where a.period_id = b.period_id and a.processor_id = c.processor_id group by a.processor_id ,c.employment, c.total_days_ub , b.day_id";
			smt.executeUpdate(sql);
			sql = "create temporary table a select processor_id,employment,total_days_ub,count(*) as days , count(*) - CAST(total_days_ub AS SIGNED) as over from t group by processor_id,total_days_ub having count(*) - CAST(total_days_ub AS SIGNED) > 0";
			smt.executeUpdate(sql);
			sql = "select sum(over*weight) as penalties from a,penalties p where (a.employment = 1 and p.name='R_DAYSOVER')or (a.employment = 2 and p.name='N_DAYSOVER')";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("penalties");
				penalties = penalties +num;
			}
			sql = "drop table t";
			smt.executeUpdate(sql);
			sql = "drop table a";
			smt.executeUpdate(sql);
			
			//英語で行う各講義について、担当講師のうち英語を流暢に話せない物の人数
			//日本語で行う各講義について、担当講師のうち日本語を流暢に話せない物の人数
			sql = "select count(*) as num from "
				+table+" a , processor_qualification b , task_properties c where c.qualification_id = b.qualification_id and a.processor_id = b.processor_id and a.task_id = c.task_id";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("num");
				num = timetable - num;
				penalties = penalties + (num * 1000);
			}
			
			//各講師、各曜日について、担当する講義の間にあるある空き時間の数
			sql = "create table t as select a.task_id,a.period_id,a.processor_id,c.employment,b.day_id from "
				+table+" a,period_properties b ,processor_properties c where a.period_id = b.period_id and a.processor_id = c.processor_id";
			smt.executeUpdate(sql);			
			//空き時間1の時
			sql = "create temporary table c select a.processor_id,a.employment from t a , t b where a.period_id +1 != all(select period_id from t where a.processor_id = t.processor_id) and a.processor_id = b.processor_id and a.period_id = b.period_id - 2 and a.day_id = b.day_id";
			smt.executeUpdate(sql);
			sql = "select sum(weight) as penalties from c , penalties p where (c.employment = 1 and p.name='R_HOLE') or (c.employment = 2 and p.name='N_HOLE')";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("penalties");
				penalties = penalties + num;
			}
			sql = "drop table c";
			smt.executeUpdate(sql);
			//空き時間2の時
			sql = "create temporary table c select a.processor_id,a.employment from t a , t b where a.period_id+1 != all(select period_id from t where a.processor_id = t.processor_id)  and a.period_id+2 != all(select period_id from t where a.processor_id = t.processor_id) and a.processor_id = b.processor_id and a.period_id = b.period_id - 3 and a.day_id = b.day_id";
			smt.executeUpdate(sql);
			sql = "select sum(weight*2) as penalties from c , penalties p where (c.employment = 1 and p.name='R_HOLE') or (c.employment = 2 and p.name='N_HOLE')";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("penalties");
				penalties = penalties +num;
			}
			sql = "drop table c";
			smt.executeUpdate(sql);
			//空き時間3の時
			sql = "create temporary table c select a.processor_id,a.employment from t a , t b where  a.period_id+1 != all(select period_id from t where a.processor_id = t.processor_id)  and a.period_id+2 != all(select period_id from t where a.processor_id = t.processor_id) and a.period_id+3 != all(select period_id from t where a.processor_id = t.processor_id) and a.processor_id = b.processor_id and a.period_id = b.period_id - 4 and a.day_id = b.day_id";
			smt.executeUpdate(sql);
			sql = "select sum(weight*3) as penalties from c , penalties p where (c.employment = 1 and p.name='R_HOLE') or (c.employment = 2 and p.name='N_HOLE')";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("penalties");
				penalties = penalties +num;
			}
			sql = "drop table c";
			smt.executeUpdate(sql);
			sql = "drop table t";
			smt.executeUpdate(sql);
			
			//各講師について、講義を担当する講義時間における担当不満度
			sql = "create temporary table t select a.processor_id,sum(b.preferred_level_proc) as num from "
				+table+" a,processor_schedules b where a.period_id = b.period_id and a.processor_id = b.processor_id group by a.processor_id";
			smt.executeUpdate(sql);
			sql = "select sum(num) as penalties from t";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("penalties");
				penalties = penalties + num;
			}
			sql = "drop table t";
			smt.executeUpdate(sql);
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
		return penalties;
	}
	
	//	解の評価値を要望別に数えて合計を返す(詳細)
	public int valuateDetail(String table) throws Exception{
		Connection con = null;
		ResultSet rs = null;
		ResultSet rs2 = null;
		Statement smt = null;
		Statement smt2 = null;
		int penalties = 0;
		int timetable = 0;
		try{
			//MySQLサーバ接続
			con = cmanager.getConnection();
			//Statementインターフェースの生成
			smt = con.createStatement();
			smt2 = con.createStatement();
			//時間割の元の個数を調べる
			String sql = "select count(*) as num from "+ table ;
			rs = smt.executeQuery(sql);
			if(rs.next()){
				timetable = rs.getInt("num");
			}
			//各講義の担当講師の人数の上限を上回るかどうか？
			sql = "create temporary table t select a.task_id,count(*),b.required_processors_ub,count(*)-CAST(b.required_processors_ub AS SIGNED) as num from "
				+table+" a, task_properties b where a.task_id = b.task_id group by a.task_id,b.required_processors_lb having count(*)-CAST(b.required_processors_ub AS SIGNED) > 0";
			smt.executeUpdate(sql);
			sql = "select sum(num) as penalties from t";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("penalties");
				penalties = num * 10000;
				System.out.println("各講義の担当講師の人数の上限を上回るかどうか？:"+num);
				if(num!=0){
					String sql2 = "select a.task_id,count(*),b.required_processors_ub,count(*)-CAST(b.required_processors_ub AS SIGNED) as num from "
						+table+" a, task_properties b where a.task_id = b.task_id group by a.task_id,b.required_processors_lb having count(*)-CAST(b.required_processors_ub AS SIGNED) > 0";
					System.out.println("task_id"+" "+"count(*)"+" "+"required_processors_ub"+" "+"count(*)-required_processors_ub");
					rs2 = smt2.executeQuery(sql2);
					while(rs2.next()){
						System.out.println(rs2.getInt("task_id")+"\t\t"+rs2.getInt("count(*)")+"\t\t"+rs2.getInt("required_processors_ub")+"\t\t"+rs2.getInt("num"));
					}
				}
			}
			sql = "drop table t";
			smt.executeUpdate(sql);

			//各講義の担当講師の人数の下限を下回るかどうか？
			sql = "create temporary table t select a.task_id,count(*),b.required_processors_lb,CAST(b.required_processors_lb AS SIGNED)-count(*) as num from "
				+table+" a, task_properties b where a.task_id = b.task_id group by a.task_id,b.required_processors_ub having CAST(b.required_processors_lb AS SIGNED)-count(*) > 0";
			smt.executeUpdate(sql);
			sql = "select sum(num) as penalties from t";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("penalties");
				penalties = penalties + (num * 10000);
				System.out.println("各講義の担当講師の人数の下限を下回るかどうか？:"+num);
				if(num!=0){
					String sql2 = "select a.task_id,count(*),b.required_processors_lb,CAST(b.required_processors_lb AS SIGNED)-count(*) as num from "
						+table+" a, task_properties b where a.task_id = b.task_id group by a.task_id,b.required_processors_ub having CAST(b.required_processors_lb AS SIGNED)-count(*) > 0";
					System.out.println("task_id"+" "+"count(*)"+" "+"required_processors_lb"+" "+"required_processors_lb-count(*)");
					rs2 = smt2.executeQuery(sql2);
					while(rs2.next()){
						System.out.println(rs2.getInt("task_id")+"\t\t"+rs2.getInt("count(*)")+"\t\t"+rs2.getInt("required_processors_lb")+"\t\t"+rs2.getInt("num"));
					}
				}
			}
			sql = "drop table t";
			smt.executeUpdate(sql);

			//各講師について、講義を担当する回数が上限を上回るかどうか？
			sql = "create temporary table t select a.processor_id,b.employment,count(*),b.total_periods_ub,(count(*) - CAST(b.total_periods_ub AS SIGNED)) as num from "
				+table+ " a, processor_properties b where a.processor_id = b.processor_id group by a.processor_id,b.total_periods_ub having count(*) - CAST(b.total_periods_ub AS SIGNED) > 0";
			smt.executeUpdate(sql);
			sql = "select sum(num*weight) as penalties from t,penalties p where (t.employment = 1 and p.name='R_WEEKOVER')or (t.employment = 2 and p.name='N_WEEKOVER')";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("penalties");
				penalties = penalties + num;
				if(num!=0){
					String sql2 = "select sum(num) as num2 from t";
					rs2 = smt2.executeQuery(sql2);
					if(rs2.next()){
						int num2 = rs2.getInt("num2");
						System.out.println("各講師について、講義を担当する回数が上限を上回るかどうか？:"+num2);
					}
					sql2 = "select a.processor_id,b.employment,count(*),b.total_periods_ub,(count(*) - CAST(b.total_periods_ub AS SIGNED)) as num from "
						+table+ " a, processor_properties b where a.processor_id = b.processor_id group by a.processor_id,b.total_periods_ub having count(*) - CAST(b.total_periods_ub AS SIGNED) > 0";
					System.out.println("processor_id"+" "+"employment"+" "+"count(*)"+" "+"total_periods_ub"+" "+"count(*)-total_periods_ub");
					rs2 = smt2.executeQuery(sql2);
					while(rs2.next()){
						System.out.println(rs2.getInt("processor_id")+"\t\t"+rs2.getInt("employment")+"\t\t"+rs2.getInt("count(*)")+"\t\t"+rs2.getInt("total_periods_ub")+"\t\t"+rs2.getInt("num"));
					}
				}
			}
			sql = "drop table t";
			smt.executeUpdate(sql);

			//各講師について、講義を担当する回数が下限を下回るかどうか？
			sql = "create temporary table t select a.processor_id,b.employment,count(*),b.total_periods_lb,(CAST(b.total_periods_lb AS SIGNED) - count(*)) as num from "
				+table+" a, processor_properties b where a.processor_id = b.processor_id group by a.processor_id,b.total_periods_lb having (CAST(b.total_periods_lb AS SIGNED) - count(*)) > 0";
			smt.executeUpdate(sql);
			sql = "select sum(num*weight) as penalties from t,penalties p where (t.employment = 1 and p.name='R_WEEKUNDER')or (t.employment = 2 and p.name='N_WEEKUNDER')";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("penalties");
				penalties = penalties +num;
				if(num!=0){
					String sql2 = "select sum(num) as num2 from t";
					rs2 = smt2.executeQuery(sql2);
					if(rs2.next()){
						int num2 = rs2.getInt("num2");
						System.out.println("各講師について、講義を担当する回数が下限を下回るかどうか？:"+num2);
					}
					sql2 = "select a.processor_id,b.employment,count(*),b.total_periods_lb,(CAST(b.total_periods_lb AS SIGNED) - count(*)) as num from "
						+table+" a, processor_properties b where a.processor_id = b.processor_id group by a.processor_id,b.total_periods_lb having (CAST(b.total_periods_lb AS SIGNED) - count(*)) > 0";
					System.out.println("processor_id"+" "+"employment"+" "+"count(*)"+" "+"total_periods_lb"+" "+"total_periods_lb-count(*)");
					rs2 = smt2.executeQuery(sql2);
					while(rs2.next()){
						System.out.println(rs2.getInt("processor_id")+"\t\t"+rs2.getInt("employment")+"\t\t"+rs2.getInt("count(*)")+"\t\t"+rs2.getInt("total_periods_lb")+"\t\t"+rs2.getInt("num"));
					}
				}
			}
			sql = "drop table t";
			smt.executeUpdate(sql);

			//非常勤講師全員の講義担当回数の合計と上限値との差(上限値:40)
			sql = "select count(*) as num from "
				+table+ " a,processor_properties b where b.employment = 2 and a.processor_id = b.processor_id";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("num");
				int temp = num - 40;
				if(temp > 0){
					penalties = penalties + ( temp * 1000);
					System.out.println("非常勤講師全員の講義担当回数の合計と上限値との差(上限値:40):"+temp);
				}
			}

			//各講師について、講義を担当する日数が上限を上回るか？
			sql = "create temporary table t select a.processor_id ,c.employment,c.total_days_ub , b.day_id , count(*) from "
				+table+" a , period_properties b , processor_properties c where a.period_id = b.period_id and a.processor_id = c.processor_id group by a.processor_id ,c.employment, c.total_days_ub , b.day_id";
			smt.executeUpdate(sql);
			sql = "create temporary table a select processor_id,employment,total_days_ub,count(*) as days , count(*) - CAST(total_days_ub AS SIGNED) as over from t group by processor_id,total_days_ub having count(*) - CAST(total_days_ub AS SIGNED) > 0";
			smt.executeUpdate(sql);
			sql = "select sum(over*weight) as penalties from a,penalties p where (a.employment = 1 and p.name='R_DAYSOVER')or (a.employment = 2 and p.name='N_DAYSOVER')";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("penalties");
				penalties = penalties +num;
				if(num!=0){
					String sql2 = "select sum(over) as num2 from a";
					rs2 = smt2.executeQuery(sql2);
					if(rs2.next()){
						int num2 = rs2.getInt("num2");
						System.out.println("各講師について、講義を担当する日数が上限を上回るか？:"+num2);
					}
					sql2 = "select processor_id,employment,total_days_ub,count(*) as days , count(*) - CAST(total_days_ub AS SIGNED) as over from t group by processor_id,total_days_ub having count(*) - CAST(total_days_ub AS SIGNED) > 0";
					System.out.println("processor_id"+" "+"employment"+" "+"total_days_ub"+" "+"count(*)"+" "+"count(*)-total_days_ub");
					rs2 = smt2.executeQuery(sql2);
					while(rs2.next()){
						System.out.println(rs2.getInt("processor_id")+"\t\t"+rs2.getInt("employment")+"\t\t"+rs2.getInt("total_days_ub")+"\t\t"+rs2.getInt("days")+"\t\t"+rs2.getInt("over"));
					}
				}
			}
			sql = "drop table t";
			smt.executeUpdate(sql);
			sql = "drop table a";
			smt.executeUpdate(sql);

			//英語で行う各講義について、担当講師のうち英語を流暢に話せない物の人数
			//日本語で行う各講義について、担当講師のうち日本語を流暢に話せない物の人数
			sql = "select count(*) as num from "
				+table+" a , processor_qualification b , task_properties c where c.qualification_id = b.qualification_id and a.processor_id = b.processor_id and a.task_id = c.task_id";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("num");
				num = timetable - num;
				penalties = penalties + (num * 1000);
				System.out.println("担当講師のうち流暢に話せない物の人数:"+num);
				if(num!=0){
					String sql2 = "select a.processor_id,b.qualification_id as qualification_id1,a.task_id,c.qualification_id as qualification_id2 from "
						+table+" a , processor_qualification b , task_properties c where c.qualification_id != b.qualification_id and a.processor_id = b.processor_id and a.task_id = c.task_id and c.qualification_id !=3 and b.qualification_id !=3";
					System.out.println("processor_id"+" "+"qualification_id"+" "+"task_id"+" "+"qualification_id");
					rs2 = smt2.executeQuery(sql2);
					while(rs2.next()){
						System.out.println(rs2.getInt("processor_id")+"\t\t"+rs2.getInt("qualification_id1")+"\t\t"+rs2.getInt("task_id")+"\t\t"+rs2.getInt("qualification_id2"));
					}
				}
			}

			//各講師、各曜日について、担当する講義の間にあるある空き時間の数
			sql = "create table t as select a.task_id,a.period_id,a.processor_id,c.employment,b.day_id from "
				+table+" a,period_properties b ,processor_properties c where a.period_id = b.period_id and a.processor_id = c.processor_id";
			smt.executeUpdate(sql);
			//空き時間1の時
			sql = "create temporary table c select a.processor_id,a.employment from t a , t b where a.period_id +1 != all(select period_id from t where a.processor_id = t.processor_id) and a.processor_id = b.processor_id and a.period_id = b.period_id - 2 and a.day_id = b.day_id";
			smt.executeUpdate(sql);
			sql = "select sum(weight) as penalties from c , penalties p where (c.employment = 1 and p.name='R_HOLE') or (c.employment = 2 and p.name='N_HOLE')";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("penalties");
				penalties = penalties + num;
				if(num!=0){
					String sql2 = "select count(*) as num2 from c";
					rs2 = smt2.executeQuery(sql2);
					if(rs2.next()){
						int num2 = rs2.getInt("num2");
						System.out.println("空き時間1の時:"+num2);
					}
					sql2 = "select a.processor_id,a.day_id,a.task_id as task_id1,b.task_id as task_id2,a.period_id as period_id1, b.period_id as period_id2 from t a , t b where a.period_id +1 != all(select period_id from t where a.processor_id = t.processor_id) and a.processor_id = b.processor_id and a.period_id = b.period_id - 2 and a.day_id = b.day_id";
					System.out.println("processor_id"+" "+"day_id"+" "+"task_id1"+" "+"period_id1"+" "+"task_id2"+" "+"period_id2");
					rs2 = smt2.executeQuery(sql2);
					while(rs2.next()){
						System.out.println(rs2.getInt("processor_id")+"\t\t"+rs2.getInt("day_id")+"\t\t"+rs2.getInt("task_id1")+"\t\t"+rs2.getInt("period_id1")+"\t\t"+rs2.getInt("task_id2")+"\t\t"+rs2.getInt("period_id2"));
					}
				}
			}
			sql = "drop table c";
			smt.executeUpdate(sql);
			//空き時間2の時
			sql = "create temporary table c select a.processor_id,a.employment from t a , t b where a.period_id+1 != all(select period_id from t where a.processor_id = t.processor_id)  and a.period_id+2 != all(select period_id from t where a.processor_id = t.processor_id) and a.processor_id = b.processor_id and a.period_id = b.period_id - 3 and a.day_id = b.day_id";
			smt.executeUpdate(sql);
			sql = "select sum(weight*2) as penalties from c , penalties p where (c.employment = 1 and p.name='R_HOLE') or (c.employment = 2 and p.name='N_HOLE')";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("penalties");
				penalties = penalties +num;
				if(num!=0){
					String sql2 = "select count(*) as num2 from c";
					rs2 = smt2.executeQuery(sql2);
					if(rs2.next()){
						int num2 = rs2.getInt("num2");
						System.out.println("空き時間2の時:"+num2);
					}
					sql2 = "select a.processor_id,a.day_id,a.task_id as task_id1,b.task_id as task_id2,a.period_id as period_id1, b.period_id as period_id2 from t a , t b where a.period_id+1 != all(select period_id from t where a.processor_id = t.processor_id)  and a.period_id+2 != all(select period_id from t where a.processor_id = t.processor_id) and a.processor_id = b.processor_id and a.period_id = b.period_id - 3 and a.day_id = b.day_id";
					System.out.println("processor_id"+" "+"day_id"+" "+"task_id1"+" "+"period_id1"+" "+"task_id2"+" "+"period_id2");
					rs2 = smt2.executeQuery(sql2);
					while(rs2.next()){
						System.out.println(rs2.getInt("processor_id")+"\t\t"+rs2.getInt("day_id")+"\t\t"+rs2.getInt("task_id1")+"\t\t"+rs2.getInt("period_id1")+"\t\t"+rs2.getInt("task_id2")+"\t\t"+rs2.getInt("period_id2"));
					}
				}
			}
			sql = "drop table c";
			smt.executeUpdate(sql);
			//空き時間3の時
			sql = "create temporary table c select a.processor_id,a.employment from t a , t b where  a.period_id+1 != all(select period_id from t where a.processor_id = t.processor_id)  and a.period_id+2 != all(select period_id from t where a.processor_id = t.processor_id) and a.period_id+3 != all(select period_id from t where a.processor_id = t.processor_id) and a.processor_id = b.processor_id and a.period_id = b.period_id - 4 and a.day_id = b.day_id";
			smt.executeUpdate(sql);
			sql = "select sum(weight*3) as penalties from c , penalties p where (c.employment = 1 and p.name='R_HOLE') or (c.employment = 2 and p.name='N_HOLE')";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("penalties");
				penalties = penalties +num;
				if(num!=0){
					String sql2 = "select count(*) as num2 from c";
					rs2 = smt2.executeQuery(sql2);
					if(rs2.next()){
						int num2 = rs2.getInt("num2");
						System.out.println("空き時間3の時:"+num2);
					}
					sql2 = "select a.processor_id,a.day_id,a.task_id as task_id1,b.task_id as task_id2,a.period_id as period_id1, b.period_id as period_id2 from t a , t b where  a.period_id+1 != all(select period_id from t where a.processor_id = t.processor_id)  and a.period_id+2 != all(select period_id from t where a.processor_id = t.processor_id) and a.period_id+3 != all(select period_id from t where a.processor_id = t.processor_id) and a.processor_id = b.processor_id and a.period_id = b.period_id - 4 and a.day_id = b.day_id";
					System.out.println("processor_id"+" "+"day_id"+" "+"task_id1"+" "+"period_id1"+" "+"task_id2"+" "+"period_id2");
					rs2 = smt2.executeQuery(sql2);
					while(rs2.next()){
						System.out.println(rs2.getInt("processor_id")+"\t\t"+rs2.getInt("day_id")+"\t\t"+rs2.getInt("task_id1")+"\t\t"+rs2.getInt("period_id1")+"\t\t"+rs2.getInt("task_id2")+"\t\t"+rs2.getInt("period_id2"));
					}
				}
			}
			sql = "drop table c";
			smt.executeUpdate(sql);
			sql = "drop table t";
			smt.executeUpdate(sql);

			//各講師について、講義を担当する講義時間における担当不満度
			sql = "create temporary table t select a.processor_id,sum(b.preferred_level_proc) as num from "
				+table+" a,processor_schedules b where a.period_id = b.period_id and a.processor_id = b.processor_id group by a.processor_id";
			smt.executeUpdate(sql);
			sql = "select sum(num) as penalties from t";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("penalties");
				penalties = penalties + num;
				System.out.println("各講師について、講義を担当する講義時間における担当不満度:"+num);
			}
			sql = "drop table t";
			smt.executeUpdate(sql);
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
			if(rs2!=null){
				rs2.close();
				rs2 = null;
			}
			// Statementインターフェースの破棄
			if(smt!=null){
				smt.close();
				smt = null;
			}
			if(smt2!=null){
				smt2.close();
				smt2 = null;
			}
			//MySQLサーバ切断
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		return penalties;
	}
	
	public int valuate() throws Exception{
		//timetable.initTimetable();
		int penalties = 0;
		//penalties = valuateTimeTableSQLFile();
		penalties = timetable.valuate();
		return penalties;
	}
	
	//解の制約違反を種類別に数えて合計を返す(一時テーブル作成)
	public int count_offence() throws Exception{
		return timetable.countOffence();
	}
	
	//解の制約違反を種類別に数えて合計を返す(一時テーブル作成)
	public int count_offence(String table) throws Exception{
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		int timetable  = 0;
		int penalties = 0;
		int t_task_num = 0;
		try{
			//MySQLサーバ接続
			con = cmanager.getConnection();
			//Statementインターフェースの生成
			smt = con.createStatement();
			//時間割の元の個数を調べる
			String sql = "select count(*) as num from "+table;
			rs = smt.executeQuery(sql);
			if(rs.next()){
				timetable = rs.getInt("num");
			}
			//各講師は担当可能時間に講義を担当(担当可能な講義かどうかは調べていない)
			sql = "create temporary table t select a.processor_id,a.task_id,a.period_id from "
				+table+" a,processor_schedules b where a.processor_id = b.processor_id and a.period_id = b.period_id";
			smt.executeUpdate(sql);
			sql = "select count(*) as num from t";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("num");
				penalties = timetable - num;
				//System.out.println("各講師は担当可能時間に講義を担当"+penalties);
				
			}
			sql = "drop table t";
			smt.executeUpdate(sql);
			//各講師は、同じ時間に複数の講義を担当しない
			sql = "create temporary table t select processor_id,period_id, count(*)-1 as num from "
				+table+" group by processor_id,period_id having count(*)-1 > 0";
			smt.executeUpdate(sql);
			sql = "select sum(num) as num2 from t";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("num2");
				penalties += num;
				//System.out.println("各講師は、同じ時間に複数の講義を担当しない"+num);
			}
			sql = "drop table t";
			smt.executeUpdate(sql);
			//各講義は、各講義固有の開講可能時間のうち一つに開講される
			//1.各講義は開講可能時間に開講される
			sql = "create temporary table t select a.processor_id,a.task_id,a.period_id from "
				+table+" a,task_opportunities b where a.task_id = b.task_id and a.period_id = b.period_id;";
			smt.executeUpdate(sql);
			sql = "select count(*) as num from t";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("num");
				penalties = penalties + (timetable - num);
				//System.out.println("各講義は開講可能時間に開講される"+(timetable - num));
			}
			sql = "drop table t";
			smt.executeUpdate(sql);
			//2.各講義は開講可能であれば必ず開講される
			sql = "create temporary table t select count(*) as num from "
				+table+" group by task_id";
			smt.executeUpdate(sql);
			sql = "select count(*) as num2 from t";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				t_task_num = rs.getInt("num2");
			}
			sql = "drop table t";
			smt.executeUpdate(sql);
			sql = "create temporary table t select count(*) as num from task_opportunities group by task_id";
			smt.executeUpdate(sql);
			sql = "select count(*) as num2 from t";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("num2");
				penalties = penalties + (num - t_task_num);
				//System.out.println("各講義は開講可能であれば必ず開講される"+(num - t_task_num));
			}
			sql = "drop table t";
			smt.executeUpdate(sql);
			//各講義はただ一つだけ開講される
			sql = "create temporary table t select count(*) as num from "
				+table+" group by task_id,period_id";
			smt.executeUpdate(sql);
			sql = "select count(*) as num2 from t";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("num2");
				penalties = penalties + (num - t_task_num);
				//System.out.println("各講義はただ一つだけ開講される"+(num - t_task_num));
			}
			sql = "drop table t";
			smt.executeUpdate(sql);
		}catch(SQLException e) {
			System.err.println(e);
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
		return penalties;
	}
	
	//解の制約違反を種類別に数えて合計を返す(評価SQLの変更ver1)
	public int count_offence2(String table) throws Exception{
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		int timetable  = 0;
		int penalties = 0;
		int t_task_num = 0;
		try{
			//MySQLサーバ接続
			con = cmanager.getConnection();
			//Statementインターフェースの生成
			smt = con.createStatement();
			//時間割の元の個数を調べる
			String sql = "select count(*) as num from "+table;
			rs = smt.executeQuery(sql);
			if(rs.next()){
				timetable = rs.getInt("num");
			}
			//各講師は担当可能時間に講義を担当(担当可能な講義かどうかは調べていない)
			sql = "select count(*) as num from (select a.processor_id,a.task_id,a.period_id from "
				+table+" a,processor_schedules b where a.processor_id = b.processor_id and a.period_id = b.period_id) as t";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("num");
				penalties = timetable - num;
				//System.out.println("各講師は担当可能時間に講義を担当"+penalties);
				
			}
			//各講師は、同じ時間に複数の講義を担当しない
			sql = "select sum(num) as num2 from (select processor_id,period_id, count(*)-1 as num from "
				+table+" group by processor_id,period_id having count(*)-1 > 0) as t";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("num2");
				penalties += num;
				//System.out.println("各講師は、同じ時間に複数の講義を担当しない"+num);
			}
			//各講義は、各講義固有の開講可能時間のうち一つに開講される
			//1.各講義は開講可能時間に開講される
			sql = "select count(*) as num from (select a.processor_id,a.task_id,a.period_id from "
				+table+" a,task_opportunities b where a.task_id = b.task_id and a.period_id = b.period_id) as t";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("num");
				penalties = penalties + (timetable - num);
				//System.out.println("各講義は開講可能時間に開講される"+(timetable - num));
			}
			//2.各講義は開講可能であれば必ず開講される
			sql = "select count(*) as num2 from (select count(*) as num from "
				+table+" group by task_id) as t";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				t_task_num = rs.getInt("num2");
			}
			sql = "select count(*) as num2 from (select count(*) as num from task_opportunities group by task_id) as t";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("num2");
				penalties = penalties + (num - t_task_num);
				//System.out.println("各講義は開講可能であれば必ず開講される"+(num - t_task_num));
			}
			//各講義はただ一つだけ開講される
			sql = "select count(*) as num2 from (select count(*) as num from "
				+table+" group by task_id,period_id) as t";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("num2");
				penalties = penalties + (num - t_task_num);
				//System.out.println("各講義はただ一つだけ開講される"+(num - t_task_num));
			}
		}catch(SQLException e) {
			System.err.println(e);
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
		return penalties;
	}
	
	//tableをtimetableSQLにコピー
	public void LoadTimetable(String table) throws Exception{
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		try{
			//MySQLサーバ接続
			con = cmanager.getConnection();
			//Statementインターフェースの生成
			smt = con.createStatement();
			//こDB上にすでに存在している場合そのテーブルを削除する
			String sql = "drop table if exists timetableSQL";
			smt.executeUpdate(sql);
			sql = "create table timetableSQL as select * from "+table;
			smt.executeUpdate(sql);
		}catch(SQLException e) {
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
	}
	//DB上に現在のtimetableSQLをtable名で保存する
	public void SaveTimetable(String table) throws Exception{
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		try{
			//MySQLサーバ接続
			con = cmanager.getConnection();
			//Statementインターフェースの生成
			smt = con.createStatement();
			//こDB上にすでに存在している場合そのテーブルを削除する
			String sql = "drop table if exists " + table;
			smt.executeUpdate(sql);
			sql = "create table " + table + " as select * from timetableSQL";
			smt.executeUpdate(sql);
		}catch(SQLException e) {
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
	}
	//task_idである講義を取得する
	private LectureDetail SearchLecture(int task_id){
		Iterator i = lectures.iterator();
		while(i.hasNext()){
			LectureDetail l = (LectureDetail)i.next();
			//講義lがtask_idである
			if(l.getTask_id()==task_id){
				return l;
			}
		}
		return null;
	}
	
	//processor_idである講師を取得する
	private TeacherDetail SearchTeacher(int processor_id){
		Iterator i = teachers.iterator();
		while(i.hasNext()){
			TeacherDetail t = (TeacherDetail)i.next();
			//講師tがprocessor_idである
			if(t.getProcessor_id()==processor_id){
				return t;
			}
		}
		return null;
	}
}
