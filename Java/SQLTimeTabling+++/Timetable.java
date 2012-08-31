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
 	        }
	        System.out.println("Finished the initialization of lectures. ");
		} catch(SQLException e) {
			System.err.println(e);
			throw e;
			//e.printStackTrace();
//		} catch(Exception e) {
//			System.err.println(e);
			//e.printStackTrace();
		} finally{
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
		//ファイルから評価SQLを取得する
	    BufferedReader reader = null;
	    //v_sqls = new ArrayList();
	    //t_v_sqls = new ArrayList();
	    try {
	        reader = new BufferedReader(new FileReader(file));
	        String line;
	        while ((line = reader.readLine()) != null) {
	    	    ValuateSQL v_sql = null;
	    	    ValuateSQL t_v_sql = null;
	        	StringTokenizer strToken = new StringTokenizer(line, ";");
	    		if(strToken.hasMoreTokens()) {
	    			int type = Integer.parseInt(strToken.nextToken().toString());
	    			v_sql = new ValuateSQL();
	    			v_sql.setType(type);
	    			t_v_sql = new ValuateSQL();
	    			t_v_sql.setType(type);
	    		}
	    		if(strToken.hasMoreTokens()) {
	    			//String 
					sql = strToken.nextToken().toString();
	    			v_sql.setSql(sql);
	    			t_v_sql.setSql(sql.replaceAll("timetableSQL","t_timetable"));
	    		}
	    		if(v_sql!=null)v_sqls.add(v_sql);
	    		if(t_v_sql!=null)t_v_sqls.add(t_v_sql);
	        }
			//if (reader != null) {
			reader.close();
			//}
	    } catch (FileNotFoundException e) {
	    	System.out.println("File not found: " + e);
			throw e;
	    } //catch (IOException e) {
	    //} // finally {		
		  //}
	    /*
	    System.out.println("timetableSQLに対するSQL文");
		Iterator i = v_sqls.iterator();
		while(i.hasNext()){
			ValuateSQL t = (ValuateSQL)i.next();
			System.out.print(t.getType());
			System.out.println(":"+t.getSql());
		}
		System.out.println("t_timetableに対するSQL文");
		Iterator i2 = t_v_sqls.iterator();
		while(i2.hasNext()){
			ValuateSQL t = (ValuateSQL)i2.next();
			System.out.print(t.getType());
			System.out.println(":"+t.getSql());
		}
		*/
	}
		
//初期化
private void getRequestsFromFile(String file) throws Exception {
	//ファイルから評価SQLを取得する
	String sql;
	BufferedReader reader = null;
	try {
		reader = new BufferedReader(new FileReader(file));
		String line;
		while ((line = reader.readLine()) != null) {
			ValuateSQL v_sql = null;
			ValuateSQL t_v_sql = null;
			StringTokenizer strToken = new StringTokenizer(line, ";");
			if(strToken.hasMoreTokens()) {
				int type = Integer.parseInt(strToken.nextToken().toString());
				v_sql = new ValuateSQL();
				v_sql.setType(type);
				t_v_sql = new ValuateSQL();
				t_v_sql.setType(type);
			}
			if(strToken.hasMoreTokens()) {
				//String 
				sql = strToken.nextToken().toString();
				v_sql.setSql(sql);
				t_v_sql.setSql(sql.replaceAll("timetableSQL","t_timetable"));
			}
			if(v_sql!=null)v_sqls.add(v_sql);
			if(t_v_sql!=null)t_v_sqls.add(t_v_sql);
		}
		reader.close();
	} catch (FileNotFoundException e) {
		System.out.println("File not found: " + e);
		throw e;
	}
}

//初期化
private void getWeightsFromDB() throws Exception {	
	Connection con = null;
		
	try {
		//MySQLサーバ接続
		con = cmanager.getConnection();
	} catch (SQLException e) {
		System.err.println("Couldn't get connection: " + e);
		throw e;
	}
	//Statementインターフェースの生成
	Statement smt = con.createStatement();
	Statement smt2 = con.createStatement();
	ResultSet rs = null;
	ResultSet rs2 = null;
	String sql;	
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
		}
		System.out.println("Finished the initialization of lectures. ");
	} catch(SQLException e) {
		System.err.println(e);
		throw e;
	} finally{
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
	public void close(){
		cmanager.release();
	}
	
protected void finalize() {
	close();
}
	
	//初期解作成
	public void initsol() throws Exception {
		ResultSet rs = null;
        ResultSet rs2 = null;
		Connection con = null;
		Statement smt = null;
		Statement smt2 = null;
		String sql;
		//シードを与えて、Randomクラスのインスタンスを生成する。
		Random rand = new Random(Calendar.getInstance().getTimeInMillis());
		try{
			//MySQLサーバ接続
			con = cmanager.getConnection();
			//Statementインターフェースの生成
            smt = con.createStatement();
			smt2 = con.createStatement();
			//String 
			sql = "create table IF NOT EXISTS t_task_relation as select a.task_id, b.period_id, a.qualification_id, a.required_processors_lb, a.required_processors_ub, c.day_id, b.preferred_level_task from task_properties a, task_opportunities b, period_properties c where a.task_id = b.task_id and b.period_id = c.period_id order by task_id, period_id";
			smt.executeUpdate(sql);
			sql = "create table IF NOT EXISTS t_processor_relation as select a.processor_id, c.period_id, b.qualification_id, a.employment, a.total_periods_lb, a.total_periods_ub, a.total_days_lb, a.total_days_ub, a.wage_level, d.day_id, c.preferred_level_proc from processor_properties a, processor_qualification b, processor_schedules c, period_properties d where a.processor_id = b.processor_id and c.period_id = d.period_id and a.processor_id = c.processor_id order by processor_id, period_id";
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
				int preferred_level_task;
				int day_id;
				int z;
				if(rs.next()) {
					task_id = rs.getInt("task_id");
					period_id = rs.getInt("period_id");
					qualification_id = rs.getInt("qualification_id");
					required_processors_lb = rs.getInt("required_processors_lb");
					required_processors_ub = rs.getInt("required_processors_ub");
					preferred_level_task = rs.getInt("preferred_level_task");
					day_id = rs.getInt("day_id");
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
					int employment = rs2.getInt("employment");
					int total_periods_lb = rs2.getInt("total_periods_lb");
					int total_periods_ub = rs2.getInt("total_periods_ub");
					int total_days_lb = rs2.getInt("total_days_lb");
					int total_days_ub = rs2.getInt("total_days_ub");
					int wage_level = rs2.getInt("wage_level");
					int preferred_level_proc = rs2.getInt("preferred_level_proc");
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
			System.err.println("Reason：" + e.toString());
			throw e;
			//e.printStackTrace();
//		}catch(Exception e){
			//e.printStackTrace();
		}finally{
//			try{
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
//			}catch(Exception e){
				//何もしない
//			}
		}
	}
	//近傍操作move:講義を行う時間を変更する
	//近傍moveを探索し、改善解があればその中で最良のものに移動しtrueを返す。なければfalseを返す。
	public boolean search_move() throws Exception {
		Connection con = null;
		ResultSet rs = null;
	    ResultSet rs2 = null;
	    ResultSet rs3 = null;
		Statement smt = null;
		Statement smt2 = null;
		Statement smt3 = null;
		Statement smt4 = null;
		boolean change = false;
		int min_val,min_off,new_val,new_off;
		ArrayList new_teacher_ids = null;
		int new_task_id = 0,new_period_id = 0,del_period_id = 0;
		//値の評価
        min_val = valuateFile();
        //min_val = valuate("timetableSQL");
		min_off = count_offence("timetableSQL");

		try{
			//MySQLサーバ接続
			con = cmanager.getConnection();
			//Statementインターフェースの生成
 	        smt = con.createStatement();
			smt2 = con.createStatement();
			smt3 = con.createStatement();
			smt4 = con.createStatement();
			//操作用のテーブルを作成
			String sql = "create table t_timetable as select * from timetableSQL";
			smt.executeUpdate(sql);
			//現在時間割に入っている講義を取得
			sql = "select task_id ,period_id from timetableSQL group by task_id,period_id";
			rs = smt.executeQuery(sql);
	        while(rs.next()){ 
				int task_id = rs.getInt("task_id");
				int period_id = rs.getInt("period_id");
				int qualification_id;
				int required_processors_lb;
				int required_processors_ub;
				//task_id1の開講可能時間を調べる
				Iterator i = lectures.iterator();
				while(i.hasNext()){
					Lecture l = (Lecture)i.next();
					//講義lがtask_idである
					if(l.getTask_id()==task_id){
						qualification_id = l.getQualification_id();
						required_processors_lb = l.getRequired_processors_lb();
						required_processors_ub = l.getRequired_processors_ub();
						//task_id1の開講可能時間の取得
						Iterator i2 = l.getPeriods().iterator();
						while(i2.hasNext()){
							PeriodDesire lpd = (PeriodDesire)i2.next();
							int period_id2 = lpd.getPeriod_id();
							//System.out.println("task_id:"+task_id);
							//System.out.println("period_id:"+period_id);
							//講義の開講時間が現在開講されている時間でない
							if(period_id!=period_id2){
								ArrayList t_teacher3 = new ArrayList();
								//task_idを除く
					            String sql2 = "select processor_id from t_timetable where task_id = '" + task_id+"' and period_id ='"+ period_id+"'";
					            rs2 = smt2.executeQuery(sql2);
					            while(rs2.next()){
					            	t_teacher3.add(new Integer(rs2.getInt("processor_id")));
					            }
								//task_idを除く
					            sql2 = "delete from t_timetable where task_id ='" + task_id+"' and period_id ='"+ period_id+"'";
								smt2.executeUpdate(sql2);
								ArrayList t_teachers2 = new ArrayList();
								Iterator i3 = teachers.iterator();
								while(i3.hasNext()){
									Teacher t = (Teacher)i3.next();
									Iterator i4 = t.getPeriods().iterator();
									while(i4.hasNext()){
										PeriodDesire tpd = (PeriodDesire)i4.next();
										//講義の開講可能時間に講師が担当可能
										if(lpd.getPeriod_id()==tpd.getPeriod_id()){
											int processor_id = t.getProcessor_id();
											//講師は同じ時間に複数の講義をしない
											String sql3 = "select * from t_timetable where processor_id='"+processor_id+"' and period_id = '" + period_id2 +"'";
											rs3 = smt3.executeQuery(sql3);
											if(rs3.next()==false){
												t_teachers2.add(new Integer(processor_id));
												//System.out.println("追加可能:"+processor_id);
											}
											break;
										}
									}
								}
								//講義.担当人数下限 <= z <= 講義.担当人数上限
								for(int z=required_processors_lb;z<=required_processors_ub;z++){
									//period_id2に担当可能な講師を取得（講義を担当可能かを調べていない）
									ArrayList t_teachers = null;
									Enumeration e = new CombEnum(t_teachers2.toArray(), z);
									while(e.hasMoreElements()) {
										t_teachers = new ArrayList();
										Object[] a = (Object[])e.nextElement();
										//System.out.println("追加する人数:"+z);
										for(int num2=0; num2<a.length; num2++){
											int processor_id = ((Integer)a[num2]).intValue();
											//講師の追加
											String sql4="insert into t_timetable(task_id,period_id , processor_id) values("+ "'" + task_id + "','" +period_id2 + "','" + processor_id+"')";
											smt4.executeUpdate(sql4);
											t_teachers.add(new Integer(processor_id));
											//System.out.println("追加:"+processor_id);
										}
										//値の評価
										new_off = count_offence("t_timetable");
										//new_val = valuate("t_timetable");
										new_val = valuateFile2();
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
										String sql3="delete from t_timetable where task_id ='"+task_id+"'";
										smt3.executeUpdate(sql3);
									}
								}
								//除いたtask_idの追加
								Iterator i4 = t_teacher3.iterator();
								while(i4.hasNext()){
									sql2="insert into t_timetable(task_id,period_id , processor_id) values("+ "'" + task_id + "','" +period_id + "','" + ((Integer)i4.next()).intValue()+"')";
									smt2.executeUpdate(sql2);
								}
							}
						}
						break;
					}
				}
            }
			//操作用テーブルの削除
			sql = "drop table t_timetable";
			smt.executeUpdate(sql);
			//解が改善された時
			if(change){
				System.out.println("move:task_id:"+new_task_id+"がperiod_id:"+del_period_id+"からperiod_id:"+new_period_id+"に移動");
				sql="delete from timetableSQL where task_id ='"+new_task_id+"'";
				smt.executeUpdate(sql);
				Iterator i = new_teacher_ids.iterator();
				int processor_id;
				while(i.hasNext()){
					processor_id = ((Integer)i.next()).intValue();
					System.out.println("Professor for the lecture:"+processor_id);
					sql="insert into timetableSQL(task_id,period_id , processor_id) values("+ "'" + new_task_id + "','" +new_period_id + "','" + processor_id+"')";
					smt.executeUpdate(sql);
				}
			}
		}catch(SQLException e) {
			System.err.println(e);
			//e.printStackTrace();
			throw e;
		//}catch(Exception e){
		//	System.err.println(e);
			//e.printStackTrace();
		}finally{
		//	try{
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
				//ResultSetインターフェースの破棄
				if(rs3!=null){
					rs3.close();
					rs3 = null;
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
				// Statementインターフェースの破棄
				if(smt3!=null){
					smt3.close();
					smt3 = null;
				}
				// Statementインターフェースの破棄
				if(smt4!=null){
					smt4.close();
					smt4 = null;
				}
				//MySQLサーバ切断
				if(con!=null){
					cmanager.freeConnection(con);
				}
		//}catch(Exception e){
		//		System.err.println(e);
		//		e.printStackTrace();
		//	}
		}
		return change;
	}
	//近傍操作Change:講義の担当講師を変更する
	//近傍changeを探索し、改善解があればその中で最良のものに移動しtrueを返す。なければfalseを返す。
	public boolean search_change() throws Exception {
		Connection con = null;
		ResultSet rs = null;
		ResultSet rs2 = null;
		Statement smt = null;
		Statement smt2 = null;
		Statement smt3 = null;
		boolean change = false;
		int min_val,min_off,new_val,new_off;
		int del_processor_id = 0,new_task_id = 0,new_processor_id = 0,new_period_id = 0;
		//値の評価
        //min_val = valuate("timetableSQL");
		min_val = valuateFile();
		min_off = count_offence("timetableSQL");
		try{
			//MySQLサーバ接続
			con = cmanager.getConnection();
			//Statementインターフェースの生成
			smt = con.createStatement();
			smt2 = con.createStatement();
			smt3 = con.createStatement();
			//操作用のテーブルを作成
			String sql = "create table IF NOT EXISTS t_timetable as select * from timetableSQL";
			smt.executeUpdate(sql);
			//現在の時間割の取得
			sql = "select * from timetableSQL";
			rs = smt.executeQuery(sql);
			int task_id;
			int period_id;
			int processor_id;
			while(rs.next()) {
				task_id = rs.getInt("task_id");
				period_id = rs.getInt("period_id");
				processor_id = rs.getInt("processor_id");
				//(task,period,processor)を取り除く
				String sql2 = "delete from t_timetable where processor_id ='" + processor_id+"' and period_id = '" + period_id+"' and task_id = '"+task_id +"'";
				smt2.executeUpdate(sql2);
				//period_idに担当可能な講師を取得（講義を担当可能かを調べていない）
				Iterator i = teachers.iterator();
				//System.out.println("period_id:"+period_id);
				while(i.hasNext()){
					Teacher t = (Teacher)i.next();
					//現在の講師と同じ講師でない
					if(t.getProcessor_id()!=processor_id){
						Iterator i2 = t.getPeriods().iterator();
						while(i2.hasNext()){
							PeriodDesire pd = (PeriodDesire)i2.next();
							//period_idに講師tが担当可能
							if(pd.getPeriod_id()==period_id){
								//System.out.println("processor_id:"+t.getProcessor_id());
								int processor_id2 = t.getProcessor_id();
								//講師は同じ時間に複数の講義をしない
								sql2 = "select * from t_timetable where processor_id='"+processor_id2+"' and period_id = '" + period_id +"'";
								rs2 = smt2.executeQuery(sql2);
								if(rs2.next()==false){
									//講師の追加
		            				String sql3="insert into t_timetable(task_id,period_id,processor_id) values('" + task_id + "','" +period_id + "','" + processor_id2+"')";
		            				smt3.executeUpdate(sql3);
		        					new_off = count_offence("t_timetable");
		        					//new_val = valuate("t_timetable");
		        					new_val = valuateFile2();
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
		            				//追加した講師の削除
		            				sql3="delete from t_timetable where processor_id ='" + processor_id2+"' and period_id = '" + period_id+"' and task_id = '"+task_id +"'";
		            				smt3.executeUpdate(sql3);
								}
								break;
							}
						}
					}
				}
				//取り除いた講師の追加
				sql2 ="insert into t_timetable(task_id,period_id,processor_id) values('" + task_id + "','" +period_id + "','" + processor_id+"')";
				smt2.executeUpdate(sql2);
			}
			//操作用テーブルの削除
			sql = "drop table t_timetable";
			smt.executeUpdate(sql);
			//解が改善された時
			if(change){
				System.out.println("task_id:"+new_task_id+" period_id:"+new_period_id);
				System.out.println("change:processor_id:"+del_processor_id+"-> processor_id:"+new_processor_id+".");
				sql = "delete from timetableSQL where processor_id ='" + del_processor_id+"' and period_id = '" + new_period_id+"' and task_id = '"+new_task_id +"'";
				smt.executeUpdate(sql);
				sql = "insert into timetableSQL(task_id,period_id , processor_id) values("+ "'" + new_task_id + "','" +new_period_id + "','" + new_processor_id+"')";
				smt.executeUpdate(sql);
			}
		}catch(SQLException e) {
			System.err.println(e);
			//e.printStackTrace();
			throw e;
		//}catch(Exception e){
		//	System.err.println(e);
			//e.printStackTrace();
		}finally{
		//	try{
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
				// Statementインターフェースの破棄
				if(smt3!=null){
					smt3.close();
					smt3 = null;
				}
				//MySQLサーバ切断
				if(con!=null){
					cmanager.freeConnection(con);
				}
		//	}catch(Exception e){
		//		System.err.println(e);
				//e.printStackTrace();
		//	}
		}
		return change;
	}
	//近傍操作Swap:異なる2つの担当講師を入れ替える
    //近傍swapを探索し、改善解があればその中で最良のものに移動しtrueを返す。なければfalseを返す。
	public boolean search_swap() throws Exception {
		Connection con = null;
		ResultSet rs = null;
	    ResultSet rs2 = null;
	    ResultSet rs3 = null;
		ResultSet rs4 = null;
	    ResultSet rs5 = null;
		Statement smt = null;
		Statement smt2 = null;
		Statement smt3 = null;
		Statement smt4 = null;
		Statement smt5 = null;
		Statement smt6 = null;
		boolean change = false;
		int min_val,min_off,new_val,new_off;
		int new_task_id1 = 0,new_processor_id1 = 0,new_period_id1 = 0,new_task_id2 = 0,new_processor_id2 = 0,new_period_id2 = 0;
		//値の評価
        //min_val = valuate("timetableSQL");
		min_val = valuateFile();
		min_off = count_offence("timetableSQL");
		try{
			//MySQLサーバ接続
			con = cmanager.getConnection();
			//Statementインターフェースの生成
 	        smt = con.createStatement();
			smt2 = con.createStatement();
			smt3 = con.createStatement();
			smt4 = con.createStatement();
 	        smt5 = con.createStatement();
			smt6 = con.createStatement();
			//操作用のテーブルを作成
			String sql = "create table t_timetable as select * from timetableSQL";
			smt.executeUpdate(sql);
			//現在の時間割の取得
			sql = "select * from timetableSQL";
			rs = smt.executeQuery(sql);
			//現在の時間割の取得
			rs2 = smt2.executeQuery(sql);
	        while(rs.next()){
				int task_id1 = rs.getInt("task_id");
				int period_id1 = rs.getInt("period_id");
				int processor_id1 = rs.getInt("processor_id");
				while(rs2.next()){
					int task_id2 = rs2.getInt("task_id");
					int period_id2 = rs2.getInt("period_id");
					int processor_id2 = rs2.getInt("processor_id");
					//異なる講義かどうか
					if(task_id1!=task_id2){
						//(task_id1,processor_id1,period_id1)と(task_id2,processor_id2,period_id2)を取り除く
	            		String sql2 = "delete from t_timetable  where processor_id='"+processor_id1+"' and period_id= '" + period_id1+"' and task_id= '"+task_id1 +"'";
            			smt3.executeUpdate(sql2);
            			sql2 = "delete from t_timetable  where processor_id='"+processor_id2+"' and period_id = '" + period_id2+"' and task_id = '"+task_id2 +"'";
						smt3.executeUpdate(sql2);
						//(task_id1,period_id1,processor_id2)が有効かどうか
						//processor_id2がperiod_id1に可能かどうか？
						Iterator i = teachers.iterator();
						while(i.hasNext()){
							Teacher t = (Teacher)i.next();
							//講師tがprocessor_id2である
							if(t.getProcessor_id()==processor_id2){
								Iterator i2 = t.getPeriods().iterator();
								while(i2.hasNext()){
									PeriodDesire pd = (PeriodDesire)i2.next();
									//processor_id2がperiod_id1に可能
									if(pd.getPeriod_id()==period_id1){
										//講師は同じ時間に複数の講義をしない
			            				String sql3 = "select * from t_timetable where processor_id='"+processor_id2+"' and period_id = '" + period_id1 +"'";
			            				rs4 = smt4.executeQuery(sql3);
										if(rs4.next()==false){
											//(task_id2,period_id2,processor_id1)が有効かどうか
											//processor_id1がperiod_id2に担当可能かどうか？
											Iterator i3 = teachers.iterator();
											while(i3.hasNext()){
												Teacher t2 = (Teacher)i3.next();
												//講師t2がprocessor_id1である
												if(t2.getProcessor_id()==processor_id1){
													Iterator i4 = t2.getPeriods().iterator();
													while(i4.hasNext()){
														PeriodDesire pd2 = (PeriodDesire)i4.next();
														//processor_id1がperiod_id2に可能
														if(pd2.getPeriod_id()==period_id2){
															//講師は同じ時間に複数の講義をしない
						            						String sql4 = "select * from t_timetable where processor_id='"+processor_id1+"' and period_id = '" + period_id2 +"'";
						            						rs5 = smt5.executeQuery(sql4);
															if(rs5.next()==false){
																//講師の追加
						            							String sql5="insert into t_timetable(task_id,period_id , processor_id) values("+ "'" + task_id1 + "','" +period_id1 + "','" + processor_id2+"')";
						            							smt6.executeUpdate(sql5);
																sql5="insert into t_timetable(task_id,period_id , processor_id) values("+ "'" + task_id2 + "','" +period_id2 + "','" + processor_id1+"')";
						            							smt6.executeUpdate(sql5);
						                    					new_off = count_offence("t_timetable");
						                    					//new_val = valuate("t_timetable");
						                    					new_val = valuateFile2();
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
																//追加した講師の削除
																sql5="delete from t_timetable where processor_id ='" + processor_id2+"' and period_id = '" + period_id1+"' and task_id = '"+task_id1 +"'";
																smt6.executeUpdate(sql5);
																sql5="delete from t_timetable where processor_id ='" + processor_id1+"' and period_id = '" + period_id2+"' and task_id = '"+task_id2 +"'";
																smt6.executeUpdate(sql5);
															}
															break;
														}
													}
													break;
												}
											}
										}
										break;
									}
								}
								break;
							}
						}
						//取り除いた(task_id1,processor_id1,period_id1)と(task_id2,processor_id2,period_id2)を追加する
	            		sql2 = "insert into t_timetable(task_id,period_id , processor_id) values("+ "'" + task_id1 + "','" +period_id1 + "','" + processor_id1+"')";
            			smt3.executeUpdate(sql2);
            			sql2 = "insert into t_timetable(task_id,period_id , processor_id) values("+ "'" + task_id2 + "','" +period_id2 + "','" + processor_id2+"')";
						smt3.executeUpdate(sql2);
					}
				}
				//初めに戻す
				rs2.beforeFirst();
	        }
	        //操作用テーブルの削除
			sql = "drop table t_timetable";
			smt.executeUpdate(sql);
			//解が改善された時
			if(change){
				System.out.println("(task_id,processor_id,period_id)");
				System.out.println("swap:("+new_task_id1+","+new_processor_id1+","+new_period_id1+")と("+new_task_id2+","+new_processor_id2+","+new_period_id2+")");
				sql = "update timetableSQL set processor_id ='"+new_processor_id2+"' where processor_id = '"+new_processor_id1+"' and task_id = '" + new_task_id1 + "' and period_id = '" + new_period_id1 + "'";
				smt.executeUpdate(sql);
				sql = "update timetableSQL set processor_id ='"+new_processor_id1+"' where processor_id = '"+new_processor_id2+"' and task_id = '" + new_task_id2 + "' and period_id = '" + new_period_id2 + "'";
				smt.executeUpdate(sql);
			}
		}catch(SQLException e) {
			System.err.println(e);
			//e.printStackTrace();
			throw e;
		//}catch(Exception e){
		//	System.err.println(e);
			//e.printStackTrace();
		}finally{
		//	try{
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
				//ResultSetインターフェースの破棄
				if(rs3!=null){
					rs3.close();
					rs3 = null;
				}
				//ResultSetインターフェースの破棄
				if(rs4!=null){
					rs4.close();
					rs4 = null;
				}
				//ResultSetインターフェースの破棄
				if(rs5!=null){
					rs5.close();
					rs5 = null;
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
				// Statementインターフェースの破棄
				if(smt3!=null){
					smt3.close();
					smt3 = null;
				}
				// Statementインターフェースの破棄
				if(smt4!=null){
					smt4.close();
					smt4 = null;
				}
				// Statementインターフェースの破棄
				if(smt5!=null){
					smt5.close();
					smt5 = null;
				}
				// Statementインターフェースの破棄
				if(smt6!=null){
					smt6.close();
					smt6 = null;
				}
				//MySQLサーバ切断
				if(con!=null){
					cmanager.freeConnection(con);
				}
		//	}catch(Exception e){
			//	System.err.println(e);
				//e.printStackTrace();
			//}
		}
        return change;
	}
	//近傍操作add:講義の担当講師を追加する
	//近傍addを探索し、改善解があればその中で最良のものに移動しtrueを返す。なければfalseを返す。
	public boolean search_add() throws Exception {
		Connection con = null;
		ResultSet rs = null;
		ResultSet rs2 = null;
		Statement smt = null;
		Statement smt2 = null;
		Statement smt3 = null;
		boolean change = false;
		ArrayList new_teacher_ids = null;
		int min_val,min_off,new_val,new_off;
		int del_processor_id = 0,new_task_id = 0,new_processor_id = 0,new_period_id = 0;
		//値の評価
        //min_val = valuate("timetableSQL");
		min_val = valuateFile();
		min_off = count_offence("timetableSQL");
		try{
			//MySQLサーバ接続
			con = cmanager.getConnection();
			//Statementインターフェースの生成
			smt = con.createStatement();
			smt2 = con.createStatement();
			smt3 = con.createStatement();
			//操作用のテーブルを作成
			String sql = "create table t_timetable as select * from timetableSQL";
			smt.executeUpdate(sql);
			//現在の時間割の講義とその担当人数と開講時間を取得
			sql = "select task_id , period_id, count(*) as num from timetableSQL group by task_id ,period_id order by task_id";
			rs = smt.executeQuery(sql);
			int task_id;
			int period_id;
			int num;
			int low;
			while(rs.next()) {
				task_id = rs.getInt("task_id");
				period_id = rs.getInt("period_id");
				num = rs.getInt("num");
				//task_idである講義の取得
				Iterator i = lectures.iterator();
				while(i.hasNext()){
					Lecture l = (Lecture)i.next();
					//講義lがtask_idである
					if(l.getTask_id()==task_id){
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
								Iterator i4 = t.getPeriods().iterator();
								while(i4.hasNext()){
									PeriodDesire tpd = (PeriodDesire)i4.next();
									if(period_id==tpd.getPeriod_id()){
										int processor_id = t.getProcessor_id();
										//講師は同じ時間に複数の講義をしない
										String sql2 = "select * from t_timetable where processor_id='"+processor_id+"' and period_id = '" + period_id +"'";
										rs2 = smt2.executeQuery(sql2);
										if(rs2.next()==false){
											t_teachers2.add(new Integer(processor_id));	
											//System.out.println("追加可能:"+processor_id);
										}
										break;
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
									String sql3="insert into t_timetable(task_id,period_id , processor_id) values("+ "'" + task_id + "','" +period_id + "','" + processor_id+"')";
									smt3.executeUpdate(sql3);
									t_teachers.add(new Integer(processor_id));
									//System.out.println("追加:"+processor_id);
								}
								//値の評価
								new_off = count_offence("t_timetable");
								//new_val = valuate("t_timetable");
								new_val = valuateFile2();
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
									String sql2="delete from t_timetable where task_id ='"+task_id+"' and period_id = '"+period_id+"' and processor_id = '"+t.intValue()+"'";
									smt2.executeUpdate(sql2);
								}
							}
						}
						break;
					}
				}
			}
			//操作用テーブルの削除
			sql = "drop table t_timetable";
			smt.executeUpdate(sql);
			//解が改善された時
			if(change){
				System.out.println("(task_id,processor_id,period_id)");
				Iterator i = new_teacher_ids.iterator();
				int processor_id;
				while(i.hasNext()){
					processor_id = ((Integer)i.next()).intValue();
					System.out.println("add:("+new_task_id+","+processor_id+","+new_period_id+")");
					sql="insert into timetableSQL(task_id,period_id,processor_id) values('" + new_task_id + "','" +new_period_id + "','" + processor_id+"')";
					smt.executeUpdate(sql);
				}
			}
		} catch(SQLException e) {
			System.err.println(e);
			//e.printStackTrace();
			throw e;
		//}catch(Exception e){
		//	System.err.println(e);
			//e.printStackTrace();
		}finally{
			//try{
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
				// Statementインターフェースの破棄
				if(smt3!=null){
					smt3.close();
					smt3 = null;
				}
				//MySQLサーバ切断
				if(con!=null){
					cmanager.freeConnection(con);
				}
			//}catch(Exception e){
			//	System.err.println(e);
				//e.printStackTrace();
			//}
		}
		return change;		
	}
	//近傍操作leave:講義の担当講師を減らす
	//近傍leaveを探索し、改善解があればその中で最良のものに移動しtrueを返す。なければfalseを返す。
	public boolean search_leave() throws Exception {
		Connection con = null;
		ResultSet rs = null;
		ResultSet rs2 = null;
		Statement smt = null;
		Statement smt2 = null;
		Statement smt3 = null;
		boolean change = false;
		ArrayList new_teacher_ids = null;
		int min_val,min_off,new_val,new_off;
		int new_task_id = 0,new_processor_id = 0,new_period_id = 0;
		//値の評価
        //min_val = valuate("timetableSQL");
		min_val = valuateFile();
		min_off = count_offence("timetableSQL");
		try{
			//MySQLサーバ接続
			con = cmanager.getConnection();
			//Statementインターフェースの生成
			smt = con.createStatement();
			smt2 = con.createStatement();
			smt3 = con.createStatement();
			//操作用のテーブルを作成
			String sql = "create table t_timetable as select * from timetableSQL";
			smt.executeUpdate(sql);
			//現在の時間割の講義とその担当人数と開講時間を取得
			sql = "select task_id , period_id, count(*) as num from timetableSQL group by task_id order by task_id";
			rs = smt.executeQuery(sql);
			int task_id;
			int period_id;
			int num;
			int low;
			while(rs.next()) {
				task_id = rs.getInt("task_id");
				period_id = rs.getInt("period_id");
				num = rs.getInt("num");
				//task_idである講義の取得
				Iterator i = lectures.iterator();
				while(i.hasNext()){
					Lecture l = (Lecture)i.next();
					//講義lがtask_idである
					if(l.getTask_id()==task_id){
						//task_idの担当人数下限以上の人数の取得
						int up = num-l.getRequired_processors_lb();
						ArrayList t_teachers2 = null;
						//System.out.println("up:"+up);
						//もし講義の担当人数下限以上ならばその講義を担当している全講師のprocessor_idを取得
						if(up>0){
							t_teachers2 = new ArrayList();
							//period_idにtask_idを担当する講師を取得
							String sql2 = "select processor_id from t_timetable where task_id='"+task_id+"' and period_id = '" + period_id +"'";
							rs2 = smt2.executeQuery(sql2);
							//System.out.println("task_id:"+task_id);
							while(rs2.next()){
								//System.out.println("担当している講師:"+rs2.getInt("processor_id"));
								t_teachers2.add(new Integer(rs2.getInt("processor_id")));
							}
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
									String sql3="delete from t_timetable where task_id ='"+task_id+"' and period_id = '"+period_id+"' and processor_id = '"+processor_id+"'";
									smt3.executeUpdate(sql3);
									t_teachers.add(new Integer(processor_id));
									//System.out.println("削除:"+processor_id);
								}
								//値の評価
								new_off = count_offence("t_timetable");
								//new_val = valuate("t_timetable");
								new_val = valuateFile2();
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
									String sql3="insert into t_timetable(task_id,period_id , processor_id) values("+ "'" + task_id+ "','" +period_id + "','" + ((Integer)i5.next()).intValue()+"')";
									smt3.executeUpdate(sql3);
								}
							}
						}
						break;
					}
				}
			}
			//操作用テーブルの削除
			sql = "drop table t_timetable";
			smt.executeUpdate(sql);
			//解が改善された時
			if(change){
				System.out.println("(task_id,processor_id,period_id)");
				Iterator i = new_teacher_ids.iterator();
				int processor_id;
				while(i.hasNext()){
					processor_id = ((Integer)i.next()).intValue();
					System.out.println("leave:("+new_task_id+","+processor_id+","+new_period_id+")");
					sql="delete from timetableSQL where task_id ='"+new_task_id+"' and period_id = '"+new_period_id+"' and processor_id = '"+processor_id+"'";
					smt.executeUpdate(sql);
				}
			}
		}catch(SQLException e) {
			System.err.println(e);
			//e.printStackTrace();
			throw e;
		//}catch(Exception e){
		//	System.err.println(e);
			//e.printStackTrace();
		}finally{
			//try{
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
				// Statementインターフェースの破棄
				if(smt3!=null){
					smt3.close();
					smt3 = null;
				}
				//MySQLサーバ切断
				if(con!=null){
					cmanager.freeConnection(con);
				}
			//}catch(Exception e){
			//	System.err.println(e);
				//e.printStackTrace();
			//}
		}
		return change;		
	}
	//解の評価値を要望別に数えて合計を返す
	public int valuate(String table) throws Exception {
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
				System.out.println("各講義の担当講師の人数の上限を上回るかどうか？"+num);
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
				System.out.println("各講義の担当講師の人数の下限を下回るかどうか？"+num);
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
				System.out.println("各講師について、講義を担当する回数が上限を上回るかどうか？"+num);
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
				System.out.println("各講師について、講義を担当する回数が下限を下回るかどうか？"+num);
			}
			sql = "drop table t";
			smt.executeUpdate(sql);

			//非常勤講師全員の講義担当回数の合計と上限値との差(上限値:38)
			sql = "select count(*) as num from " 
				+table+ " a,processor_properties b where b.employment = 2 and a.processor_id = b.processor_id";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("num");
				int temp = num - 38;
				if(temp > 0){
					penalties = penalties + ( temp * 1000);
					System.out.println("非常勤講師全員の講義担当回数の合計と上限値との差(上限値:38)"+temp);
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
				System.out.println("各講師について、講義を担当する日数が上限を上回るか？"+num);
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
				penalties = penalties + (num * 75);
				System.out.println("担当講師のうち流暢に話せない物の人数"+num);
			}

			//各講師、各曜日について、担当する講義の間にあるある空き時間の数
			sql = "create table t as select a.task_id,a.period_id,a.processor_id,c.employment,b.day_id from "
				+table+" a,period_properties b ,processor_properties c where a.period_id = b.period_id and a.processor_id = c.processor_id";
			smt.executeUpdate(sql);			
			//空き時間1の時
			sql = "create temporary table c select a.processor_id,a.employment from t a , t b where a.processor_id = b.processor_id and a.period_id != b.period_id -1 and a.period_id = b.period_id -2 and a.day_id = b.day_id";
			smt.executeUpdate(sql);
			sql = "select sum(weight) as penalties from c , penalties p where (c.employment = 1 and p.name='R_HOLE') or (c.employment = 2 and p.name='N_HOLE')";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("penalties");
				penalties = penalties + num;
				System.out.println("空き時間1の時"+num);
			}
			sql = "drop table c";
			smt.executeUpdate(sql);
			//空き時間2の時
			sql = "create temporary table c select a.processor_id,a.employment from t a , t b where a.processor_id = b.processor_id and a.period_id != b.period_id -1 and a.period_id != b.period_id -2 and a.period_id = b.period_id -3 and a.day_id = b.day_id";
			smt.executeUpdate(sql);
			sql = "select sum(weight*2) as penalties from c , penalties p where (c.employment = 1 and p.name='R_HOLE') or (c.employment = 2 and p.name='N_HOLE')";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("penalties");
				penalties = penalties +num;
				System.out.println("空き時間2の時"+num);
			}
			sql = "drop table c";
			smt.executeUpdate(sql);
			//空き時間3の時
			sql = "create temporary table c select a.processor_id,a.employment from t a , t b where a.processor_id = b.processor_id and a.period_id != b.period_id -1 and a.period_id != b.period_id -2 and a.period_id != b.period_id -3 and a.period_id = b.period_id -4 and a.day_id = b.day_id";
			smt.executeUpdate(sql);
			sql = "select sum(weight*3) as penalties from c , penalties p where (c.employment = 1 and p.name='R_HOLE') or (c.employment = 2 and p.name='N_HOLE')";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("penalties");
				penalties = penalties +num;
				System.out.println("空き時間3の時"+num);
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
				System.out.println("各講師について、講義を担当する講義時間における担当不満度"+num);
			}
			sql = "drop table t";
			smt.executeUpdate(sql);
			
		}catch(SQLException e) {
			System.out.println(e);
			//e.printStackTrace();
			throw e;
		//}catch(Exception e){
		//	System.out.println(e);
			//e.printStackTrace();
		}finally{
			//try{
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
			//}catch(Exception e){
			//	System.out.println(e);
				//e.printStackTrace();
			//}
		}
		return penalties;
	}
	//解の評価値を要望別に数えて合計を返す(t_timetableに対して行う)
	public int valuateFile2() throws Exception {
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
 	        Iterator i = t_v_sqls.iterator();
 			while(i.hasNext()){
 				ValuateSQL t = (ValuateSQL)i.next();
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
			System.out.println(e);
			//e.printStackTrace();
			throw e;
		//}catch(Exception e){
		//	System.out.println(e);
			//e.printStackTrace();
		}finally{
			//try{
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
			//}catch(Exception e){
			//	System.out.println(e);
				//e.printStackTrace();
			//}
		}
		return penalties;
	}
	
	//解の評価値を要望別に数えて合計を返す(timetableSQLに対して行う)
	public int valuateFile() throws Exception {
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		int penalties = 0;
		int timetable = 0;
		
		ValuateSQL t = null;
		
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
			//e.printStackTrace();
			throw e;
		//}catch(Exception e){
		//	System.out.println(e);
			//e.printStackTrace();
		}finally{
			//try{
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
			//}catch(Exception e){
			//	System.out.println(e);
				//e.printStackTrace();
			//}
		}
		return penalties;
	}
	
	//解の制約違反を種類別に数えて合計を返す
	public int count_offence(String table) throws Exception {
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
			System.out.println(e);
			//e.printStackTrace();
			throw e;
		//}catch(Exception e){
		//	System.out.println(e);
			//e.printStackTrace();
		}finally{
			//try{
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
			//}catch(Exception e){
			//	System.out.println(e);
				//e.printStackTrace();
			//}
		}
		return penalties;
	}
}
