/*
 * 作成日: 2007/04/02
 *
 * TODO この生成されたファイルのテンプレートを変更するには次へジャンプ:
 * ウィンドウ - 設定 - Java - コード・スタイル - コード・テンプレート
 */
package data.notdb;

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
import data.DayPeriod;
import data.PeriodDesire;
import data.ProcessorPeriod;
import data.TaskPeriod;
import data.TaskPeriodNum;
import data.TimeTableRow;

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
			//属性が入力されている全講師を取得
			sql = "select a.processor_id from processors a, processor_properties b where a.processor_id = b.processor_id order by a.processor_id";
			rs = smt.executeQuery(sql);
			while(rs.next()){
				TeacherDetail t = new TeacherDetail();
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
	public void printTimeTable(){
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			System.out.println(t.toString());
		}
	}
	
	public void SaveTimetable() throws Exception{
		Connection con = null;
		PreparedStatement psmt = null;
		Statement smt = null;
		try{
			//MySQLサーバ接続
			con = cmanager.getConnection();
			//Statementインターフェースの生成
			smt = con.createStatement();
			smt.executeUpdate("delete from timetableSQL");
			psmt = con.prepareStatement("insert into timetableSQL(task_id,period_id,processor_id) values( ? , ? , ? )");
			Iterator i = timetablerows.iterator();
			while(i.hasNext()){
				TimeTableRow t = (TimeTableRow)i.next();
				psmt.setInt(1,t.getTask_id());
				psmt.setInt(2,t.getPeriod_id());
				psmt.setInt(3,t.getProcessor_id());
				psmt.executeUpdate();
				
			}
		}catch(SQLException e) {
			System.out.println(e);				
			throw e;
		}finally{
			if(psmt!=null){
				smt.close();
				smt = null;
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
	public void initTimetable() throws Exception{
		timetablerows.clear();
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		int task_id,period_id,processor_id;
		try{
			//MySQLサーバ接続
			con = cmanager.getConnection();
			//Statementインターフェースの生成
			smt = con.createStatement();
			//現在の時間割の取得
			String sql = "select * from timetableSQL";
			rs = smt.executeQuery(sql);
			while(rs.next()) {
				task_id = rs.getInt("task_id");
				period_id = rs.getInt("period_id");
				processor_id = rs.getInt("processor_id");
				timetablerows.add(new TimeTableRow(task_id,period_id,processor_id));
			}
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
	
	public void insertTimeTableRow(int task_id,int period_id,int processor_id){
		timetablerows.add(new TimeTableRow(task_id,period_id,processor_id));
	}
	
	public void deleteTimeTableRow(int task_id,int period_id,int processor_id){
		timetablerows.remove(new TimeTableRow(task_id,period_id,processor_id));
	}
	
	public void updateProcessorTimeTable(int new_processor_id,int old_processor_id, int old_task_id, int old_period_id){
		timetablerows.remove(new TimeTableRow(old_task_id,old_period_id,old_processor_id));
		timetablerows.add(new TimeTableRow(old_task_id,old_period_id,new_processor_id));
	}
	
	public void deleteTaskTimeTable(int task_id){
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			if(t.getTask_id()==task_id) i.remove();
		}
	}
	
	public void deleteTaskTimeTable(int task_id, int period_id){
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			if((t.getTask_id()==task_id)&&(t.getPeriod_id()==period_id)) i.remove();
		}
	}
	
	//制約違反していないかどうか(違反していない場合:true している:false)
	public int countOffence(){
		Set temp1 = new TreeSet();
		ArrayList temp2 = new ArrayList();
	    ArrayList temp3 = new ArrayList();
		//●1.各講師は担当可能時間に(担当可能な講義)を担当
		//2.各講義は、各講義固有の開講可能時間のうち一つに開講される
		//●2-1.各講義は開講可能時間に開講される
		//●2-2.各講義は開講可能であれば必ず開講される
		//▲2-3.各講義はただ一つだけ開講される
		//▲3.各講師は、同じ時間に複数の講義を担当しない
		Iterator i = timetablerows.iterator();
		int count = 0;
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			TeacherDetail teacher = SearchTeacher(t.getProcessor_id());
			LectureDetail lecture = SearchLecture(t.getTask_id());
			if(!teacher.isPeriod(t.getPeriod_id()))count++;
			//if(!teacher.isQualification(lecture.getQualification_id()))count++;
			if(!lecture.isPeriod(t.getPeriod_id()))count++;
			temp1.add(new Integer(t.getTask_id()));
			if(temp2.contains(new TaskPeriod(t.getTask_id(),t.getPeriod_id()))){
				temp2.add(new TaskPeriod(t.getTask_id(),t.getPeriod_id()));
			}
			if(temp3.contains(new ProcessorPeriod(t.getProcessor_id(),t.getPeriod_id()))){
				temp3.add(new ProcessorPeriod(t.getProcessor_id(),t.getPeriod_id()));
			}
		}
		if(temp1.size()!=lectures.size())count++;
		i = temp2.iterator();
		while(i.hasNext()){
			TaskPeriod t = (TaskPeriod)i.next();
			if(!isNotTasks3(t.getTask_id(),t.getPeriod_id()))count++;
		}
		
		i = temp3.iterator();
		while(i.hasNext()){
			ProcessorPeriod t = (ProcessorPeriod)i.next();
			if(!isNotTasks2(t.getProcessor_id(),t.getPeriod_id()))count++;
		}
		return count;
	}
	
	public int valuate3(){
		int penalties = 0;
		int N_WEEKOVER = 402;
		int N_WEEKUNDER = 200;
		int R_WEEKOVER = 300;
		int R_WEEKUNDER = 502;
		int r_employment = 1;
		int r_total_periods_ub =0,r_total_periods_lb=0;
		int n_total_periods_ub =0,n_total_periods_lb=0;
		//各講義の担当講師の人数の上限と下限
		Iterator i = lectures.iterator();
		while(i.hasNext()){
			LectureDetail l = (LectureDetail)i.next();
			int num = getProcessorNum(l.getTask_id());
			if(num>l.getRequired_processors_ub()){
				penalties += (10000*(num-l.getRequired_processors_ub()));
			}
			if(num<l.getRequired_processors_lb()){
				penalties += (10000*(l.getRequired_processors_lb()-num));
			}
		}
		//各講師が講義を担当する回数の上限と下限
		Iterator i2 = teachers.iterator();
		while(i2.hasNext()){
			TeacherDetail t = (TeacherDetail)i2.next();
			int num = getTaskNum(t.getProcessor_id());
			if(num>t.getTotal_periods_ub()){
				if(t.getEmployment()==r_employment){
					r_total_periods_ub +=(R_WEEKOVER*(num-t.getTotal_periods_ub()));
					penalties += (R_WEEKOVER*(num-t.getTotal_periods_ub()));
				}else{
					penalties += (N_WEEKOVER*(num-t.getTotal_periods_ub()));
					n_total_periods_ub += (N_WEEKOVER*(num-t.getTotal_periods_ub()));
				}
			}
			if(num<t.getTotal_periods_lb()){
				if(t.getEmployment()==r_employment){
					penalties += (R_WEEKUNDER*(t.getTotal_periods_lb()-num));
					r_total_periods_lb +=(R_WEEKUNDER*(t.getTotal_periods_lb()-num));
				}else{
					n_total_periods_lb +=(N_WEEKUNDER*(t.getTotal_periods_lb()-num));
					penalties += (N_WEEKUNDER*(t.getTotal_periods_lb()-num));
					//System.out.println("num:"+num);
					//System.out.println("processor_id:"+t.getProcessor_id());
				}
			}
		}
		//System.out.println("上限:"+r_total_periods_ub);
		//System.out.println("上限:"+n_total_periods_ub);
		//System.out.println("下限:"+r_total_periods_lb);
		//System.out.println("下限:"+n_total_periods_lb);
		return penalties;
	}
	public int valuate(){
		int penalties = 0;
		int N_WEEKOVER = 402;
		int N_WEEKUNDER = 200;
		int R_WEEKOVER = 300;
		int R_WEEKUNDER = 502;
		int R_DAYSOVER = 10;
		int N_DAYSOVER = 50;
		int r_employment = 1;
		//非常勤講師の担当している講義数
		int count = 0;
		//各講義の担当講師の人数の上限と下限
		Iterator i = lectures.iterator();
		while(i.hasNext()){
			LectureDetail l = (LectureDetail)i.next();
			int num = getProcessorNum(l.getTask_id());
			if(num>l.getRequired_processors_ub()){
				penalties += (10000*(num-l.getRequired_processors_ub()));
			}
			if(num<l.getRequired_processors_lb()){
				penalties += (10000*(l.getRequired_processors_lb()-num));
			}
		}
		//各講師が講義を担当する回数の上限と下限
		//各講師が講義を担当する日数の上限
		//各講師について、担当する講義の間に空き時間をつくらない
		//非常勤講師全員についての講義担当回数の合計の上限
		//各講師が担当を希望する講義時間に講義を担当する
		Iterator i2 = teachers.iterator();
		while(i2.hasNext()){
			TeacherDetail t = (TeacherDetail)i2.next();
			int num = getTaskNum(t.getProcessor_id());
			if(num>t.getTotal_periods_ub()){
				if(t.getEmployment()==r_employment){
					penalties += (R_WEEKOVER*(num-t.getTotal_periods_ub()));
				}else{
					penalties += (N_WEEKOVER*(num-t.getTotal_periods_ub()));
				}
			}
			if(num<t.getTotal_periods_lb()){
				if(t.getEmployment()==r_employment){
					penalties += (R_WEEKUNDER*(t.getTotal_periods_lb()-num));
				}else{
					penalties += (N_WEEKUNDER*(t.getTotal_periods_lb()-num));
				}
			}
			if(t.getEmployment()!=r_employment){
				count += num;
			}
			num = getProcessorDayNum(t.getProcessor_id());
			if(num>t.getTotal_days_ub()){
				if(t.getEmployment()==r_employment){
					penalties += (R_DAYSOVER*(num-t.getTotal_days_ub()));
				}else{
					penalties += (N_DAYSOVER*(num-t.getTotal_days_ub()));
				}
			}
			penalties += getHolePenalties(t);
			penalties += getProcessorPeriodDesires(t);
		}
		if(40<count){
			penalties += 1000*(count-40);
		}
		//英語で行う講義は英語を流暢に話せる講師が担当する
		//日本語で行う講義は日本語を話せる講師が担当する
		Iterator i3 = timetablerows.iterator();
		while(i3.hasNext()){
			TimeTableRow ttr = (TimeTableRow)i3.next();
			TeacherDetail t = SearchTeacher(ttr.getProcessor_id());
			LectureDetail l = SearchLecture(ttr.getTask_id());
			if(!t.isQualification(l.getQualification_id()))penalties += 1000;
		}
		return penalties;
	}
	//変更予定
	public int getHolePenalties(TeacherDetail t){
		int R_HOLE = 5;
		int N_HOLE = 25;
		int penalties = 0;
		ArrayList list = getDayPeriods(t.getProcessor_id());
		Iterator i = list.iterator();
		while(i.hasNext()){
			DayPeriod dp = (DayPeriod)i.next();
			if(list.contains(new DayPeriod(dp.getDay_id(),dp.getPeriod_id()+2))){
				if(!list.contains(new DayPeriod(dp.getDay_id(),dp.getPeriod_id()+1))){
					//System.out.println("processor_id:"+t.getProcessor_id());
					//System.out.println("day_id:"+dp.getDay_id()+"period_id:"+dp.getPeriod_id());
					if(t.getEmployment()==1){
						penalties += R_HOLE;
					}else{
						penalties += N_HOLE;
					}
				}
			}
			if(list.contains(new DayPeriod(dp.getDay_id(),dp.getPeriod_id()+3))){
				if(!list.contains(new DayPeriod(dp.getDay_id(),dp.getPeriod_id()+1))){
					if(!list.contains(new DayPeriod(dp.getDay_id(),dp.getPeriod_id()+2))){
						//System.out.println("processor_id:"+t.getProcessor_id());
						//System.out.println("day_id:"+dp.getDay_id()+"period_id:"+dp.getPeriod_id());
						if(t.getEmployment()==1){
							penalties += R_HOLE*2;
						}else{
							penalties += N_HOLE*2;
						}
					}
				}
			}
			if(list.contains(new DayPeriod(dp.getDay_id(),dp.getPeriod_id()+4))){
				if(!list.contains(new DayPeriod(dp.getDay_id(),dp.getPeriod_id()+1))){
					if(!list.contains(new DayPeriod(dp.getDay_id(),dp.getPeriod_id()+2))){
						if(!list.contains(new DayPeriod(dp.getDay_id(),dp.getPeriod_id()+3))){
							//System.out.println("processor_id:"+t.getProcessor_id());
							//System.out.println("day_id:"+dp.getDay_id()+"period_id:"+dp.getPeriod_id());
							if(t.getEmployment()==1){
								penalties += R_HOLE*3;
							}else{
								penalties += N_HOLE*3;
							}
						}
					}
				}
			}
		}
		return penalties;
	}
	
//	変更予定
	public int getProcessorPeriodDesires(TeacherDetail t){
		int penalties = 0;
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow ttr = (TimeTableRow)i.next();
			if(t.getProcessor_id()==ttr.getProcessor_id()){
				penalties += t.getPeriodDesire(ttr.getPeriod_id());
			}
		}
		return penalties;
	}
	
	//変更予定
	public ArrayList getDayPeriods(int processor_id){
		ArrayList list = new ArrayList();
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow ttr = (TimeTableRow)i.next();
			if(processor_id==ttr.getProcessor_id()){
				if(ttr.getPeriod_id()<=5){
					list.add(new DayPeriod(1,ttr.getPeriod_id()));
				}else if(ttr.getPeriod_id()<=10){
					list.add(new DayPeriod(2,ttr.getPeriod_id()));
				}else if(ttr.getPeriod_id()<=15){
					list.add(new DayPeriod(3,ttr.getPeriod_id()));
				}else if(ttr.getPeriod_id()<=20){
					list.add(new DayPeriod(4,ttr.getPeriod_id()));
				}else if(ttr.getPeriod_id()<=25){
					list.add(new DayPeriod(5,ttr.getPeriod_id()));
				}
			}
		}
		return list;
	}
	//変更予定
	public int getProcessorNum(int task_id){
		int num = 0;
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow ttr = (TimeTableRow)i.next();
			if(task_id==ttr.getTask_id())num++;
		}
		return num;
	}
	//変更予定
	public int getTaskNum(int processor_id){
		int num = 0;
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow ttr = (TimeTableRow)i.next();
			if(processor_id==ttr.getProcessor_id())num++;
		}
		return num;
	}
	//変更予定
	public int getProcessorDayNum(int processor_id){
		int num = 0;
		int count[] = new int[5];	
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow ttr = (TimeTableRow)i.next();
			if(processor_id==ttr.getProcessor_id()){
				if(ttr.getPeriod_id()<=5){
					count[0]++;
				}else if(ttr.getPeriod_id()<=10){
					count[1]++;
				}else if(ttr.getPeriod_id()<=15){
					count[2]++;
				}else if(ttr.getPeriod_id()<=20){
					count[3]++;
				}else if(ttr.getPeriod_id()<=25){
					count[4]++;
				}
			}
		}
		for(int j = 0; j<5 ; j++){
			if(count[j]!=0)num++;
		}
		return num;
	}
	//制約違反していないかどうか(違反していない場合:true している:false)
	public boolean isNotOffence(){
		Set temp = new TreeSet();
		//●1.各講師は担当可能時間に担当可能な講義を担当
		//2.各講義は、各講義固有の開講可能時間のうち一つに開講される
		//●2-1.各講義は開講可能時間に開講される
		//●2-2.各講義は開講可能であれば必ず開講される
		//▲2-3.各講義はただ一つだけ開講される
		//▲3.各講師は、同じ時間に複数の講義を担当しない
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			TeacherDetail teacher = SearchTeacher(t.getProcessor_id());
			LectureDetail lecture = SearchLecture(t.getTask_id());
			if(!teacher.isPeriod(t.getPeriod_id()))return false;
			if(!teacher.isQualification(lecture.getQualification_id()))return false;
			if(!lecture.isPeriod(t.getPeriod_id()))return false;
			if(!isNotTasks2(t.getProcessor_id(),t.getPeriod_id()))return false;
			if(!isNotTasks3(t.getTask_id(),t.getPeriod_id()))return false;
			temp.add(new Integer(t.getTask_id()));
		}
		if(temp.size()!=lectures.size())return false;
		return true;
	}
	
	//講師processor_idがperiod_idに複数の講義を担当しないかどうか(担当しない:true 担当する:false)	
	public boolean isNotTasks(int processor_id, int period_id){
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			if((t.getProcessor_id()==processor_id)&&(t.getPeriod_id()==period_id)) return false;
		}
		return true;
	}
	
	//講師processor_idがperiod_idに複数の講義を担当しないかどうか(担当しない:true 担当する:false)	
	public boolean isNotTasks2(int processor_id, int period_id){
		int count = 0;
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			if((t.getProcessor_id()==processor_id)&&(t.getPeriod_id()==period_id)){
				if(count==1)return false;
				count++;
			}
		}
		return true;
	}
	
	//現在の時間割中に講義が一つしか開講されていない(開講されていない:true 開講されている:false)
	public boolean isNotTasks3(int task_id, int period_id){
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			if((t.getTask_id()==task_id)&&(t.getPeriod_id()!=period_id)){
				return false;
			}
		}
		return true;
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
