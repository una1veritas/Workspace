package data.valuate;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.sql.*;

import common.DBConnectionPool;
import data.DayPeriod;
import data.PeriodDesire;
import data.TimeTable;
import data.TimeTableRow;
import data.notdb.LectureDetail;
import data.notdb.TeacherDetail;

public class TimeTableValuate2 implements Valuate{
	private TimeTable timetable;
	private LinkedList timetablerows;
	private ArrayList lectureDetails;
	private ArrayList teacherDetails;
	private DBConnectionPool cmanager;
	int project_id = 3;
	int N_ALLNUMOVER = 1000,N_ALLNUM = 40,PROCESSOROVER = 10000,PROCESSORUNDER = 10000;
	int N_WEEKOVER = 402,N_WEEKUNDER = 200,R_WEEKOVER = 300,R_WEEKUNDER = 502;
	int R_DAYSOVER = 10,N_DAYSOVER = 50,R_EMPLOYMENT = 1;
	
	public TimeTableValuate2(){
		System.out.println("aaa");
	}
	
	public void setValuateData(Object obj) throws Exception{
		timetable = (TimeTable)obj;
		timetablerows = timetable.getTimetablerows();
		cmanager = timetable.getCmanager();
		initData();
	}
	
	public void initData() throws Exception{
		teacherDetails = new ArrayList();
		lectureDetails = new ArrayList();
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
			//講師の初期化
			//属性が入力されている全講師を取得
			String sql = "select a.processor_id ,b.employment, b.total_periods_lb , b.total_periods_ub,b.total_days_ub from processors a, processor_properties b where a.processor_id = b.processor_id and project_id = "+project_id+" order by a.processor_id";
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
				teacherDetails.add(t);
			}
			//講義の初期化
			//属性が入力されている全講義の取得
			sql = "select * from tasks a, task_properties b where a.task_id = b.task_id and a.project_id = "+project_id+" order by a.task_id";
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
				lectureDetails.add(l);
			}
			/*
			sql = "select * from penalties";
			rs = smt.executeQuery(sql);
			while(rs.next()){
				String temp = rs.getString("name");
				if(temp.equals("N_WEEKOVER"))N_WEEKOVER=rs.getInt("weight");
				else if(temp.equals("N_WEEKUNDER")) N_WEEKUNDER = rs.getInt("weight");
				else if(temp.equals("R_WEEKOVER")) R_WEEKOVER = rs.getInt("weight");
				else if(temp.equals("R_WEEKUNDER")) R_WEEKUNDER = rs.getInt("weight");
				else if(temp.equals("R_DAYSOVER")) R_DAYSOVER = rs.getInt("weight");
				else if(temp.equals("N_DAYSOVER")) N_DAYSOVER = rs.getInt("weight");;
			}
			*/
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
	
	public int valuate(){
		int penalties = 0;
		//非常勤講師の担当している講義数
		int count = 0;
		//各講義の担当講師の人数の上限と下限
		Iterator i = lectureDetails.iterator();
		while(i.hasNext()){
			LectureDetail l = (LectureDetail)i.next();
			int num = getProcessorNum(l.getTask_id());
			if(num>l.getRequired_processors_ub()){
				penalties += (PROCESSOROVER*(num-l.getRequired_processors_ub()));
			}
			if(num<l.getRequired_processors_lb()){
				penalties += (PROCESSORUNDER*(l.getRequired_processors_lb()-num));
			}
		}
		//各講師が講義を担当する回数の上限と下限
		//各講師が講義を担当する日数の上限
		//各講師について、担当する講義の間に空き時間をつくらない
		//非常勤講師全員についての講義担当回数の合計の上限
		//各講師が担当を希望する講義時間に講義を担当する
		Iterator i2 = teacherDetails.iterator();
		while(i2.hasNext()){
			TeacherDetail t = (TeacherDetail)i2.next();
			int num = getTaskNum(t.getProcessor_id());
			if(num>t.getTotal_periods_ub()){
				if(t.getEmployment()==R_EMPLOYMENT){
					penalties += (R_WEEKOVER*(num-t.getTotal_periods_ub()));
				}else{
					penalties += (N_WEEKOVER*(num-t.getTotal_periods_ub()));
				}
			}
			if(num<t.getTotal_periods_lb()){
				if(t.getEmployment()==R_EMPLOYMENT){
					penalties += (R_WEEKUNDER*(t.getTotal_periods_lb()-num));
				}else{
					penalties += (N_WEEKUNDER*(t.getTotal_periods_lb()-num));
				}
			}
			if(t.getEmployment()!=R_EMPLOYMENT){
				count += num;
			}
			num = getProcessorDayNum(t.getProcessor_id());
			if(num>t.getTotal_days_ub()){
				if(t.getEmployment()==R_EMPLOYMENT){
					penalties += (R_DAYSOVER*(num-t.getTotal_days_ub()));
				}else{
					penalties += (N_DAYSOVER*(num-t.getTotal_days_ub()));
				}
			}
			penalties += getHolePenalties(t);
			penalties += getProcessorPeriodDesires(t);
		}
		if(N_ALLNUM<count){
			penalties += N_ALLNUMOVER*(count-N_ALLNUM);
		}
		/*
		//英語で行う講義は英語を流暢に話せる講師が担当する
		//日本語で行う講義は日本語を話せる講師が担当する
		Iterator i3 = timetablerows.iterator();
		while(i3.hasNext()){
			TimeTableRow ttr = (TimeTableRow)i3.next();
			TeacherDetail t = SearchTeacher(ttr.getProcessor_id());
			LectureDetail l = SearchLecture(ttr.getTask_id());
			if(!t.isQualification(l.getQualification_id()))penalties += 1000;
		}
		*/
		return penalties;
	}
	
	//変更予定
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
	
	//task_idである講義を取得する
	private LectureDetail SearchLecture(int task_id){
		Iterator i = lectureDetails.iterator();
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
		Iterator i = teacherDetails.iterator();
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
