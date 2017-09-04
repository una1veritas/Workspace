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

public class TimeTableValuate implements Valuate{
	private TimeTable timetable;
	private LinkedList timetablerows;
	private ArrayList lectureDetails;
	private ArrayList teacherDetails;
	private DBConnectionPool cmanager;
	int N_ALLNUMOVER = 1000,N_ALLNUM = 40,PROCESSOROVER = 10000,PROCESSORUNDER = 10000;
	int N_WEEKOVER = 402,N_WEEKUNDER = 200,R_WEEKOVER = 300,R_WEEKUNDER = 502;
	int R_DAYSOVER = 10,N_DAYSOVER = 50,R_EMPLOYMENT = 1;
	
	public TimeTableValuate(){
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
			//MySQLƒT[ƒoÚ‘±
			con = cmanager.getConnection();
		} catch (SQLException e) {
			System.err.println("Couldn't get connection: " + e);
			throw e;
		}
		//StatementƒCƒ“ƒ^[ƒtƒF[ƒX‚Ì¶¬
		smt = con.createStatement();
		smt2 = con.createStatement();
		try {
			//uŽt‚Ì‰Šú‰»
			//‘®«‚ª“ü—Í‚³‚ê‚Ä‚¢‚é‘SuŽt‚ðŽæ“¾
			String sql = "select a.processor_id ,b.employment, b.total_periods_lb , b.total_periods_ub,b.total_days_ub from processors a, processor_properties b where a.processor_id = b.processor_id order by a.processor_id";
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
				//processor_id‚ª’S“–‰Â”\‚ÈŽžŠÔ‚ðŽæ“¾
				String sql2 = "select period_id , preferred_level_proc from processor_schedules where processor_id = '" + t.getProcessor_id() + "' order by period_id";
				rs2 = smt2.executeQuery(sql2);
				while(rs2.next()){
					PeriodDesire pd = new PeriodDesire();
					pd.setPeriod_id(rs2.getInt("period_id"));
					pd.setPreferred_level(rs2.getInt("preferred_level_proc"));
					pds.add(pd);
				}
				t.setPeriods(pds);
				//processor_id‚ª’S“–‰Â”\‚Èu‹`‚ÌŽí—Þ‚ðŽæ“¾
				sql2 = "select qualification_id from processor_qualification where processor_id = '" + t.getProcessor_id() + "' order by qualification_id";
				rs2 = smt2.executeQuery(sql2);
				while(rs2.next()){
					qs.add(new Integer(rs2.getInt("qualification_id")));
				}
				t.setQualifications(qs);
				teacherDetails.add(t);
			}
			//u‹`‚Ì‰Šú‰»
			//‘®«‚ª“ü—Í‚³‚ê‚Ä‚¢‚é‘Su‹`‚ÌŽæ“¾
			sql = "select * from tasks a, task_properties b where a.task_id = b.task_id order by a.task_id";
			rs = smt.executeQuery(sql);
			while(rs.next()){
				LectureDetail l = new LectureDetail();
				ArrayList pds = new ArrayList();
				l.setTask_id(rs.getInt("task_id"));
				l.setRequired_processors_lb(rs.getInt("required_processors_lb"));
				l.setRequired_processors_ub(rs.getInt("required_processors_ub"));
				l.setQualification_id(rs.getInt("qualification_id"));
				//task_id‚ªŠJu‰Â”\‚ÈŽžŠÔ‚ðŽæ“¾
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
			sql = "select * from penalties";
			rs = smt.executeQuery(sql);
			while(rs.next()){
				String temp = rs.getString("name");
				if(temp.equals("N_WEEKOVER"))N_WEEKOVER=rs.getInt("weight");
				else if(temp.equals("N_WEEKUNDER")) N_WEEKUNDER = rs.getInt("weight");
				else if(temp.equals("R_WEEKOVER")) R_WEEKOVER = rs.getInt("weight");
				else if(temp.equals("R_WEEKUNDER")) R_WEEKUNDER = rs.getInt("weight");
				else if(temp.equals("R_DAYSOVER")) R_DAYSOVER = rs.getInt("weight");
				else if(temp.equals("N_DAYSOVER")) N_DAYSOVER = rs.getInt("weight");
			}
		}catch(SQLException e) {
			System.err.println(e);
			throw e;
		}finally{
			//ResultSetƒCƒ“ƒ^[ƒtƒF[ƒX‚Ì”jŠü
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// StatementƒCƒ“ƒ^[ƒtƒF[ƒX‚Ì”jŠü
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQLƒT[ƒoØ’f
			if(con!=null){
				cmanager.freeConnection(con);
				con = null;
			}
		}
	}
	
	public int valuate(){
		int penalties = 0;
		//”ñí‹ÎuŽt‚Ì’S“–‚µ‚Ä‚¢‚éu‹`”
		int count = 0;
		//Šeu‹`‚Ì’S“–uŽt‚Ìl”‚ÌãŒÀ‚Æ‰ºŒÀ
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
		//ŠeuŽt‚ªu‹`‚ð’S“–‚·‚é‰ñ”‚ÌãŒÀ‚Æ‰ºŒÀ
		//ŠeuŽt‚ªu‹`‚ð’S“–‚·‚é“ú”‚ÌãŒÀ
		//ŠeuŽt‚É‚Â‚¢‚ÄA’S“–‚·‚éu‹`‚ÌŠÔ‚É‹ó‚«ŽžŠÔ‚ð‚Â‚­‚ç‚È‚¢
		//”ñí‹ÎuŽt‘Sˆõ‚É‚Â‚¢‚Ä‚Ìu‹`’S“–‰ñ”‚Ì‡Œv‚ÌãŒÀ
		//ŠeuŽt‚ª’S“–‚ðŠó–]‚·‚éu‹`ŽžŠÔ‚Éu‹`‚ð’S“–‚·‚é
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
		//‰pŒê‚Ås‚¤u‹`‚Í‰pŒê‚ð—¬’¨‚É˜b‚¹‚éuŽt‚ª’S“–‚·‚é
		//“ú–{Œê‚Ås‚¤u‹`‚Í“ú–{Œê‚ð˜b‚¹‚éuŽt‚ª’S“–‚·‚é
		Iterator i3 = timetablerows.iterator();
		while(i3.hasNext()){
			TimeTableRow ttr = (TimeTableRow)i3.next();
			TeacherDetail t = SearchTeacher(ttr.getProcessor_id());
			LectureDetail l = SearchLecture(ttr.getTask_id());
			if(!t.isQualification(l.getQualification_id()))penalties += 1000;
		}
		return penalties;
	}
	
	//•ÏX—\’è
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
	
	//•ÏX—\’è
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
	//•ÏX—\’è
	public int getProcessorNum(int task_id){
		int num = 0;
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow ttr = (TimeTableRow)i.next();
			if(task_id==ttr.getTask_id())num++;
		}
		return num;
	}
	//•ÏX—\’è
	public int getTaskNum(int processor_id){
		int num = 0;
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow ttr = (TimeTableRow)i.next();
			if(processor_id==ttr.getProcessor_id())num++;
		}
		return num;
	}
	//•ÏX—\’è
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
	
	//•ÏX—\’è
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
	
	//task_id‚Å‚ ‚éu‹`‚ðŽæ“¾‚·‚é
	private LectureDetail SearchLecture(int task_id){
		Iterator i = lectureDetails.iterator();
		while(i.hasNext()){
			LectureDetail l = (LectureDetail)i.next();
			//u‹`l‚ªtask_id‚Å‚ ‚é
			if(l.getTask_id()==task_id){
				return l;
			}
		}
		return null;
	}
	//processor_id‚Å‚ ‚éuŽt‚ðŽæ“¾‚·‚é
	private TeacherDetail SearchTeacher(int processor_id){
		Iterator i = teacherDetails.iterator();
		while(i.hasNext()){
			TeacherDetail t = (TeacherDetail)i.next();
			//uŽtt‚ªprocessor_id‚Å‚ ‚é
			if(t.getProcessor_id()==processor_id){
				return t;
			}
		}
		return null;
	}
	
}
