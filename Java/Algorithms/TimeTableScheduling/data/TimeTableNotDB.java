/*
 * ì¬“ú: 2007/04/02
 *
 * TODO ‚±‚Ì¶¬‚³‚ê‚½ƒtƒ@ƒCƒ‹‚Ìƒeƒ“ƒvƒŒ[ƒg‚ð•ÏX‚·‚é‚É‚ÍŽŸ‚ÖƒWƒƒƒ“ƒv:
 * ƒEƒBƒ“ƒhƒE - Ý’è - Java - ƒR[ƒhEƒXƒ^ƒCƒ‹ - ƒR[ƒhEƒeƒ“ƒvƒŒ[ƒg
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
 * TODO ‚±‚Ì¶¬‚³‚ê‚½Œ^ƒRƒƒ“ƒg‚Ìƒeƒ“ƒvƒŒ[ƒg‚ð•ÏX‚·‚é‚É‚ÍŽŸ‚ÖƒWƒƒƒ“ƒv:
 * ƒEƒBƒ“ƒhƒE - Ý’è - Java - ƒR[ƒhEƒXƒ^ƒCƒ‹ - ƒR[ƒhEƒeƒ“ƒvƒŒ[ƒg
 */
public class TimeTableNotDB {
	//(task_id,processor_id,period_id)‚Ì‘g‚ÌW‡
	private LinkedList timetablerows;
	private DBConnectionPool cmanager;
	//‘SuŽt
	private ArrayList teachers;
	//‘Su‹`
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
	
	//‰Šú‰»
	private void init() throws Exception{
		teachers = new ArrayList();
		lectures = new ArrayList();
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
			//‚±‚ÌƒvƒƒOƒ‰ƒ€ã‚ÅˆêŽž“I‚Éì‚éƒe[ƒuƒ‹‚ªDBã‚É‚·‚Å‚É‘¶Ý‚µ‚Ä‚¢‚éê‡‚»‚Ìƒe[ƒuƒ‹‚ðíœ‚·‚é
			String sql = "drop table if exists t";
			smt.executeUpdate(sql);
			sql = "drop table if exists t_timetable";
			smt.executeUpdate(sql);
			sql = "drop table if exists t_task_relation";
			smt.executeUpdate(sql);
			sql = "drop table if exists t_processor_relation";
			smt.executeUpdate(sql);
			//uŽt‚Ì‰Šú‰»
			//‘®«‚ª“ü—Í‚³‚ê‚Ä‚¢‚é‘SuŽt‚ðŽæ“¾
			sql = "select a.processor_id from processors a, processor_properties b where a.processor_id = b.processor_id order by a.processor_id";
			rs = smt.executeQuery(sql);
			while(rs.next()){
				TeacherDetail t = new TeacherDetail();
				ArrayList pds = new ArrayList();
				ArrayList qs = new ArrayList();
				t.setProcessor_id(rs.getInt("processor_id"));
				//processor_id‚ª’S“–‰Â”\‚ÈŽžŠÔ‚ðŽæ“¾
				String sql2 = "select period_id , preferred_level_proc from processor_schedules where processor_id = '" + t.getProcessor_id() + "' order by period_id";
				rs2 = smt2.executeQuery(sql2);
				while(rs2.next()){
					PeriodDesire pd = new PeriodDesire();
					pd.setPeriod_id(rs2.getInt("period_id"));
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
				teachers.add(t);
			}
			System.out.println("uŽt‚Ì‰Šú‰»I—¹");
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
				lectures.add(l);
			}
			System.out.println("u‹`‚Ì‰Šú‰»I—¹");
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
			//MySQLƒT[ƒoÚ‘±
			con = cmanager.getConnection();
			//StatementƒCƒ“ƒ^[ƒtƒF[ƒX‚Ì¶¬
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
			// StatementƒCƒ“ƒ^[ƒtƒF[ƒX‚Ì”jŠü
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQLƒT[ƒoØ’f
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
			//MySQLƒT[ƒoÚ‘±
			con = cmanager.getConnection();
			//StatementƒCƒ“ƒ^[ƒtƒF[ƒX‚Ì¶¬
			smt = con.createStatement();
			//Œ»Ý‚ÌŽžŠÔŠ„‚ÌŽæ“¾
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
	
	//§–ñˆá”½‚µ‚Ä‚¢‚È‚¢‚©‚Ç‚¤‚©(ˆá”½‚µ‚Ä‚¢‚È‚¢ê‡:true ‚µ‚Ä‚¢‚é:false)
	public int countOffence(){
		Set temp1 = new TreeSet();
		ArrayList temp2 = new ArrayList();
	    ArrayList temp3 = new ArrayList();
		//œ1.ŠeuŽt‚Í’S“–‰Â”\ŽžŠÔ‚É(’S“–‰Â”\‚Èu‹`)‚ð’S“–
		//2.Šeu‹`‚ÍAŠeu‹`ŒÅ—L‚ÌŠJu‰Â”\ŽžŠÔ‚Ì‚¤‚¿ˆê‚Â‚ÉŠJu‚³‚ê‚é
		//œ2-1.Šeu‹`‚ÍŠJu‰Â”\ŽžŠÔ‚ÉŠJu‚³‚ê‚é
		//œ2-2.Šeu‹`‚ÍŠJu‰Â”\‚Å‚ ‚ê‚Î•K‚¸ŠJu‚³‚ê‚é
		//£2-3.Šeu‹`‚Í‚½‚¾ˆê‚Â‚¾‚¯ŠJu‚³‚ê‚é
		//£3.ŠeuŽt‚ÍA“¯‚¶ŽžŠÔ‚É•¡”‚Ìu‹`‚ð’S“–‚µ‚È‚¢
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
		//Šeu‹`‚Ì’S“–uŽt‚Ìl”‚ÌãŒÀ‚Æ‰ºŒÀ
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
		//ŠeuŽt‚ªu‹`‚ð’S“–‚·‚é‰ñ”‚ÌãŒÀ‚Æ‰ºŒÀ
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
		//System.out.println("ãŒÀ:"+r_total_periods_ub);
		//System.out.println("ãŒÀ:"+n_total_periods_ub);
		//System.out.println("‰ºŒÀ:"+r_total_periods_lb);
		//System.out.println("‰ºŒÀ:"+n_total_periods_lb);
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
		//”ñí‹ÎuŽt‚Ì’S“–‚µ‚Ä‚¢‚éu‹`”
		int count = 0;
		//Šeu‹`‚Ì’S“–uŽt‚Ìl”‚ÌãŒÀ‚Æ‰ºŒÀ
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
		//ŠeuŽt‚ªu‹`‚ð’S“–‚·‚é‰ñ”‚ÌãŒÀ‚Æ‰ºŒÀ
		//ŠeuŽt‚ªu‹`‚ð’S“–‚·‚é“ú”‚ÌãŒÀ
		//ŠeuŽt‚É‚Â‚¢‚ÄA’S“–‚·‚éu‹`‚ÌŠÔ‚É‹ó‚«ŽžŠÔ‚ð‚Â‚­‚ç‚È‚¢
		//”ñí‹ÎuŽt‘Sˆõ‚É‚Â‚¢‚Ä‚Ìu‹`’S“–‰ñ”‚Ì‡Œv‚ÌãŒÀ
		//ŠeuŽt‚ª’S“–‚ðŠó–]‚·‚éu‹`ŽžŠÔ‚Éu‹`‚ð’S“–‚·‚é
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
	
//	•ÏX—\’è
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
	//§–ñˆá”½‚µ‚Ä‚¢‚È‚¢‚©‚Ç‚¤‚©(ˆá”½‚µ‚Ä‚¢‚È‚¢ê‡:true ‚µ‚Ä‚¢‚é:false)
	public boolean isNotOffence(){
		Set temp = new TreeSet();
		//œ1.ŠeuŽt‚Í’S“–‰Â”\ŽžŠÔ‚É’S“–‰Â”\‚Èu‹`‚ð’S“–
		//2.Šeu‹`‚ÍAŠeu‹`ŒÅ—L‚ÌŠJu‰Â”\ŽžŠÔ‚Ì‚¤‚¿ˆê‚Â‚ÉŠJu‚³‚ê‚é
		//œ2-1.Šeu‹`‚ÍŠJu‰Â”\ŽžŠÔ‚ÉŠJu‚³‚ê‚é
		//œ2-2.Šeu‹`‚ÍŠJu‰Â”\‚Å‚ ‚ê‚Î•K‚¸ŠJu‚³‚ê‚é
		//£2-3.Šeu‹`‚Í‚½‚¾ˆê‚Â‚¾‚¯ŠJu‚³‚ê‚é
		//£3.ŠeuŽt‚ÍA“¯‚¶ŽžŠÔ‚É•¡”‚Ìu‹`‚ð’S“–‚µ‚È‚¢
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
	
	//uŽtprocessor_id‚ªperiod_id‚É•¡”‚Ìu‹`‚ð’S“–‚µ‚È‚¢‚©‚Ç‚¤‚©(’S“–‚µ‚È‚¢:true ’S“–‚·‚é:false)	
	public boolean isNotTasks(int processor_id, int period_id){
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			if((t.getProcessor_id()==processor_id)&&(t.getPeriod_id()==period_id)) return false;
		}
		return true;
	}
	
	//uŽtprocessor_id‚ªperiod_id‚É•¡”‚Ìu‹`‚ð’S“–‚µ‚È‚¢‚©‚Ç‚¤‚©(’S“–‚µ‚È‚¢:true ’S“–‚·‚é:false)	
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
	
	//Œ»Ý‚ÌŽžŠÔŠ„’†‚Éu‹`‚ªˆê‚Â‚µ‚©ŠJu‚³‚ê‚Ä‚¢‚È‚¢(ŠJu‚³‚ê‚Ä‚¢‚È‚¢:true ŠJu‚³‚ê‚Ä‚¢‚é:false)
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
	
	//task_id‚Å‚ ‚éu‹`‚ðŽæ“¾‚·‚é
	private LectureDetail SearchLecture(int task_id){
		Iterator i = lectures.iterator();
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
		Iterator i = teachers.iterator();
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
