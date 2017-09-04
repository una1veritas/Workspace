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
					pd.setPeriod_id(rs2.getInt("period_id")