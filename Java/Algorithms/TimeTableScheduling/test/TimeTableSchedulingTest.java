package test;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.Random;
import java.util.StringTokenizer;

import org.apache.commons.lang.time.StopWatch;

import common.DBConnectionPool;

import data.CombEnum;
import data.Lecture;
import data.PeriodDesire;
import data.TaskPeriod;
import data.TaskPeriodNum;
import data.Teacher;
import data.TimeTable;
import data.TimeTableRow;
import data.ValuateSQL;

/*
 * �쐬��: 2006/12/11
 *
 * TODO ���̐������ꂽ�t�@�C���̃e���v���[�g��ύX����ɂ͎��փW�����v:
 * �E�B���h�E - �ݒ� - Java - �R�[�h�E�X�^�C�� - �R�[�h�E�e���v���[�g
 */

/**
 * @author masayoshi
 *
 * TODO ���̐������ꂽ�^�R�����g�̃e���v���[�g��ύX����ɂ͎��փW�����v:
 * �E�B���h�E - �ݒ� - Java - �R�[�h�E�X�^�C�� - �R�[�h�E�e���v���[�g
 */
public class TimeTableSchedulingTest {
	//�S�u�t
	private ArrayList teachers;
	//�S�u�`
	private ArrayList lectures;
	//SQL�ɂ��v�]�]����
	private ArrayList v_sqls;
	//SQL�ɂ��v�]�]����(�ꎞ�e�[�u��(t_timetableSQL)�ɑ΂���)
	private ArrayList t_v_sqls;
	//�f�[�^�x�[�X�̃R�l�N�V�����v�[��
	private DBConnectionPool cmanager;
	//�u�`���Ԋ���
	private TimeTable timetable;
	
	//�����p�X�g�b�v�E�H�b�`
	private StopWatch vsw;
	private boolean SWflag = true;
	
	public TimeTableSchedulingTest(String host,String db,String user, String pass, String file, int readType) throws Exception{
		cmanager = new DBConnectionPool("jdbc:mysql://"+host+"/"+db+"?useUnicode=true&characterEncoding=sjis",user,pass);
		init(file,readType);
		timetable = new TimeTable(cmanager,teachers,lectures);
		vsw = new StopWatch();
	}
	
	//�Ǐ��T��(���P�O)
	public int local_search() throws Exception
	{
		//SWflag = true;
		int i;
		for (i=1; i<=50; i++)
		{
			System.out.println(i+"���");
			if (!search_move()&&!search_change()&&!search_swap()&&!search_add()&&!search_leave()){
				System.out.println("����ȏ㖳��");
				break;
			}
		}
		System.out.println("�]���ɂ�����������:"+vsw);
		return i;
	}
	//�Ǐ��T��(���P�P)
	public int local_search1() throws Exception
	{
		//SWflag = true;
		int i;
		for (i=1; i<=50; i++)
		{
			System.out.println(i+"���");
			if (!search_move1()&&!search_change1()&&!search_swap1()&&!search_add1()&&!search_leave1()){
				System.out.println("����ȏ㖳��");
				break;
			}
		}
		System.out.println("�]���ɂ�����������:"+vsw);
		return i;
	}
	
	//�Ǐ��T��(���P�P:���\�b�h�g�p)
	public int local_search2() throws Exception
	{
		SWflag = true;
		int i;
		for (i=1; i<=50; i++)
		{
			System.out.println(i+"���");
			if (!search_move2()&&!search_change2()&&!search_swap2()&&!search_add2()&&!search_leave2()){
				System.out.println("����ȏ㖳��");
				break;
			}
		}
		System.out.println("�]���ɂ�����������:"+vsw);
		return i;
	}
	
	//�Ǐ��T��(���P�P:���\�b�h�g�p)
	//(����p�̃e�[�u�����g�p���Ȃ�)
	public int local_search2b() throws Exception
	{
		SWflag = true;
		int i;
		for (i=1; i<=50; i++)
		{
			System.out.println(i+"���");
			if (!search_move2b()&&!search_change2b()&&!search_swap2b()&&!search_add2b()&&!search_leave2b()){
				System.out.println("����ȏ㖳��");
				break;
			}
		}
		System.out.println("�]���ɂ�����������:"+vsw);
		return i;
	}
	//�Ǐ��T��(���P�P,�Q)
	public int local_search3() throws Exception
	{
		SWflag = true;
		int i;
		for (i=1; i<=50; i++)
		{
			System.out.println(i+"���");
			if (!search_move3()&&!search_change3()&&!search_swap3()&&!search_add3()&&!search_leave3()){
				System.out.println("����ȏ㖳��");
				break;
			}
		}
		System.out.println("�]���ɂ�����������:"+vsw);
		return i;
	}
	//�Ǐ��T��(���P�P,�Q)
	//(�v���O�������̃e�[�u�������������Ȃ�)
	public int local_search3b() throws Exception
	{
		SWflag = true;
		timetable.loadTimeTable();
		int i;
		for (i=1; i<=50; i++)
		{
			System.out.println(i+"���");
			if (!search_move3b()&&!search_change3b()&&!search_swap3b()&&!search_add3b()&&!search_leave3b()){
				System.out.println("����ȏ㖳��");
				break;
			}
		}
		System.out.println("�]���ɂ�����������:"+vsw);
		return i;
	}
	//�Ǐ��T��(���P�P,�Q)
	//(�v���O�������̃e�[�u�������������Ȃ�)
	//(����p�e�[�u�����g�p���Ȃ�)
	public int local_search3c() throws Exception
	{
		//SWflag = true;
		timetable.loadTimeTable();
		int i;
		for (i=1; i<=50; i++)
		{
			System.out.println(i+"���");
			if (!search_move3c()&&!search_change3c()&&!search_swap3c()&&!search_add3c()&&!search_leave3c()){
				System.out.println("����ȏ㖳��");
				break;
			}
		}
		System.out.println("�]���ɂ�����������:"+vsw);
		return i;
	}
	
	//������
	private void init(String file,int readType) throws Exception{
		teachers = new ArrayList();
		lectures = new ArrayList();
		Connection con = null;
		ResultSet rs = null;
		ResultSet rs2 = null;
		Statement smt = null;
		Statement smt2 = null;
		try {
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
		} catch (SQLException e) {
			System.err.println("Couldn't get connection: " + e);
			throw e;
		}
		//Statement�C���^�[�t�F�[�X�̐���
		smt = con.createStatement();
		smt2 = con.createStatement();
		try {
			//���̃v���O������ňꎞ�I�ɍ��e�[�u����DB��ɂ��łɑ��݂��Ă���ꍇ���̃e�[�u�����폜����
			String sql = "drop table if exists t";
			smt.executeUpdate(sql);
			sql = "drop table if exists t_timetable";
			smt.executeUpdate(sql);
			sql = "drop table if exists t_task_relation";
			smt.executeUpdate(sql);
			sql = "drop table if exists t_processor_relation";
			smt.executeUpdate(sql);
			//�u�t�̏�����
			//���������͂���Ă���S�u�t���擾
			sql = "select a.processor_id from processors a, processor_properties b where a.processor_id = b.processor_id order by a.processor_id";
			rs = smt.executeQuery(sql);
			while(rs.next()){
				Teacher t = new Teacher();
				ArrayList pds = new ArrayList();
				ArrayList qs = new ArrayList();
				t.setProcessor_id(rs.getInt("processor_id"));
				//processor_id���S���\�Ȏ��Ԃ��擾
				String sql2 = "select period_id , preferred_level_proc from processor_schedules where processor_id = '" + t.getProcessor_id() + "' order by period_id";
				rs2 = smt2.executeQuery(sql2);
				while(rs2.next()){
					PeriodDesire pd = new PeriodDesire();
					pd.setPeriod_id(rs2.getInt("period_id"));
					pds.add(pd);
				}
				t.setPeriods(pds);
				//processor_id���S���\�ȍu�`�̎�ނ��擾
				sql2 = "select qualification_id from processor_qualification where processor_id = '" + t.getProcessor_id() + "' order by qualification_id";
				rs2 = smt2.executeQuery(sql2);
				while(rs2.next()){
					qs.add(new Integer(rs2.getInt("qualification_id")));
				}
				t.setQualifications(qs);
				teachers.add(t);
			}
			System.out.println("�u�t�̏������I��");
			//�u�`�̏�����
			//���������͂���Ă���S�u�`�̎擾
			sql = "select * from tasks a, task_properties b where a.task_id = b.task_id order by a.task_id";
			rs = smt.executeQuery(sql);
			while(rs.next()){
				Lecture l = new Lecture();
				ArrayList pds = new ArrayList();
				l.setTask_id(rs.getInt("task_id"));
				l.setRequired_processors_lb(rs.getInt("required_processors_lb"));
				l.setRequired_processors_ub(rs.getInt("required_processors_ub"));
				l.setQualification_id(rs.getInt("qualification_id"));
				//task_id���J�u�\�Ȏ��Ԃ��擾
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
			System.out.println("�u�`�̏������I��");
		}catch(SQLException e) {
			System.err.println(e);
			throw e;
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
				con = null;
			}
		}
		if(readType==1)ReaderValuateSQL(file);
		else ReaderValuateSQL2(file);
	}
	
	//�t�@�C������]��SQL���擾����
	private void ReaderValuateSQL(String file) throws Exception{
		//�t�@�C������]��SQL���擾����
		BufferedReader reader = null;
		v_sqls = new ArrayList();
		t_v_sqls = new ArrayList();
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
					String sql = strToken.nextToken().toString();
					v_sql.setSql(sql);
					t_v_sql.setSql(sql.replaceAll("timetableSQL","t_timetable"));
				}
				if(v_sql!=null)v_sqls.add(v_sql);
				if(t_v_sql!=null){
					t_v_sqls.add(t_v_sql);
				}
			}
			reader.close();
		} catch (FileNotFoundException e) {
			System.out.println("�t�@�C����������܂���");
			throw e;
		}
	}
	
	//�t�@�C������]��SQL���擾����(�ԍ��Ȃ�)
	//select���͕K��penalties���擾����
	private void ReaderValuateSQL2(String file) throws Exception{
		BufferedReader reader = null;
		v_sqls = new ArrayList();
		t_v_sqls = new ArrayList();
		try {
			reader = new BufferedReader(new FileReader(file));
			String line;
			while ((line = reader.readLine()) != null) {
				if(!line.equals("")){
					String[] str = line.split(" ");
					ValuateSQL v_sql = new ValuateSQL();
					ValuateSQL t_v_sql = new ValuateSQL();
					if(str[0].equals("select")){
						v_sql.setType(2);
						t_v_sql.setType(2);
					}else{
						v_sql.setType(1);
						t_v_sql.setType(1);
					}
					v_sql.setSql(line);
					t_v_sql.setSql(line.replaceAll("timetableSQL","t_timetable"));
					v_sqls.add(v_sql);
					t_v_sqls.add(t_v_sql);
				}
			}
		} catch (FileNotFoundException e) {
			System.out.println("�t�@�C����������܂���");
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
	
	//�R�l�N�V�����̉��
	public void close() throws Exception{
		cmanager.release();
	}
	
	//�������쐬
	public void initsol() throws Exception{
		ResultSet rs = null;
		ResultSet rs2 = null;
		Connection con = null;
		Statement smt = null;
		Statement smt2 = null;
		//�V�[�h��^���āARandom�N���X�̃C���X�^���X�𐶐�����B
		Random rand = new Random(Calendar.getInstance().getTimeInMillis());
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			smt2 = con.createStatement();
			String sql = "create table t_task_relation as select a.task_id, b.period_id, a.qualification_id, a.required_processors_lb, a.required_processors_ub, c.day_id, b.preferred_level_task from task_properties a, task_opportunities b, period_properties c where a.task_id = b.task_id and b.period_id = c.period_id order by task_id, period_id";
			smt.executeUpdate(sql);
			sql = "create table t_processor_relation as select a.processor_id, c.period_id, b.qualification_id, a.employment, a.total_periods_lb, a.total_periods_ub, a.total_days_lb, a.total_days_ub, a.wage_level, d.day_id, c.preferred_level_proc from processor_properties a, processor_qualification b, processor_schedules c, period_properties d where a.processor_id = b.processor_id and c.period_id = d.period_id and a.processor_id = c.processor_id order by processor_id, period_id";
			smt.executeUpdate(sql);
			//���Ԋ��̏�����
			sql = "delete from timetableSQL";
			smt.executeUpdate(sql);
			while(true){
				//�u�`�Ƃ��̍u�`�J�u�\���Ԃ������_���ɑI��
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
				//���̑���ɂ��e�u�`�́A�e�u�`�ŗL�̊J�u�\���Ԃ̂�����ɊJ�u�����B
				sql="delete from t_task_relation where task_id ='" + task_id+"'";
				smt.executeUpdate(sql);
			}
			sql = "drop table t_task_relation";
			smt.executeUpdate(sql);
			sql = "drop table t_processor_relation";
			smt.executeUpdate(sql);
		}catch(SQLException e) {
			System.err.println("���R�F" + e.toString());
			throw e;
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs2!=null){
				rs2.close();
				rs2 = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt2!=null){
				smt2.close();
				smt2 = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}
	
	//�������쐬
	public void initsol2() throws Exception{
		ResultSet rs = null;
		ResultSet rs2 = null;
		Connection con = null;
		Statement smt = null;
		Statement smt2 = null;
		//�V�[�h��^���āARandom�N���X�̃C���X�^���X�𐶐�����B
		Random rand = new Random(Calendar.getInstance().getTimeInMillis());
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			smt2 = con.createStatement();
			String sql = "create table t_task_relation as select a.task_id, b.period_id, a.qualification_id, a.required_processors_lb, a.required_processors_ub, c.day_id, b.preferred_level_task from task_properties a, task_opportunities b, period_properties c where a.task_id = b.task_id and b.period_id = c.period_id order by task_id, period_id";
			smt.executeUpdate(sql);
			sql = "create table t_processor_relation as select a.processor_id, c.period_id, b.qualification_id, a.employment, a.total_periods_lb, a.total_periods_ub, a.total_days_lb, a.total_days_ub, a.wage_level, d.day_id, c.preferred_level_proc from processor_properties a, processor_qualification b, processor_schedules c, period_properties d where a.processor_id = b.processor_id and c.period_id = d.period_id and a.processor_id = c.processor_id order by processor_id, period_id";
			smt.executeUpdate(sql);
			//���Ԋ��̏�����
			sql = "delete from timetableSQL";
			smt.executeUpdate(sql);
			while(true){
				//�u�`�Ƃ��̍u�`�J�u�\���Ԃ������_���ɑI��
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
				//���̑���ɂ��e�u�`�́A�e�u�`�ŗL�̊J�u�\���Ԃ̂�����ɊJ�u�����B
				sql="delete from t_task_relation where task_id ='" + task_id+"'";
				smt.executeUpdate(sql);
			}
			sql = "drop table t_task_relation";
			smt.executeUpdate(sql);
			sql = "drop table t_processor_relation";
			smt.executeUpdate(sql);
		}catch(SQLException e) {
			System.err.println("���R�F" + e.toString());
			throw e;
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs2!=null){
				rs2.close();
				rs2 = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt2!=null){
				smt2.close();
				smt2 = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}

	//�������쐬
	public void initsolNotRandom(int count) throws Exception{
		ResultSet rs = null;
		ResultSet rs2 = null;
		Connection con = null;
		Statement smt = null;
		Statement smt2 = null;
		//�V�[�h��^���āARandom�N���X�̃C���X�^���X�𐶐�����B
		Random rand = new Random();
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			smt2 = con.createStatement();
			String sql = "create table t_task_relation as select a.task_id, b.period_id, a.qualification_id, a.required_processors_lb, a.required_processors_ub, c.day_id, b.preferred_level_task from task_properties a, task_opportunities b, period_properties c where a.task_id = b.task_id and b.period_id = c.period_id order by task_id, period_id";
			smt.executeUpdate(sql);
			sql = "create table t_processor_relation as select a.processor_id, c.period_id, b.qualification_id, a.employment, a.total_periods_lb, a.total_periods_ub, a.total_days_lb, a.total_days_ub, a.wage_level, d.day_id, c.preferred_level_proc from processor_properties a, processor_qualification b, processor_schedules c, period_properties d where a.processor_id = b.processor_id and c.period_id = d.period_id and a.processor_id = c.processor_id order by processor_id, period_id";
			smt.executeUpdate(sql);
			//���Ԋ��̏�����
			sql = "delete from timetableSQL";
			smt.executeUpdate(sql);
			while(true){
				//�u�`�Ƃ��̍u�`�J�u�\���Ԃ������_���ɑI��
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
				//���̑���ɂ��e�u�`�́A�e�u�`�ŗL�̊J�u�\���Ԃ̂�����ɊJ�u�����B
				sql="delete from t_task_relation where task_id ='" + task_id+"'";
				smt.executeUpdate(sql);
			}
			sql = "drop table t_task_relation";
			smt.executeUpdate(sql);
			sql = "drop table t_processor_relation";
			smt.executeUpdate(sql);
		}catch(SQLException e) {
			System.err.println("���R�F" + e.toString());
			throw e;
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs2!=null){
				rs2.close();
				rs2 = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt2!=null){
				smt2.close();
				smt2 = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		//�����p
		timetable.loadTimeTable();
	}
	
	//�ߖT����move:�u�`���s�����Ԃ�ύX����(���ׂĒ��ׂ�)
	//�ߖTmove��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	//���P�O
	public boolean search_move() throws Exception{
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
		//�l�̕]��
		min_val = valuateTimeTableSQLFile();
		min_off = count_offence("timetableSQL");
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			smt2 = con.createStatement();
			smt3 = con.createStatement();
			smt4 = con.createStatement();
			//����p�̃e�[�u�����쐬
			String sql = "create table t_timetable as select * from timetableSQL";
			smt.executeUpdate(sql);
			//���ݎ��Ԋ��ɓ����Ă���u�`���擾
			sql = "select task_id ,period_id from timetableSQL group by task_id,period_id";
			rs = smt.executeQuery(sql);
			while(rs.next()){ 
				int task_id = rs.getInt("task_id");
				int period_id = rs.getInt("period_id");
				int qualification_id;
				int required_processors_lb;
				int required_processors_ub;
				Lecture l = SearchLecture(task_id);
				qualification_id = l.getQualification_id();
				required_processors_lb = l.getRequired_processors_lb();
				required_processors_ub = l.getRequired_processors_ub();
				//task_id�̊J�u�\���Ԃ̎擾
				Iterator i = l.getPeriods().iterator();
				while(i.hasNext()){
					PeriodDesire lpd = (PeriodDesire)i.next();
					int period_id2 = lpd.getPeriod_id();
					//�u�`�̊J�u���Ԃ����݊J�u����Ă��鎞�ԂłȂ�
					if(period_id!=period_id2){
						//task_id��S�����Ă���u�t������
						ArrayList t_teacher3 = new ArrayList();
						//task_id��S�����Ă���u�t���擾
						String sql2 = "select processor_id from t_timetable where task_id = '" + task_id+"' and period_id ='"+ period_id+"'";
						rs2 = smt2.executeQuery(sql2);
						while(rs2.next()){
							t_teacher3.add(new Integer(rs2.getInt("processor_id")));
						}
						//task_id������
						sql2 = "delete from t_timetable where task_id ='" + task_id+"' and period_id ='"+ period_id+"'";
						smt2.executeUpdate(sql2);
						//�V����task_id��S������u�t
						ArrayList t_teachers2 = new ArrayList();
						Iterator i2 = teachers.iterator();
						while(i2.hasNext()){
							Teacher t = (Teacher)i2.next();
							//�u�tt���u�`��S���ł��邩
							if(t.isQualification(qualification_id)){
								if(t.isPeriod(period_id2)){
									int processor_id = t.getProcessor_id();
									//�u�t�͓������Ԃɕ����̍u�`�����Ȃ����Ƃ𒲂ׂ�
									String sql3 = "select * from t_timetable where processor_id='"+processor_id+"' and period_id = '" + period_id2 +"'";
									rs3 = smt3.executeQuery(sql3);
									if(rs3.next()==false){
										t_teachers2.add(new Integer(processor_id));
									}
								}
							}
						}
						//�u�`.�S���l������ <= z <= �u�`.�S���l�����
						for(int z=required_processors_lb;z<=required_processors_ub;z++){
							//period_id2�ɒS���\�ȍu�t���擾
							ArrayList t_teachers = null;
							Enumeration e = new CombEnum(t_teachers2.toArray(), z);
							while(e.hasMoreElements()) {
								t_teachers = new ArrayList();
								Object[] a = (Object[])e.nextElement();
								for(int num2=0; num2<a.length; num2++){
									int processor_id = ((Integer)a[num2]).intValue();
									//�u�t�̒ǉ�
									String sql4="insert into t_timetable(task_id,period_id , processor_id) values("+ "'" + task_id + "','" +period_id2 + "','" + processor_id+"')";
									smt4.executeUpdate(sql4);
									t_teachers.add(new Integer(processor_id));
								}
								//�l�̕]��
								new_off = count_offence("t_timetable");
								new_val = valuateTTimeTableFile();
								if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
								{
									//�ω������������Ă���
									new_teacher_ids = t_teachers;
									new_task_id = task_id;
									new_period_id = period_id2;
									del_period_id = period_id;
									min_off = new_off;
									min_val = new_val;
									change = true;
								}
								//�ǉ������u�`�̍폜
								String sql3="delete from t_timetable where task_id ='"+task_id+"'";
								smt3.executeUpdate(sql3);
							}
						}
						//������task_id�̒ǉ�
						Iterator i4 = t_teacher3.iterator();
						while(i4.hasNext()){
							sql2="insert into t_timetable(task_id,period_id , processor_id) values("+ "'" + task_id + "','" +period_id + "','" + ((Integer)i4.next()).intValue()+"')";
							smt2.executeUpdate(sql2);
						}
					}
				}
			}
			//����p�e�[�u���̍폜
			sql = "drop table t_timetable";
			smt.executeUpdate(sql);
			//�������P���ꂽ��
			if(change){
				System.out.println("move:task_id:"+new_task_id+"��period_id:"+del_period_id+"����period_id:"+new_period_id+"�Ɉړ�");
				sql="delete from timetableSQL where task_id ='"+new_task_id+"'";
				smt.executeUpdate(sql);
				Iterator i = new_teacher_ids.iterator();
				int processor_id;
				while(i.hasNext()){
					processor_id = ((Integer)i.next()).intValue();
					System.out.println("�S���u�t:"+processor_id);
					sql="insert into timetableSQL(task_id,period_id , processor_id) values("+ "'" + new_task_id + "','" +new_period_id + "','" + processor_id+"')";
					smt.executeUpdate(sql);
				}
			}
		}catch(SQLException e) {
			System.err.println(e);
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs2!=null){
				rs2.close();
				rs2 = null;
			}
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs3!=null){
				rs3.close();
				rs3 = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt2!=null){
				smt2.close();
				smt2 = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt3!=null){
				smt3.close();
				smt3 = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt4!=null){
				smt4.close();
				smt4 = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		return change;
	}

	//�ߖT����move:�u�`���s�����Ԃ�ύX����(���ׂĒ��ׂ�)
	//�ߖTmove��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_move1() throws Exception{
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
		//�l�̕]��
		min_val = valuateTimeTableSQLFile();
		min_off = count_offence2("timetableSQL");
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			smt2 = con.createStatement();
			smt3 = con.createStatement();
			smt4 = con.createStatement();
			//����p�̃e�[�u�����쐬
			String sql = "create table t_timetable as select * from timetableSQL";
			smt.executeUpdate(sql);
			//���ݎ��Ԋ��ɓ����Ă���u�`���擾
			sql = "select task_id ,period_id from timetableSQL group by task_id,period_id";
			rs = smt.executeQuery(sql);
			while(rs.next()){ 
				int task_id = rs.getInt("task_id");
				int period_id = rs.getInt("period_id");
				int qualification_id;
				int required_processors_lb;
				int required_processors_ub;
				Lecture l = SearchLecture(task_id);
				qualification_id = l.getQualification_id();
				required_processors_lb = l.getRequired_processors_lb();
				required_processors_ub = l.getRequired_processors_ub();
				//task_id�̊J�u�\���Ԃ̎擾
				Iterator i = l.getPeriods().iterator();
				while(i.hasNext()){
					PeriodDesire lpd = (PeriodDesire)i.next();
					int period_id2 = lpd.getPeriod_id();
					//�u�`�̊J�u���Ԃ����݊J�u����Ă��鎞�ԂłȂ�
					if(period_id!=period_id2){
						//task_id��S�����Ă���u�t������
						ArrayList t_teacher3 = new ArrayList();
						//task_id��S�����Ă���u�t���擾
						String sql2 = "select processor_id from t_timetable where task_id = '" + task_id+"' and period_id ='"+ period_id+"'";
						rs2 = smt2.executeQuery(sql2);
						while(rs2.next()){
							t_teacher3.add(new Integer(rs2.getInt("processor_id")));
						}
						//task_id������
						sql2 = "delete from t_timetable where task_id ='" + task_id+"' and period_id ='"+ period_id+"'";
						smt2.executeUpdate(sql2);
						//�V����task_id��S������u�t
						ArrayList t_teachers2 = new ArrayList();
						Iterator i2 = teachers.iterator();
						while(i2.hasNext()){
							Teacher t = (Teacher)i2.next();
							//�u�tt���u�`��S���ł��邩
							if(t.isQualification(qualification_id)){
								if(t.isPeriod(period_id2)){
									int processor_id = t.getProcessor_id();
									//�u�t�͓������Ԃɕ����̍u�`�����Ȃ����Ƃ𒲂ׂ�
									String sql3 = "select * from t_timetable where processor_id='"+processor_id+"' and period_id = '" + period_id2 +"'";
									rs3 = smt3.executeQuery(sql3);
									if(rs3.next()==false){
										t_teachers2.add(new Integer(processor_id));
										//System.out.println("�ǉ��\:"+processor_id);
									}
								}
							}
						}
						//�u�`.�S���l������ <= z <= �u�`.�S���l�����
						for(int z=required_processors_lb;z<=required_processors_ub;z++){
							//period_id2�ɒS���\�ȍu�t���擾
							ArrayList t_teachers = null;
							Enumeration e = new CombEnum(t_teachers2.toArray(), z);
							while(e.hasMoreElements()) {
								t_teachers = new ArrayList();
								Object[] a = (Object[])e.nextElement();
								for(int num2=0; num2<a.length; num2++){
									int processor_id = ((Integer)a[num2]).intValue();
									//�u�t�̒ǉ�
									String sql4="insert into t_timetable(task_id,period_id , processor_id) values("+ "'" + task_id + "','" +period_id2 + "','" + processor_id+"')";
									smt4.executeUpdate(sql4);
									t_teachers.add(new Integer(processor_id));
								}
								//�l�̕]��
								new_off = count_offence2("t_timetable");
								new_val = valuateTTimeTableFile();
								if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
								{
									//�ω������������Ă���
									new_teacher_ids = t_teachers;
									new_task_id = task_id;
									new_period_id = period_id2;
									del_period_id = period_id;
									min_off = new_off;
									min_val = new_val;
									change = true;
								}
								//�ǉ������u�`�̍폜
								String sql3="delete from t_timetable where task_id ='"+task_id+"'";
								smt3.executeUpdate(sql3);
							}
						}
						//������task_id�̒ǉ�
						Iterator i4 = t_teacher3.iterator();
						while(i4.hasNext()){
							sql2="insert into t_timetable(task_id,period_id , processor_id) values("+ "'" + task_id + "','" +period_id + "','" + ((Integer)i4.next()).intValue()+"')";
							smt2.executeUpdate(sql2);
						}
					}
				}
			}
			//����p�e�[�u���̍폜
			sql = "drop table t_timetable";
			smt.executeUpdate(sql);
			//�������P���ꂽ��
			if(change){
				System.out.println("move:task_id:"+new_task_id+"��period_id:"+del_period_id+"����period_id:"+new_period_id+"�Ɉړ�");
				sql="delete from timetableSQL where task_id ='"+new_task_id+"'";
				smt.executeUpdate(sql);
				Iterator i = new_teacher_ids.iterator();
				int processor_id;
				while(i.hasNext()){
					processor_id = ((Integer)i.next()).intValue();
					System.out.println("�S���u�t:"+processor_id);
					sql="insert into timetableSQL(task_id,period_id , processor_id) values("+ "'" + new_task_id + "','" +new_period_id + "','" + processor_id+"')";
					smt.executeUpdate(sql);
				}
			}
		}catch(SQLException e) {
			System.err.println(e);
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs2!=null){
				rs2.close();
				rs2 = null;
			}
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs3!=null){
				rs3.close();
				rs3 = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt2!=null){
				smt2.close();
				smt2 = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt3!=null){
				smt3.close();
				smt3 = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt4!=null){
				smt4.close();
				smt4 = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		return change;
	}

	//�ߖT����move:�u�`���s�����Ԃ�ύX����(���ׂĒ��ׂ�)(���\�b�h�g�p)
	//�ߖTmove��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_move2() throws Exception{
		Connection con = null;
		ResultSet rs = null;
		ResultSet rs2 = null;
		Statement smt = null;
		Statement smt2 = null;
		boolean change = false;
		int min_val,min_off,new_val,new_off;
		ArrayList new_teacher_ids = null;
		int new_task_id = 0,new_period_id = 0,del_period_id = 0;
		//�l�̕]��
		min_val = valuateTimeTableSQLFile();
		min_off = count_offence2("timetableSQL");
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			smt2 = con.createStatement();
			//����p�̃e�[�u�����쐬
			createTTimeTable();
			//���ݎ��Ԋ��ɓ����Ă���u�`���擾
			String sql = "select task_id ,period_id from timetableSQL group by task_id,period_id";
			rs = smt.executeQuery(sql);
			while(rs.next()){ 
				int task_id = rs.getInt("task_id");
				int period_id = rs.getInt("period_id");
				int qualification_id;
				int required_processors_lb;
				int required_processors_ub;
				Lecture l = SearchLecture(task_id);
				qualification_id = l.getQualification_id();
				required_processors_lb = l.getRequired_processors_lb();
				required_processors_ub = l.getRequired_processors_ub();
				//task_id�̊J�u�\���Ԃ̎擾
				Iterator i = l.getPeriods().iterator();
				while(i.hasNext()){
					PeriodDesire lpd = (PeriodDesire)i.next();
					int period_id2 = lpd.getPeriod_id();
					//�u�`�̊J�u���Ԃ����݊J�u����Ă��鎞�ԂłȂ�
					if(period_id!=period_id2){
						//task_id��S�����Ă���u�t������
						ArrayList t_teacher3 = new ArrayList();
						//task_id��S�����Ă���u�t���擾
						String sql2 = "select processor_id from t_timetable where task_id = '" + task_id+"' and period_id ='"+ period_id+"'";
						rs2 = smt2.executeQuery(sql2);
						while(rs2.next()){
							t_teacher3.add(new Integer(rs2.getInt("processor_id")));
						}
						//task_id������
						deleteTaskTTimeTable(task_id,period_id);
						//�V����task_id��S������u�t
						ArrayList t_teachers2 = new ArrayList();
						Iterator i2 = teachers.iterator();
						while(i2.hasNext()){
							Teacher t = (Teacher)i2.next();
							//�u�tt���u�`��S���ł��邩
							if(t.isQualification(qualification_id)){
								if(t.isPeriod(period_id2)){
									int processor_id = t.getProcessor_id();
									//�u�t�͓������Ԃɕ����̍u�`�����Ȃ����Ƃ𒲂ׂ�
									if(isNotTasks(processor_id,period_id2)){
										t_teachers2.add(new Integer(processor_id));
										//System.out.println("�ǉ��\:"+processor_id);
									}
								}
							}
						}
						//�u�`.�S���l������ <= z <= �u�`.�S���l�����
						for(int z=required_processors_lb;z<=required_processors_ub;z++){
							//period_id2�ɒS���\�ȍu�t���擾
							ArrayList t_teachers = null;
							Enumeration e = new CombEnum(t_teachers2.toArray(), z);
							while(e.hasMoreElements()) {
								t_teachers = new ArrayList();
								Object[] a = (Object[])e.nextElement();
								for(int num2=0; num2<a.length; num2++){
									int processor_id = ((Integer)a[num2]).intValue();
									//�u�t�̒ǉ�
									insertTTimeTableRow(task_id,period_id2,processor_id);
									t_teachers.add(new Integer(processor_id));
								}
								//�l�̕]��
								new_off = count_offence2("t_timetable");
								new_val = valuateTTimeTableFile();
								if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
								{
									//�ω������������Ă���
									new_teacher_ids = t_teachers;
									new_task_id = task_id;
									new_period_id = period_id2;
									del_period_id = period_id;
									min_off = new_off;
									min_val = new_val;
									change = true;
								}
								//�ǉ������u�`�̍폜
								deleteTaskTTimeTable(task_id);
							}
						}
						//������task_id�̒ǉ�
						Iterator i4 = t_teacher3.iterator();
						while(i4.hasNext()){
							insertTTimeTableRow(task_id,period_id,((Integer)i4.next()).intValue());
						}
					}
				}
			}
			//����p�e�[�u���̍폜
			deleteTTimeTable();
			//�������P���ꂽ��
			if(change){
				//�ύX�̕\��
				System.out.println("move:task_id:"+new_task_id+"��period_id:"+del_period_id+"����period_id:"+new_period_id+"�Ɉړ�");
				deleteTaskTimeTable(new_task_id);
				Iterator i = new_teacher_ids.iterator();
				int processor_id;
				while(i.hasNext()){
					processor_id = ((Integer)i.next()).intValue();
					System.out.println("�S���u�t:"+processor_id);
					insertTimeTableRow(new_task_id,new_period_id,processor_id);
				}
			}
		}catch(SQLException e) {
			System.err.println(e);
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs2!=null){
				rs2.close();
				rs2 = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt2!=null){
				smt2.close();
				smt2 = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		return change;
	}
	//(����p�̃e�[�u�����g�p���Ȃ�)
	//�ߖT����move:�u�`���s�����Ԃ�ύX����(���ׂĒ��ׂ�)(���\�b�h�g�p)
	//�ߖTmove��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_move2b() throws Exception{
		Connection con = null;
		ResultSet rs = null;
		ResultSet rs2 = null;
		Statement smt = null;
		Statement smt2 = null;
		boolean change = false;
		int min_val,min_off,new_val,new_off;
		ArrayList new_teacher_ids = null;
		int new_task_id = 0,new_period_id = 0,del_period_id = 0;
		//�l�̕]��
		min_val = valuateTimeTableSQLFile();
		min_off = count_offence2("timetableSQL");
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			smt2 = con.createStatement();
			//���ݎ��Ԋ��ɓ����Ă���u�`���擾
			String sql = "select task_id ,period_id from timetableSQL group by task_id,period_id";
			rs = smt.executeQuery(sql);
			while(rs.next()){ 
				int task_id = rs.getInt("task_id");
				int period_id = rs.getInt("period_id");
				int qualification_id;
				int required_processors_lb;
				int required_processors_ub;
				Lecture l = SearchLecture(task_id);
				qualification_id = l.getQualification_id();
				required_processors_lb = l.getRequired_processors_lb();
				required_processors_ub = l.getRequired_processors_ub();
				//task_id�̊J�u�\���Ԃ̎擾
				Iterator i = l.getPeriods().iterator();
				while(i.hasNext()){
					PeriodDesire lpd = (PeriodDesire)i.next();
					int period_id2 = lpd.getPeriod_id();
					//�u�`�̊J�u���Ԃ����݊J�u����Ă��鎞�ԂłȂ�
					if(period_id!=period_id2){
						//task_id��S�����Ă���u�t������
						ArrayList t_teacher3 = new ArrayList();
						//task_id��S�����Ă���u�t���擾
						String sql2 = "select processor_id from timetableSQL where task_id = '" + task_id+"' and period_id ='"+ period_id+"'";
						rs2 = smt2.executeQuery(sql2);
						while(rs2.next()){
							t_teacher3.add(new Integer(rs2.getInt("processor_id")));
						}
						//task_id������
						deleteTaskTimeTable(task_id,period_id);
						//�V����task_id��S������u�t
						ArrayList t_teachers2 = new ArrayList();
						Iterator i2 = teachers.iterator();
						while(i2.hasNext()){
							Teacher t = (Teacher)i2.next();
							//�u�tt���u�`��S���ł��邩
							if(t.isQualification(qualification_id)){
								if(t.isPeriod(period_id2)){
									int processor_id = t.getProcessor_id();
									//�u�t�͓������Ԃɕ����̍u�`�����Ȃ����Ƃ𒲂ׂ�
									if(isNotTasksb(processor_id,period_id2)){
										t_teachers2.add(new Integer(processor_id));
										//System.out.println("�ǉ��\:"+processor_id);
									}
								}
							}
						}
						//�u�`.�S���l������ <= z <= �u�`.�S���l�����
						for(int z=required_processors_lb;z<=required_processors_ub;z++){
							//period_id2�ɒS���\�ȍu�t���擾
							ArrayList t_teachers = null;
							Enumeration e = new CombEnum(t_teachers2.toArray(), z);
							while(e.hasMoreElements()) {
								t_teachers = new ArrayList();
								Object[] a = (Object[])e.nextElement();
								for(int num2=0; num2<a.length; num2++){
									int processor_id = ((Integer)a[num2]).intValue();
									//�u�t�̒ǉ�
									insertTimeTableRow(task_id,period_id2,processor_id);
									t_teachers.add(new Integer(processor_id));
								}
								//�l�̕]��
								new_off = count_offence2("timetableSQL");
								new_val = valuateTimeTableSQLFile();
								if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
								{
									//�ω������������Ă���
									new_teacher_ids = t_teachers;
									new_task_id = task_id;
									new_period_id = period_id2;
									del_period_id = period_id;
									min_off = new_off;
									min_val = new_val;
									change = true;
								}
								//�ǉ������u�`�̍폜
								deleteTaskTimeTable(task_id);
							}
						}
						//������task_id�̒ǉ�
						Iterator i4 = t_teacher3.iterator();
						while(i4.hasNext()){
							insertTimeTableRow(task_id,period_id,((Integer)i4.next()).intValue());
						}
					}
				}
			}
			//�������P���ꂽ��
			if(change){
				//�ύX�̕\��
				System.out.println("move:task_id:"+new_task_id+"��period_id:"+del_period_id+"����period_id:"+new_period_id+"�Ɉړ�");
				deleteTaskTimeTable(new_task_id);
				Iterator i = new_teacher_ids.iterator();
				int processor_id;
				while(i.hasNext()){
					processor_id = ((Integer)i.next()).intValue();
					System.out.println("�S���u�t:"+processor_id);
					insertTimeTableRow(new_task_id,new_period_id,processor_id);
				}
			}
		}catch(SQLException e) {
			System.err.println(e);
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs2!=null){
				rs2.close();
				rs2 = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt2!=null){
				smt2.close();
				smt2 = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		return change;
	}
	
	//�ߖT����move:�u�`���s�����Ԃ�ύX����(���ׂĒ��ׂ�)(���\�b�h�g�p)
	//�ߖTmove��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_move3() throws Exception{
		boolean change = false;
		int min_val,min_off,new_val,new_off;
		ArrayList new_teacher_ids = null;
		int new_task_id = 0,new_period_id = 0,del_period_id = 0;
		timetable.loadTimeTable();
		//�l�̕]��
		min_val = valuateTimeTableSQLFile();
		min_off = timetable.countOffence();
		//����p�̃e�[�u�����쐬
		createTTimeTable();
		//���ݎ��Ԋ��ɓ����Ă���u�`���擾
		ArrayList tps = timetable.getTaskPeriods();
		Iterator i_tps = tps.iterator();
		while(i_tps.hasNext()) {			
			TaskPeriod tp = (TaskPeriod)i_tps.next();
			int task_id = tp.getTask_id();
			int period_id = tp.getPeriod_id();
			int qualification_id;
			int required_processors_lb;
			int required_processors_ub;
			Lecture l = SearchLecture(task_id);
			qualification_id = l.getQualification_id();
			required_processors_lb = l.getRequired_processors_lb();
			required_processors_ub = l.getRequired_processors_ub();
			//task_id�̊J�u�\���Ԃ̎擾
			Iterator i = l.getPeriods().iterator();
			while(i.hasNext()){
				PeriodDesire lpd = (PeriodDesire)i.next();
				int period_id2 = lpd.getPeriod_id();
				//�u�`�̊J�u���Ԃ����݊J�u����Ă��鎞�ԂłȂ�
				if(period_id!=period_id2){
					//task_id��S�����Ă���u�t������
					//period_id��task_id��S�����Ă���u�t���擾
					ArrayList t_teacher3 = timetable.getProcessors(task_id,period_id);
					//task_id������
					deleteTaskTTimeTable(task_id,period_id);
					timetable.deleteTaskTimeTable(task_id,period_id);
					//�V����task_id��S������u�t
					ArrayList t_teachers2 = new ArrayList();
					Iterator i2 = teachers.iterator();
					while(i2.hasNext()){
						Teacher t = (Teacher)i2.next();
						//�u�tt���u�`��S���ł��邩
						if(t.isQualification(qualification_id)){
							if(t.isPeriod(period_id2)){
								int processor_id = t.getProcessor_id();
								//�u�t�͓������Ԃɕ����̍u�`�����Ȃ����Ƃ𒲂ׂ�
								if(timetable.isNotTasks(processor_id,period_id2)){
									t_teachers2.add(new Integer(processor_id));
								}
							}
						}
					}
					//�u�`.�S���l������ <= z <= �u�`.�S���l�����
					for(int z=required_processors_lb;z<=required_processors_ub;z++){
						//period_id2�ɒS���\�ȍu�t���擾
						ArrayList t_teachers = null;
						Enumeration e = new CombEnum(t_teachers2.toArray(), z);
						while(e.hasMoreElements()) {
							t_teachers = new ArrayList();
							Object[] a = (Object[])e.nextElement();
							for(int num2=0; num2<a.length; num2++){
								int processor_id = ((Integer)a[num2]).intValue();
								//�u�t�̒ǉ�
								insertTTimeTableRow(task_id,period_id2,processor_id);
								timetable.insertTimeTableRow(task_id,period_id2,processor_id);
								t_teachers.add(new Integer(processor_id));
							}
							//�l�̕]��
							new_off = timetable.countOffence();
							new_val = valuateTTimeTableFile();
							if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
							{
								//�ω������������Ă���
								new_teacher_ids = t_teachers;
								new_task_id = task_id;
								new_period_id = period_id2;
								del_period_id = period_id;
								min_off = new_off;
								min_val = new_val;
								change = true;
							}
							//�ǉ������u�`�̍폜
							deleteTaskTTimeTable(task_id);
							timetable.deleteTaskTimeTable(task_id);
						}
					}
					//������task_id�̒ǉ�
					Iterator i4 = t_teacher3.iterator();
					while(i4.hasNext()){
						insertTTimeTableRow(task_id,period_id,((Integer)i4.next()).intValue());
						timetable.insertTimeTableRow(task_id,period_id,((Integer)i4.next()).intValue());
					}
				}
			}
		}
		//����p�e�[�u���̍폜
		deleteTTimeTable();
		//�������P���ꂽ��
		if(change){
			//�ύX�̕\��
			System.out.println("move:task_id:"+new_task_id+"��period_id:"+del_period_id+"����period_id:"+new_period_id+"�Ɉړ�");
			deleteTaskTimeTable(new_task_id);
			//timetable.deleteTaskTimeTable(new_task_id);
			Iterator i = new_teacher_ids.iterator();
			int processor_id;
			while(i.hasNext()){
				processor_id = ((Integer)i.next()).intValue();
				System.out.println("�S���u�t:"+processor_id);
				insertTimeTableRow(new_task_id,new_period_id,processor_id);
				//timetable.insertTimeTableRow(new_task_id,new_period_id,processor_id);
			}
		}
		return change;
	}
	//(�v���O�������̃e�[�u�������������Ȃ�)
	//�ߖT����move:�u�`���s�����Ԃ�ύX����(���ׂĒ��ׂ�)(���\�b�h�g�p)
	//�ߖTmove��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_move3b() throws Exception{
		boolean change = false;
		int min_val,min_off,new_val,new_off;
		ArrayList new_teacher_ids = null;
		int new_task_id = 0,new_period_id = 0,del_period_id = 0;
		//�l�̕]��
		min_val = valuateTimeTableSQLFile();
		min_off = timetable.countOffence();
		//����p�̃e�[�u�����쐬
		createTTimeTable();
		//���ݎ��Ԋ��ɓ����Ă���u�`���擾
		ArrayList tps = timetable.getTaskPeriods();
		Iterator i_tps = tps.iterator();
		while(i_tps.hasNext()) {			
			TaskPeriod tp = (TaskPeriod)i_tps.next();
			int task_id = tp.getTask_id();
			int period_id = tp.getPeriod_id();
			int qualification_id;
			int required_processors_lb;
			int required_processors_ub;
			Lecture l = SearchLecture(task_id);
			qualification_id = l.getQualification_id();
			required_processors_lb = l.getRequired_processors_lb();
			required_processors_ub = l.getRequired_processors_ub();
			//task_id�̊J�u�\���Ԃ̎擾
			Iterator i = l.getPeriods().iterator();
			while(i.hasNext()){
				PeriodDesire lpd = (PeriodDesire)i.next();
				int period_id2 = lpd.getPeriod_id();
				//�u�`�̊J�u���Ԃ����݊J�u����Ă��鎞�ԂłȂ�
				if(period_id!=period_id2){
					//task_id��S�����Ă���u�t������
					//period_id��task_id��S�����Ă���u�t���擾
					ArrayList t_teacher3 = timetable.getProcessors(task_id,period_id);
					//task_id������
					deleteTaskTTimeTable(task_id,period_id);
					timetable.deleteTaskTimeTable(task_id,period_id);
					//�V����task_id��S������u�t
					ArrayList t_teachers2 = new ArrayList();
					Iterator i2 = teachers.iterator();
					while(i2.hasNext()){
						Teacher t = (Teacher)i2.next();
						//�u�tt���u�`��S���ł��邩
						if(t.isQualification(qualification_id)){
							if(t.isPeriod(period_id2)){
								int processor_id = t.getProcessor_id();
								//�u�t�͓������Ԃɕ����̍u�`�����Ȃ����Ƃ𒲂ׂ�
								if(timetable.isNotTasks(processor_id,period_id2)){
									t_teachers2.add(new Integer(processor_id));
									//System.out.println("�ǉ��\:"+processor_id);
								}
							}
						}
					}
					//�u�`.�S���l������ <= z <= �u�`.�S���l�����
					for(int z=required_processors_lb;z<=required_processors_ub;z++){
						//period_id2�ɒS���\�ȍu�t���擾
						ArrayList t_teachers = null;
						Enumeration e = new CombEnum(t_teachers2.toArray(), z);
						while(e.hasMoreElements()) {
							t_teachers = new ArrayList();
							Object[] a = (Object[])e.nextElement();
							for(int num2=0; num2<a.length; num2++){
								int processor_id = ((Integer)a[num2]).intValue();
								//�u�t�̒ǉ�
								insertTTimeTableRow(task_id,period_id2,processor_id);
								timetable.insertTimeTableRow(task_id,period_id2,processor_id);
								t_teachers.add(new Integer(processor_id));
							}
							//�l�̕]��
							new_off = timetable.countOffence();
							new_val = valuateTTimeTableFile();
							if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
							{
								//�ω������������Ă���
								new_teacher_ids = t_teachers;
								new_task_id = task_id;
								new_period_id = period_id2;
								del_period_id = period_id;
								min_off = new_off;
								min_val = new_val;
								change = true;
							}
							//�ǉ������u�`�̍폜
							deleteTaskTTimeTable(task_id);
							timetable.deleteTaskTimeTable(task_id);
						}
					}
					//������task_id�̒ǉ�
					Iterator i4 = t_teacher3.iterator();
					while(i4.hasNext()){
						insertTTimeTableRow(task_id,period_id,((Integer)i4.next()).intValue());
						timetable.insertTimeTableRow(task_id,period_id,((Integer)i4.next()).intValue());
					}
				}
			}
		}
		//����p�e�[�u���̍폜
		deleteTTimeTable();
		//�������P���ꂽ��
		if(change){
			//�ύX�̕\��
			System.out.println("move:task_id:"+new_task_id+"��period_id:"+del_period_id+"����period_id:"+new_period_id+"�Ɉړ�");
			deleteTaskTimeTable(new_task_id);
			timetable.deleteTaskTimeTable(new_task_id);
			Iterator i = new_teacher_ids.iterator();
			int processor_id;
			while(i.hasNext()){
				processor_id = ((Integer)i.next()).intValue();
				System.out.println("�S���u�t:"+processor_id);
				insertTimeTableRow(new_task_id,new_period_id,processor_id);
				timetable.insertTimeTableRow(new_task_id,new_period_id,processor_id);
			}
		}
		return change;
	}

	//(�v���O�������̃e�[�u�������������Ȃ�)
	//�ߖT����move:�u�`���s�����Ԃ�ύX����(���ׂĒ��ׂ�)(���\�b�h�g�p)
	//�ߖTmove��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_move3c() throws Exception{
		boolean change = false;
		int min_val,min_off,new_val,new_off;
		ArrayList new_teacher_ids = null;
		int new_task_id = 0,new_period_id = 0,del_period_id = 0;
		//�l�̕]��
		min_val = valuateTimeTableSQLFile();
		min_off = timetable.countOffence();
		//���ݎ��Ԋ��ɓ����Ă���u�`���擾
		ArrayList tps = timetable.getTaskPeriods();
		Iterator i_tps = tps.iterator();
		while(i_tps.hasNext()) {			
			TaskPeriod tp = (TaskPeriod)i_tps.next();
			int task_id = tp.getTask_id();
			int period_id = tp.getPeriod_id();
			int qualification_id;
			int required_processors_lb;
			int required_processors_ub;
			Lecture l = SearchLecture(task_id);
			qualification_id = l.getQualification_id();
			required_processors_lb = l.getRequired_processors_lb();
			required_processors_ub = l.getRequired_processors_ub();
			//task_id�̊J�u�\���Ԃ̎擾
			Iterator i = l.getPeriods().iterator();
			while(i.hasNext()){
				PeriodDesire lpd = (PeriodDesire)i.next();
				int period_id2 = lpd.getPeriod_id();
				//�u�`�̊J�u���Ԃ����݊J�u����Ă��鎞�ԂłȂ�
				if(period_id!=period_id2){
					//task_id��S�����Ă���u�t������
					//period_id��task_id��S�����Ă���u�t���擾
					ArrayList t_teacher3 = timetable.getProcessors(task_id,period_id);
					//task_id������
					deleteTaskTimeTable(task_id,period_id);
					timetable.deleteTaskTimeTable(task_id,period_id);
					//�V����task_id��S������u�t
					ArrayList t_teachers2 = new ArrayList();
					Iterator i2 = teachers.iterator();
					while(i2.hasNext()){
						Teacher t = (Teacher)i2.next();
						//�u�tt���u�`��S���ł��邩
						if(t.isQualification(qualification_id)){
							if(t.isPeriod(period_id2)){
								int processor_id = t.getProcessor_id();
								//�u�t�͓������Ԃɕ����̍u�`�����Ȃ����Ƃ𒲂ׂ�
								if(timetable.isNotTasks(processor_id,period_id2)){
									t_teachers2.add(new Integer(processor_id));
									//System.out.println("�ǉ��\:"+processor_id);
								}
							}
						}
					}
					//�u�`.�S���l������ <= z <= �u�`.�S���l�����
					for(int z=required_processors_lb;z<=required_processors_ub;z++){
						//period_id2�ɒS���\�ȍu�t���擾
						ArrayList t_teachers = null;
						Enumeration e = new CombEnum(t_teachers2.toArray(), z);
						while(e.hasMoreElements()) {
							t_teachers = new ArrayList();
							Object[] a = (Object[])e.nextElement();
							for(int num2=0; num2<a.length; num2++){
								int processor_id = ((Integer)a[num2]).intValue();
								//�u�t�̒ǉ�
								insertTimeTableRow(task_id,period_id2,processor_id);
								timetable.insertTimeTableRow(task_id,period_id2,processor_id);
								t_teachers.add(new Integer(processor_id));
							}
							//�l�̕]��
							new_off = timetable.countOffence();
							new_val = valuateTimeTableSQLFile();
							if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
							{
								//�ω������������Ă���
								new_teacher_ids = t_teachers;
								new_task_id = task_id;
								new_period_id = period_id2;
								del_period_id = period_id;
								min_off = new_off;
								min_val = new_val;
								change = true;
							}
							//�ǉ������u�`�̍폜
							deleteTaskTimeTable(task_id);
							timetable.deleteTaskTimeTable(task_id);
						}
					}
					//������task_id�̒ǉ�
					Iterator i4 = t_teacher3.iterator();
					while(i4.hasNext()){
						insertTimeTableRow(task_id,period_id,((Integer)i4.next()).intValue());
						timetable.insertTimeTableRow(task_id,period_id,((Integer)i4.next()).intValue());
					}
				}
			}
		}
		//�������P���ꂽ��
		if(change){
			//�ύX�̕\��
			System.out.println("move:task_id:"+new_task_id+"��period_id:"+del_period_id+"����period_id:"+new_period_id+"�Ɉړ�");
			timetable.deleteTaskTimeTable(new_task_id);
			Iterator i = new_teacher_ids.iterator();
			int processor_id;
			while(i.hasNext()){
				processor_id = ((Integer)i.next()).intValue();
				System.out.println("�S���u�t:"+processor_id);
				insertTimeTableRow(new_task_id,new_period_id,processor_id);
				timetable.insertTimeTableRow(new_task_id,new_period_id,processor_id);
			}
		}
		return change;
	}

	//�ߖT����Change:�u�`�̒S���u�t��ύX����
	//�ߖTchange��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_change() throws Exception{
		Connection con = null;
		ResultSet rs = null;
		ResultSet rs2 = null;
		Statement smt = null;
		Statement smt2 = null;
		Statement smt3 = null;
		boolean change = false;
		int min_val,min_off,new_val,new_off;
		int del_processor_id = 0,new_task_id = 0,new_processor_id = 0,new_period_id = 0;
		//�l�̕]��
		//min_val = valuate("timetableSQL");
		min_val = valuateTimeTableSQLFile();
		min_off = count_offence("timetableSQL");
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			smt2 = con.createStatement();
			smt3 = con.createStatement();
			//����p�̃e�[�u�����쐬
			String sql = "create table t_timetable as select * from timetableSQL";
			smt.executeUpdate(sql);
			//���݂̎��Ԋ��̎擾
			sql = "select * from timetableSQL";
			rs = smt.executeQuery(sql);
			int task_id;
			int period_id;
			int processor_id;
			int qualification_id = 0;
			while(rs.next()) {
				task_id = rs.getInt("task_id");
				period_id = rs.getInt("period_id");
				processor_id = rs.getInt("processor_id");
				//(task,period,processor)����菜��
				String sql2 = "delete from t_timetable where processor_id ='" + processor_id+"' and period_id = '" + period_id+"' and task_id = '"+task_id +"'";
				smt2.executeUpdate(sql2);
				//task_id�̍u�`��ނ𒲂ׂ�
				Lecture l = SearchLecture(task_id);
				qualification_id = l.getQualification_id();
				//period_id�ɒS���\�ȍu�t�ɑ΂��ĕύX��������
				Iterator i = teachers.iterator();
				while(i.hasNext()){
					Teacher t = (Teacher)i.next();
					//���݂̍u�t�Ɠ����u�t�łȂ�
					if(t.getProcessor_id()!=processor_id){
						//�u�tt�͍u�`�̎��qualification_id��S���ł���
						if(t.isQualification(qualification_id)){
							//�u�tt��period_id�ŒS���ł���
							if(t.isPeriod(period_id)){
								//System.out.println("processor_id:"+t.getProcessor_id());
								int processor_id2 = t.getProcessor_id();
								//�u�t�͓������Ԃɕ����̍u�`�����Ȃ�
								sql2 = "select * from t_timetable where processor_id='"+processor_id2+"' and period_id = '" + period_id +"'";
								rs2 = smt2.executeQuery(sql2);
								if(rs2.next()==false){
									//�u�t�̒ǉ�
									String sql3="insert into t_timetable(task_id,period_id,processor_id) values('" + task_id + "','" +period_id + "','" + processor_id2+"')";
									smt3.executeUpdate(sql3);
									new_off = count_offence("t_timetable");
									new_val = valuateTTimeTableFile();
									if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
									{
										//�ω������������Ă���
										new_task_id = task_id;
										new_processor_id = processor_id2;
										del_processor_id = processor_id;
										new_period_id = period_id;
										min_off = new_off;
										min_val = new_val;
										change = true;
									}
									//�ǉ������u�t�̍폜
									sql3="delete from t_timetable where processor_id ='" + processor_id2+"' and period_id = '" + period_id+"' and task_id = '"+task_id +"'";
									smt3.executeUpdate(sql3);
								}
							}
						}
					}
				}
				//��菜�����u�t�̒ǉ�
				sql2 ="insert into t_timetable(task_id,period_id,processor_id) values('" + task_id + "','" +period_id + "','" + processor_id+"')";
				smt2.executeUpdate(sql2);
			}
			//����p�e�[�u���̍폜
			sql = "drop table t_timetable";
			smt.executeUpdate(sql);
			//�������P���ꂽ��
			if(change){
				System.out.println("task_id:"+new_task_id+" period_id:"+new_period_id);
				System.out.println("change:processor_id:"+del_processor_id+"��processor_id:"+new_processor_id+"�ɕω�");
				sql = "delete from timetableSQL where processor_id ='" + del_processor_id+"' and period_id = '" + new_period_id+"' and task_id = '"+new_task_id +"'";
				smt.executeUpdate(sql);
				sql = "insert into timetableSQL(task_id,period_id , processor_id) values("+ "'" + new_task_id + "','" +new_period_id + "','" + new_processor_id+"')";
				smt.executeUpdate(sql);
			}
		}catch(SQLException e) {
			System.err.println(e);
			throw e;
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs2!=null){
				rs2.close();
				rs2 = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt2!=null){
				smt2.close();
				smt2 = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt3!=null){
				smt3.close();
				smt3 = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		return change;
	}

	//�ߖT����Change:�u�`�̒S���u�t��ύX����
	//�ߖTchange��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_change1() throws Exception{
		Connection con = null;
		ResultSet rs = null;
		ResultSet rs2 = null;
		Statement smt = null;
		Statement smt2 = null;
		Statement smt3 = null;
		boolean change = false;
		int min_val,min_off,new_val,new_off;
		int del_processor_id = 0,new_task_id = 0,new_processor_id = 0,new_period_id = 0;
		//�l�̕]��
		//min_val = valuate("timetableSQL");
		min_val = valuateTimeTableSQLFile();
		min_off = count_offence2("timetableSQL");
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			smt2 = con.createStatement();
			smt3 = con.createStatement();
			//����p�̃e�[�u�����쐬
			String sql = "create table t_timetable as select * from timetableSQL";
			smt.executeUpdate(sql);
			//���݂̎��Ԋ��̎擾
			sql = "select * from timetableSQL";
			rs = smt.executeQuery(sql);
			int task_id;
			int period_id;
			int processor_id;
			int qualification_id = 0;
			while(rs.next()) {
				task_id = rs.getInt("task_id");
				period_id = rs.getInt("period_id");
				processor_id = rs.getInt("processor_id");
				//(task,period,processor)����菜��
				String sql2 = "delete from t_timetable where processor_id ='" + processor_id+"' and period_id = '" + period_id+"' and task_id = '"+task_id +"'";
				smt2.executeUpdate(sql2);
				//task_id�̍u�`��ނ𒲂ׂ�
				Lecture l = SearchLecture(task_id);
				qualification_id = l.getQualification_id();
				//period_id�ɒS���\�ȍu�t�ɑ΂��ĕύX��������
				Iterator i = teachers.iterator();
				while(i.hasNext()){
					Teacher t = (Teacher)i.next();
					//���݂̍u�t�Ɠ����u�t�łȂ�
					if(t.getProcessor_id()!=processor_id){
						//�u�tt�͍u�`�̎��qualification_id��S���ł���
						if(t.isQualification(qualification_id)){
							//�u�tt��period_id�ŒS���ł���
							if(t.isPeriod(period_id)){
								//System.out.println("processor_id:"+t.getProcessor_id());
								int processor_id2 = t.getProcessor_id();
								//�u�t�͓������Ԃɕ����̍u�`�����Ȃ�
								sql2 = "select * from t_timetable where processor_id='"+processor_id2+"' and period_id = '" + period_id +"'";
								rs2 = smt2.executeQuery(sql2);
								if(rs2.next()==false){
									//�u�t�̒ǉ�
									String sql3="insert into t_timetable(task_id,period_id,processor_id) values('" + task_id + "','" +period_id + "','" + processor_id2+"')";
									smt3.executeUpdate(sql3);
									//new_off = count_offence("t_timetable");
									new_off = count_offence2("t_timetable");
									//new_val = valuate("t_timetable");
									new_val = valuateTTimeTableFile();
									if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
									{
										//�ω������������Ă���
										new_task_id = task_id;
										new_processor_id = processor_id2;
										del_processor_id = processor_id;
										new_period_id = period_id;
										min_off = new_off;
										min_val = new_val;
										change = true;
									}
									//�ǉ������u�t�̍폜
									sql3="delete from t_timetable where processor_id ='" + processor_id2+"' and period_id = '" + period_id+"' and task_id = '"+task_id +"'";
									smt3.executeUpdate(sql3);
								}
							}
						}
					}
				}
				//��菜�����u�t�̒ǉ�
				sql2 ="insert into t_timetable(task_id,period_id,processor_id) values('" + task_id + "','" +period_id + "','" + processor_id+"')";
				smt2.executeUpdate(sql2);
			}
			//����p�e�[�u���̍폜
			sql = "drop table t_timetable";
			smt.executeUpdate(sql);
			//�������P���ꂽ��
			if(change){
				System.out.println("task_id:"+new_task_id+" period_id:"+new_period_id);
				System.out.println("change:processor_id:"+del_processor_id+"��processor_id:"+new_processor_id+"�ɕω�");
				sql = "delete from timetableSQL where processor_id ='" + del_processor_id+"' and period_id = '" + new_period_id+"' and task_id = '"+new_task_id +"'";
				smt.executeUpdate(sql);
				sql = "insert into timetableSQL(task_id,period_id , processor_id) values("+ "'" + new_task_id + "','" +new_period_id + "','" + new_processor_id+"')";
				smt.executeUpdate(sql);
			}
		}catch(SQLException e) {
			System.err.println(e);
			throw e;
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs2!=null){
				rs2.close();
				rs2 = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt2!=null){
				smt2.close();
				smt2 = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt3!=null){
				smt3.close();
				smt3 = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		return change;
	}
	
	//�ߖT����Change:�u�`�̒S���u�t��ύX����(���\�b�h�g�p)
	//�ߖTchange��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_change2() throws Exception{
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		boolean change = false;
		int min_val,min_off,new_val,new_off;
		int del_processor_id = 0,new_task_id = 0,new_processor_id = 0,new_period_id = 0;
		//�l�̕]��
		//min_val = valuate("timetableSQL");
		min_val = valuateTimeTableSQLFile();
		min_off = count_offence2("timetableSQL");
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			//����p�̃e�[�u�����쐬
			createTTimeTable();
			//���݂̎��Ԋ��̎擾
			String sql = "select * from timetableSQL";
			rs = smt.executeQuery(sql);
			int task_id;
			int period_id;
			int processor_id;
			int qualification_id = 0;
			while(rs.next()) {
				task_id = rs.getInt("task_id");
				period_id = rs.getInt("period_id");
				processor_id = rs.getInt("processor_id");
				//(task,period,processor)����菜��
				deleteTTimeTableRow(task_id,period_id,processor_id);
				//task_id�̍u�`��ނ𒲂ׂ�
				Lecture l = SearchLecture(task_id);
				qualification_id = l.getQualification_id();
				//period_id�ɒS���\�ȍu�t�ɑ΂��ĕύX��������
				Iterator i = teachers.iterator();
				while(i.hasNext()){
					Teacher t = (Teacher)i.next();
					//���݂̍u�t�Ɠ����u�t�łȂ�
					if(t.getProcessor_id()!=processor_id){
						//�u�tt�͍u�`�̎��qualification_id��S���ł���
						if(t.isQualification(qualification_id)){
							//�u�tt��period_id�ŒS���ł���
							if(t.isPeriod(period_id)){
								//System.out.println("processor_id:"+t.getProcessor_id());
								int processor_id2 = t.getProcessor_id();
								//�u�t�͓������Ԃɕ����̍u�`�����Ȃ�
								if(isNotTasks(processor_id2,period_id)){
									//(task_id,period_id,processor_id2)�̒ǉ�
									insertTTimeTableRow(task_id,period_id,processor_id2);
									//�l�̕]��
									new_off = count_offence2("t_timetable");
									new_val = valuateTTimeTableFile();
									if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
									{
										//�ω������������Ă���
										new_task_id = task_id;
										new_processor_id = processor_id2;
										del_processor_id = processor_id;
										new_period_id = period_id;
										min_off = new_off;
										min_val = new_val;
										change = true;
									}
									//�ǉ�����(task_id,period_id,processor_id2)�̍폜
									deleteTTimeTableRow(task_id,period_id,processor_id2);
								}
							}
						}
					}
				}
				//��菜����(task_id,period_id,processor_id)�̒ǉ�
				insertTTimeTableRow(task_id,period_id,processor_id);
				
			}
			//����p�e�[�u���̍폜
			deleteTTimeTable();
			//�������P���ꂽ��
			if(change){
				//���܂ł̉���ύX����
				deleteTimeTableRow(new_task_id,new_period_id,del_processor_id);
				insertTimeTableRow(new_task_id,new_period_id,new_processor_id);
				//�ύX�̕\��
				System.out.println("task_id:"+new_task_id+" period_id:"+new_period_id);
				System.out.println("change:processor_id:"+del_processor_id+"��processor_id:"+new_processor_id+"�ɕω�");
			}
		}catch(SQLException e) {
			System.err.println(e);
			throw e;
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		return change;
	}
	//����p�̃e�[�u�����g�p���Ȃ�
	//�ߖT����Change:�u�`�̒S���u�t��ύX����(���\�b�h�g�p)
	//�ߖTchange��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_change2b() throws Exception{
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		boolean change = false;
		int min_val,min_off,new_val,new_off;
		int del_processor_id = 0,new_task_id = 0,new_processor_id = 0,new_period_id = 0;
		//�l�̕]��
		min_val = valuateTimeTableSQLFile();
		min_off = count_offence2("timetableSQL");
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			//���݂̎��Ԋ��̎擾
			String sql = "select * from timetableSQL";
			rs = smt.executeQuery(sql);
			int task_id;
			int period_id;
			int processor_id;
			int qualification_id = 0;
			while(rs.next()) {
				task_id = rs.getInt("task_id");
				period_id = rs.getInt("period_id");
				processor_id = rs.getInt("processor_id");
				//(task,period,processor)����菜��
				deleteTimeTableRow(task_id,period_id,processor_id);
				//task_id�̍u�`��ނ𒲂ׂ�
				Lecture l = SearchLecture(task_id);
				qualification_id = l.getQualification_id();
				//period_id�ɒS���\�ȍu�t�ɑ΂��ĕύX��������
				Iterator i = teachers.iterator();
				while(i.hasNext()){
					Teacher t = (Teacher)i.next();
					//���݂̍u�t�Ɠ����u�t�łȂ�
					if(t.getProcessor_id()!=processor_id){
						//�u�tt�͍u�`�̎��qualification_id��S���ł���
						if(t.isQualification(qualification_id)){
							//�u�tt��period_id�ŒS���ł���
							if(t.isPeriod(period_id)){
								//System.out.println("processor_id:"+t.getProcessor_id());
								int processor_id2 = t.getProcessor_id();
								//�u�t�͓������Ԃɕ����̍u�`�����Ȃ�
								if(isNotTasksb(processor_id2,period_id)){
									//(task_id,period_id,processor_id2)�̒ǉ�
									insertTimeTableRow(task_id,period_id,processor_id2);
									//�l�̕]��
									new_off = count_offence2("timetableSQL");
									new_val = valuateTimeTableSQLFile();
									if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
									{
										//�ω������������Ă���
										new_task_id = task_id;
										new_processor_id = processor_id2;
										del_processor_id = processor_id;
										new_period_id = period_id;
										min_off = new_off;
										min_val = new_val;
										change = true;
									}
									//�ǉ�����(task_id,period_id,processor_id2)�̍폜
									deleteTimeTableRow(task_id,period_id,processor_id2);
								}
							}
						}
					}
				}
				//��菜����(task_id,period_id,processor_id)�̒ǉ�
				insertTimeTableRow(task_id,period_id,processor_id);
				
			}
			//�������P���ꂽ��
			if(change){
				//���܂ł̉���ύX����
				deleteTimeTableRow(new_task_id,new_period_id,del_processor_id);
				insertTimeTableRow(new_task_id,new_period_id,new_processor_id);
				//�ύX�̕\��
				System.out.println("task_id:"+new_task_id+" period_id:"+new_period_id);
				System.out.println("change:processor_id:"+del_processor_id+"��processor_id:"+new_processor_id+"�ɕω�");
			}
		}catch(SQLException e) {
			System.err.println(e);
			throw e;
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		return change;
	}
	
	//�ߖT����Change:�u�`�̒S���u�t��ύX����(���\�b�h�g�p)
	//�ߖTchange��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_change3() throws Exception{
		ArrayList t_timetable = null;
		boolean change = false;
		int min_val,min_off,new_val,new_off;
		int del_processor_id = 0,new_task_id = 0,new_processor_id = 0,new_period_id = 0;
		timetable.loadTimeTable();
		//�l�̕]��
		min_val = valuateTimeTableSQLFile();
		min_off = timetable.countOffence();
		t_timetable = timetable.getTimeTableRowsCopy();
		//����p�̃e�[�u�����쐬
		createTTimeTable();
		//���݂̎��Ԋ��̎擾
		Iterator i_t_timetable = t_timetable.iterator();
		int task_id, period_id, processor_id;
		int qualification_id = 0;
		while(i_t_timetable.hasNext()) {
			TimeTableRow ttr = (TimeTableRow)i_t_timetable.next();
			task_id = ttr.getTask_id();
			period_id = ttr.getPeriod_id();
			processor_id = ttr.getProcessor_id();
			//(task,period,processor)����菜��
			deleteTTimeTableRow(task_id,period_id,processor_id);
			timetable.deleteTimeTableRow(task_id, period_id, processor_id);
			//task_id�̍u�`��ނ𒲂ׂ�
			Lecture l = SearchLecture(task_id);
			qualification_id = l.getQualification_id();
			//period_id�ɒS���\�ȍu�t�ɑ΂��ĕύX��������
			Iterator i = teachers.iterator();
			while(i.hasNext()){
				Teacher t = (Teacher)i.next();
				//���݂̍u�t�Ɠ����u�t�łȂ�
				if(t.getProcessor_id()!=processor_id){
					//�u�tt�͍u�`�̎��qualification_id��S���ł���
					if(t.isQualification(qualification_id)){
						//�u�tt��period_id�ŒS���ł���
						if(t.isPeriod(period_id)){
							//System.out.println("processor_id:"+t.getProcessor_id());
							int processor_id2 = t.getProcessor_id();
							//�u�t�͓������Ԃɕ����̍u�`�����Ȃ�
							if(timetable.isNotTasks(processor_id2,period_id)){
								//(task_id,period_id,processor_id2)�̒ǉ�
								insertTTimeTableRow(task_id,period_id,processor_id2);
								timetable.insertTimeTableRow(task_id,period_id,processor_id2);
								//�l�̕]��
								new_off = timetable.countOffence();
								new_val = valuateTTimeTableFile();
								if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
								{
									//�ω������������Ă���
									new_task_id = task_id;
									new_processor_id = processor_id2;
									del_processor_id = processor_id;
									new_period_id = period_id;
									min_off = new_off;
									min_val = new_val;
									change = true;
								}
								//�ǉ�����(task_id,period_id,processor_id2)�̍폜
								deleteTTimeTableRow(task_id,period_id,processor_id2);
								timetable.deleteTimeTableRow(task_id,period_id,processor_id2);
							}
						}
					}
				}
			}
			//��菜����(task_id,period_id,processor_id)�̒ǉ�
			insertTTimeTableRow(task_id,period_id,processor_id);
			timetable.insertTimeTableRow(task_id,period_id,processor_id);
		}
		//����p�e�[�u���̍폜
		deleteTTimeTable();
		//�������P���ꂽ��
		if(change){
			//���܂ł̉���ύX����
			deleteTimeTableRow(new_task_id,new_period_id,del_processor_id);
			//timetable.deleteTimeTableRow(new_task_id,new_period_id,del_processor_id);
			insertTimeTableRow(new_task_id,new_period_id,new_processor_id);
			//timetable.insertTimeTableRow(new_task_id,new_period_id,new_processor_id);
			//�ύX�̕\��
			System.out.println("task_id:"+new_task_id+" period_id:"+new_period_id);
			System.out.println("change:processor_id:"+del_processor_id+"��processor_id:"+new_processor_id+"�ɕω�");
		}
		return change;
	}
	//(�v���O�������̃e�[�u�������������Ȃ�)	
	//�ߖT����Change:�u�`�̒S���u�t��ύX����(���\�b�h�g�p)
	//�ߖTchange��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_change3b() throws Exception{
		ArrayList t_timetable = null;
		boolean change = false;
		int min_val,min_off,new_val,new_off;
		int del_processor_id = 0,new_task_id = 0,new_processor_id = 0,new_period_id = 0;
		//�l�̕]��
		min_val = valuateTimeTableSQLFile();
		min_off = timetable.countOffence();
		t_timetable = timetable.getTimeTableRowsCopy();
		//����p�̃e�[�u�����쐬
		createTTimeTable();
		//���݂̎��Ԋ��̎擾
		Iterator i_t_timetable = t_timetable.iterator();
		int task_id, period_id, processor_id;
		int qualification_id = 0;
		while(i_t_timetable.hasNext()) {
			TimeTableRow ttr = (TimeTableRow)i_t_timetable.next();
			task_id = ttr.getTask_id();
			period_id = ttr.getPeriod_id();
			processor_id = ttr.getProcessor_id();
			//(task,period,processor)����菜��
			deleteTTimeTableRow(task_id,period_id,processor_id);
			timetable.deleteTimeTableRow(task_id, period_id, processor_id);
			//task_id�̍u�`��ނ𒲂ׂ�
			Lecture l = SearchLecture(task_id);
			qualification_id = l.getQualification_id();
			//period_id�ɒS���\�ȍu�t�ɑ΂��ĕύX��������
			Iterator i = teachers.iterator();
			while(i.hasNext()){
				Teacher t = (Teacher)i.next();
				//���݂̍u�t�Ɠ����u�t�łȂ�
				if(t.getProcessor_id()!=processor_id){
					//�u�tt�͍u�`�̎��qualification_id��S���ł���
					if(t.isQualification(qualification_id)){
						//�u�tt��period_id�ŒS���ł���
						if(t.isPeriod(period_id)){
							//System.out.println("processor_id:"+t.getProcessor_id());
							int processor_id2 = t.getProcessor_id();
							//�u�t�͓������Ԃɕ����̍u�`�����Ȃ�
							if(timetable.isNotTasks(processor_id2,period_id)){
								//(task_id,period_id,processor_id2)�̒ǉ�
								insertTTimeTableRow(task_id,period_id,processor_id2);
								timetable.insertTimeTableRow(task_id,period_id,processor_id2);
								//�l�̕]��
								new_off = timetable.countOffence();
								new_val = valuateTTimeTableFile();
								if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
								{
									//�ω������������Ă���
									new_task_id = task_id;
									new_processor_id = processor_id2;
									del_processor_id = processor_id;
									new_period_id = period_id;
									min_off = new_off;
									min_val = new_val;
									change = true;
								}
								//�ǉ�����(task_id,period_id,processor_id2)�̍폜
								deleteTTimeTableRow(task_id,period_id,processor_id2);
								timetable.deleteTimeTableRow(task_id,period_id,processor_id2);
							}
						}
					}
				}
			}
			//��菜����(task_id,period_id,processor_id)�̒ǉ�
			insertTTimeTableRow(task_id,period_id,processor_id);
			timetable.insertTimeTableRow(task_id,period_id,processor_id);
		}
		//����p�e�[�u���̍폜
		deleteTTimeTable();
		//�������P���ꂽ��
		if(change){
			//���܂ł̉���ύX����
			deleteTimeTableRow(new_task_id,new_period_id,del_processor_id);
			timetable.deleteTimeTableRow(new_task_id,new_period_id,del_processor_id);
			insertTimeTableRow(new_task_id,new_period_id,new_processor_id);
			timetable.insertTimeTableRow(new_task_id,new_period_id,new_processor_id);
			//�ύX�̕\��
			System.out.println("task_id:"+new_task_id+" period_id:"+new_period_id);
			System.out.println("change:processor_id:"+del_processor_id+"��processor_id:"+new_processor_id+"�ɕω�");
		}
		return change;
	}

	//(�v���O�������̃e�[�u�������������Ȃ�)	
	//�ߖT����Change:�u�`�̒S���u�t��ύX����(���\�b�h�g�p)
	//�ߖTchange��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_change3c() throws Exception{
		ArrayList t_timetable = null;
		boolean change = false;
		int min_val,min_off,new_val,new_off;
		int del_processor_id = 0,new_task_id = 0,new_processor_id = 0,new_period_id = 0;
		//�l�̕]��
		min_val = valuateTimeTableSQLFile();
		min_off = timetable.countOffence();
		t_timetable = timetable.getTimeTableRowsCopy();
		//���݂̎��Ԋ��̎擾
		Iterator i_t_timetable = t_timetable.iterator();
		int task_id, period_id, processor_id;
		int qualification_id = 0;
		while(i_t_timetable.hasNext()) {
			TimeTableRow ttr = (TimeTableRow)i_t_timetable.next();
			task_id = ttr.getTask_id();
			period_id = ttr.getPeriod_id();
			processor_id = ttr.getProcessor_id();
			//(task,period,processor)����菜��
			deleteTimeTableRow(task_id,period_id,processor_id);
			timetable.deleteTimeTableRow(task_id, period_id, processor_id);
			//task_id�̍u�`��ނ𒲂ׂ�
			Lecture l = SearchLecture(task_id);
			qualification_id = l.getQualification_id();
			//period_id�ɒS���\�ȍu�t�ɑ΂��ĕύX��������
			Iterator i = teachers.iterator();
			while(i.hasNext()){
				Teacher t = (Teacher)i.next();
				//���݂̍u�t�Ɠ����u�t�łȂ�
				if(t.getProcessor_id()!=processor_id){
					//�u�tt�͍u�`�̎��qualification_id��S���ł���
					if(t.isQualification(qualification_id)){
						//�u�tt��period_id�ŒS���ł���
						if(t.isPeriod(period_id)){
							//System.out.println("processor_id:"+t.getProcessor_id());
							int processor_id2 = t.getProcessor_id();
							//�u�t�͓������Ԃɕ����̍u�`�����Ȃ�
							if(timetable.isNotTasks(processor_id2,period_id)){
								//(task_id,period_id,processor_id2)�̒ǉ�
								insertTimeTableRow(task_id,period_id,processor_id2);
								timetable.insertTimeTableRow(task_id,period_id,processor_id2);
								//�l�̕]��
								new_off = timetable.countOffence();
								new_val = valuateTimeTableSQLFile();
								if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
								{
									//�ω������������Ă���
									new_task_id = task_id;
									new_processor_id = processor_id2;
									del_processor_id = processor_id;
									new_period_id = period_id;
									min_off = new_off;
									min_val = new_val;
									change = true;
								}
								//�ǉ�����(task_id,period_id,processor_id2)�̍폜
								deleteTimeTableRow(task_id,period_id,processor_id2);
								timetable.deleteTimeTableRow(task_id,period_id,processor_id2);
							}
						}
					}
				}
			}
			//��菜����(task_id,period_id,processor_id)�̒ǉ�
			insertTimeTableRow(task_id,period_id,processor_id);
			timetable.insertTimeTableRow(task_id,period_id,processor_id);
		}
		//�������P���ꂽ��
		if(change){
			//���܂ł̉���ύX����
			deleteTimeTableRow(new_task_id,new_period_id,del_processor_id);
			timetable.deleteTimeTableRow(new_task_id,new_period_id,del_processor_id);
			insertTimeTableRow(new_task_id,new_period_id,new_processor_id);
			timetable.insertTimeTableRow(new_task_id,new_period_id,new_processor_id);
			//�ύX�̕\��
			System.out.println("task_id:"+new_task_id+" period_id:"+new_period_id);
			System.out.println("change:processor_id:"+del_processor_id+"��processor_id:"+new_processor_id+"�ɕω�");
		}
		return change;
	}
	
//�ߖT����Swap:�قȂ�2�̒S���u�t�����ւ���
	//�ߖTswap��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_swap() throws Exception{
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
		int next = 0;
		//�l�̕]��
		min_val = valuateTimeTableSQLFile();
		min_off = count_offence("timetableSQL");
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			smt2 = con.createStatement();
			smt3 = con.createStatement();
			smt4 = con.createStatement();
			smt5 = con.createStatement();
			smt6 = con.createStatement();
			//����p�̃e�[�u�����쐬
			String sql = "create table t_timetable as select * from timetableSQL";
			smt.executeUpdate(sql);
			//���݂̎��Ԋ��̎擾
			sql = "select * from timetableSQL";
			//���݂̎��Ԋ��̎擾
			rs = smt.executeQuery(sql);
			rs2 = smt2.executeQuery(sql);
			while(rs.next()){
				int task_id1 = rs.getInt("task_id");
				int period_id1 = rs.getInt("period_id");
				int processor_id1 = rs.getInt("processor_id");
				while(rs2.next()){
					int task_id2 = rs2.getInt("task_id");
					int period_id2 = rs2.getInt("period_id");
					int processor_id2 = rs2.getInt("processor_id");
					int qualification_id1=0,qualification_id2=0;
					//�قȂ�u�`���ǂ���
					if(task_id1!=task_id2){
						//(task_id1,processor_id1,period_id1)��(task_id2,processor_id2,period_id2)����菜��
						String sql2 = "delete from t_timetable  where processor_id='"+processor_id1+"' and period_id= '" + period_id1+"' and task_id= '"+task_id1 +"'";
						smt3.executeUpdate(sql2);
						sql2 = "delete from t_timetable  where processor_id='"+processor_id2+"' and period_id = '" + period_id2+"' and task_id = '"+task_id2 +"'";
						smt3.executeUpdate(sql2);
						//task_id�̎�ނ𒲂ׂ�
						Lecture l1 = SearchLecture(task_id1);
						qualification_id1 = l1.getQualification_id();
						Lecture l2 = SearchLecture(task_id2);
						qualification_id2 = l2.getQualification_id();
						//(task_id1,period_id1,processor_id2)���L�����ǂ���
						//processor_id2��period_id1�ɉ\���ǂ����H
						//�u�tt��processor_id2�ł���
						Teacher t = SearchTeacher(processor_id2);
						if(t.isQualification(qualification_id1)){
							//processor_id2��period_id1�ɉ\
							if(t.isPeriod(period_id1)){
								//�u�t�͓������Ԃɕ����̍u�`�����Ȃ�
								String sql3 = "select * from t_timetable where processor_id='"+processor_id2+"' and period_id = '" + period_id1 +"'";
								rs4 = smt4.executeQuery(sql3);
								if(rs4.next()==false){
									//(task_id2,period_id2,processor_id1)���L�����ǂ���
									//processor_id1��period_id2�ɒS���\���ǂ����H
									//�u�tt2��processor_id1�ł���
									Teacher t2 = SearchTeacher(processor_id1);
									if(t2.isQualification(qualification_id2)){
										//processor_id1��period_id2�ɉ\
										if(t2.isPeriod(period_id2)){
											//�u�t�͓������Ԃɕ����̍u�`�����Ȃ�
											String sql4 = "select * from t_timetable where processor_id='"+processor_id1+"' and period_id = '" + period_id2 +"'";
											rs5 = smt5.executeQuery(sql4);
											if(rs5.next()==false){
												//�u�t�̒ǉ�
												String sql5="insert into t_timetable(task_id,period_id , processor_id) values("+ "'" + task_id1 + "','" +period_id1 + "','" + processor_id2+"')";
												smt6.executeUpdate(sql5);
												sql5="insert into t_timetable(task_id,period_id , processor_id) values("+ "'" + task_id2 + "','" +period_id2 + "','" + processor_id1+"')";
												smt6.executeUpdate(sql5);
												new_off = count_offence("t_timetable");
												new_val = valuateTTimeTableFile();
												if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
												{
													//�ω������������Ă���
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
												//�ǉ������u�t�̍폜
												sql5="delete from t_timetable where processor_id ='" + processor_id2+"' and period_id = '" + period_id1+"' and task_id = '"+task_id1 +"'";
												smt6.executeUpdate(sql5);
												sql5="delete from t_timetable where processor_id ='" + processor_id1+"' and period_id = '" + period_id2+"' and task_id = '"+task_id2 +"'";
												smt6.executeUpdate(sql5);
											}
										}
									}
								}
							}
						}
						//��菜����(task_id1,processor_id1,period_id1)��(task_id2,processor_id2,period_id2)��ǉ�����
						sql2 = "insert into t_timetable(task_id,period_id , processor_id) values("+ "'" + task_id1 + "','" +period_id1 + "','" + processor_id1+"')";
						smt3.executeUpdate(sql2);
						sql2 = "insert into t_timetable(task_id,period_id , processor_id) values("+ "'" + task_id2 + "','" +period_id2 + "','" + processor_id2+"')";
						smt3.executeUpdate(sql2);
					}
				}
				next++;
				rs2.absolute(next);
				//���߂ɖ߂�
				//rs2.beforeFirst();
			}
			//����p�e�[�u���̍폜
			sql = "drop table t_timetable";
			smt.executeUpdate(sql);
			//�������P���ꂽ��
			if(change){
				System.out.println("(task_id,processor_id,period_id)");
				System.out.println("swap:("+new_task_id1+","+new_processor_id1+","+new_period_id1+")��("+new_task_id2+","+new_processor_id2+","+new_period_id2+")");
				sql = "update timetableSQL set processor_id ='"+new_processor_id2+"' where processor_id = '"+new_processor_id1+"' and task_id = '" + new_task_id1 + "' and period_id = '" + new_period_id1 + "'";
				smt.executeUpdate(sql);
				sql = "update timetableSQL set processor_id ='"+new_processor_id1+"' where processor_id = '"+new_processor_id2+"' and task_id = '" + new_task_id2 + "' and period_id = '" + new_period_id2 + "'";
				smt.executeUpdate(sql);
			}
		}catch(SQLException e) {
			e.printStackTrace();
			System.err.println(e);
			throw e;
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs2!=null){
				rs2.close();
				rs2 = null;
			}
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs3!=null){
				rs3.close();
				rs3 = null;
			}
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs4!=null){
				rs4.close();
				rs4 = null;
			}
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs5!=null){
				rs5.close();
				rs5 = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt2!=null){
				smt2.close();
				smt2 = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt3!=null){
				smt3.close();
				smt3 = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt4!=null){
				smt4.close();
				smt4 = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt5!=null){
				smt5.close();
				smt5 = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt6!=null){
				smt6.close();
				smt6 = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		return change;
	}

	//�ߖT����Swap:�قȂ�2�̒S���u�t�����ւ���
	//�ߖTswap��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_swap1() throws Exception{
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
		int next = 0;
		//�l�̕]��
		min_val = valuateTimeTableSQLFile();
		min_off = count_offence2("timetableSQL");
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			smt2 = con.createStatement();
			smt3 = con.createStatement();
			smt4 = con.createStatement();
			smt5 = con.createStatement();
			smt6 = con.createStatement();
			//����p�̃e�[�u�����쐬
			String sql = "create table t_timetable as select * from timetableSQL";
			smt.executeUpdate(sql);
			//���݂̎��Ԋ��̎擾
			sql = "select * from timetableSQL";
			//���݂̎��Ԋ��̎擾
			rs = smt.executeQuery(sql);
			rs2 = smt2.executeQuery(sql);
			while(rs.next()){
				int task_id1 = rs.getInt("task_id");
				int period_id1 = rs.getInt("period_id");
				int processor_id1 = rs.getInt("processor_id");
				while(rs2.next()){
					int task_id2 = rs2.getInt("task_id");
					int period_id2 = rs2.getInt("period_id");
					int processor_id2 = rs2.getInt("processor_id");
					int qualification_id1=0,qualification_id2=0;
					//�قȂ�u�`���ǂ���
					if(task_id1!=task_id2){
						//(task_id1,processor_id1,period_id1)��(task_id2,processor_id2,period_id2)����菜��
						String sql2 = "delete from t_timetable  where processor_id='"+processor_id1+"' and period_id= '" + period_id1+"' and task_id= '"+task_id1 +"'";
						smt3.executeUpdate(sql2);
						sql2 = "delete from t_timetable  where processor_id='"+processor_id2+"' and period_id = '" + period_id2+"' and task_id = '"+task_id2 +"'";
						smt3.executeUpdate(sql2);
						//task_id�̎�ނ𒲂ׂ�
						Lecture l1 = SearchLecture(task_id1);
						qualification_id1 = l1.getQualification_id();
						Lecture l2 = SearchLecture(task_id2);
						qualification_id2 = l2.getQualification_id();
						//(task_id1,period_id1,processor_id2)���L�����ǂ���
						//processor_id2��period_id1�ɉ\���ǂ����H
						//�u�tt��processor_id2�ł���
						Teacher t = SearchTeacher(processor_id2);
						if(t.isQualification(qualification_id1)){
							//processor_id2��period_id1�ɉ\
							if(t.isPeriod(period_id1)){
								//�u�t�͓������Ԃɕ����̍u�`�����Ȃ�
								String sql3 = "select * from t_timetable where processor_id='"+processor_id2+"' and period_id = '" + period_id1 +"'";
								rs4 = smt4.executeQuery(sql3);
								if(rs4.next()==false){
									//(task_id2,period_id2,processor_id1)���L�����ǂ���
									//processor_id1��period_id2�ɒS���\���ǂ����H
									//�u�tt2��processor_id1�ł���
									Teacher t2 = SearchTeacher(processor_id1);
									if(t2.isQualification(qualification_id2)){
										//processor_id1��period_id2�ɉ\
										if(t2.isPeriod(period_id2)){
											//�u�t�͓������Ԃɕ����̍u�`�����Ȃ�
											String sql4 = "select * from t_timetable where processor_id='"+processor_id1+"' and period_id = '" + period_id2 +"'";
											rs5 = smt5.executeQuery(sql4);
											if(rs5.next()==false){
												//�u�t�̒ǉ�
												String sql5="insert into t_timetable(task_id,period_id , processor_id) values("+ "'" + task_id1 + "','" +period_id1 + "','" + processor_id2+"')";
												smt6.executeUpdate(sql5);
												sql5="insert into t_timetable(task_id,period_id , processor_id) values("+ "'" + task_id2 + "','" +period_id2 + "','" + processor_id1+"')";
												smt6.executeUpdate(sql5);
												//new_off = count_offence("t_timetable");
												new_off = count_offence2("t_timetable");
												//new_val = valuate("t_timetable");
												new_val = valuateTTimeTableFile();
												if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
												{
													//�ω������������Ă���
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
												//�ǉ������u�t�̍폜
												sql5="delete from t_timetable where processor_id ='" + processor_id2+"' and period_id = '" + period_id1+"' and task_id = '"+task_id1 +"'";
												smt6.executeUpdate(sql5);
												sql5="delete from t_timetable where processor_id ='" + processor_id1+"' and period_id = '" + period_id2+"' and task_id = '"+task_id2 +"'";
												smt6.executeUpdate(sql5);
											}
										}
									}
								}
							}
						}
						//��菜����(task_id1,processor_id1,period_id1)��(task_id2,processor_id2,period_id2)��ǉ�����
						sql2 = "insert into t_timetable(task_id,period_id , processor_id) values("+ "'" + task_id1 + "','" +period_id1 + "','" + processor_id1+"')";
						smt3.executeUpdate(sql2);
						sql2 = "insert into t_timetable(task_id,period_id , processor_id) values("+ "'" + task_id2 + "','" +period_id2 + "','" + processor_id2+"')";
						smt3.executeUpdate(sql2);
					}
				}
				next++;
				rs2.absolute(next);
				//���߂ɖ߂�
				//rs2.beforeFirst();
			}
			//����p�e�[�u���̍폜
			sql = "drop table t_timetable";
			smt.executeUpdate(sql);
			//�������P���ꂽ��
			if(change){
				System.out.println("(task_id,processor_id,period_id)");
				System.out.println("swap:("+new_task_id1+","+new_processor_id1+","+new_period_id1+")��("+new_task_id2+","+new_processor_id2+","+new_period_id2+")");
				sql = "update timetableSQL set processor_id ='"+new_processor_id2+"' where processor_id = '"+new_processor_id1+"' and task_id = '" + new_task_id1 + "' and period_id = '" + new_period_id1 + "'";
				smt.executeUpdate(sql);
				sql = "update timetableSQL set processor_id ='"+new_processor_id1+"' where processor_id = '"+new_processor_id2+"' and task_id = '" + new_task_id2 + "' and period_id = '" + new_period_id2 + "'";
				smt.executeUpdate(sql);
			}
		}catch(SQLException e) {
			e.printStackTrace();
			System.err.println(e);
			throw e;
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs2!=null){
				rs2.close();
				rs2 = null;
			}
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs3!=null){
				rs3.close();
				rs3 = null;
			}
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs4!=null){
				rs4.close();
				rs4 = null;
			}
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs5!=null){
				rs5.close();
				rs5 = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt2!=null){
				smt2.close();
				smt2 = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt3!=null){
				smt3.close();
				smt3 = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt4!=null){
				smt4.close();
				smt4 = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt5!=null){
				smt5.close();
				smt5 = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt6!=null){
				smt6.close();
				smt6 = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		return change;
	}
	//����p�̃e�[�u�����g�p���Ȃ�
	//�ߖT����Swap:�قȂ�2�̒S���u�t�����ւ���(���\�b�h�g�p)
	//�ߖTswap��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_swap2b() throws Exception{
		Connection con = null;
		ResultSet rs = null;
		ResultSet rs2 = null;
		Statement smt = null;
		Statement smt2 = null;
		boolean change = false;
		int min_val,min_off,new_val,new_off;
		int new_task_id1 = 0,new_processor_id1 = 0,new_period_id1 = 0,new_task_id2 = 0,new_processor_id2 = 0,new_period_id2 = 0;
		int next = 0;
		//�l�̕]��
		min_val = valuateTimeTableSQLFile();
		min_off = count_offence2("timetableSQL");
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			smt2 = con.createStatement();
			//���݂̎��Ԋ��̎擾
			String sql = "select * from timetableSQL";
			//���݂̎��Ԋ��̎擾
			rs = smt.executeQuery(sql);
			rs2 = smt2.executeQuery(sql);
			while(rs.next()){
				int task_id1 = rs.getInt("task_id");
				int period_id1 = rs.getInt("period_id");
				int processor_id1 = rs.getInt("processor_id");
				while(rs2.next()){
					int task_id2 = rs2.getInt("task_id");
					int period_id2 = rs2.getInt("period_id");
					int processor_id2 = rs2.getInt("processor_id");
					int qualification_id1=0,qualification_id2=0;
					//�قȂ�u�`���ǂ���
					if(task_id1!=task_id2){
						//(task_id1,period_id1,processor_id1)��(task_id2,period_id2,processor_id2)����菜��
						deleteTimeTableRow(task_id1,period_id1,processor_id1);
						deleteTimeTableRow(task_id2,period_id2,processor_id2);
						//task_id1��task_id2�̎�ނ𒲂ׂ�
						Lecture l1 = SearchLecture(task_id1);
						qualification_id1 = l1.getQualification_id();
						Lecture l2 = SearchLecture(task_id2);
						qualification_id2 = l2.getQualification_id();
						//(task_id1,period_id1,processor_id2)���L�����ǂ���
						//processor_id2��period_id1�ɉ\���ǂ����H
						//�u�tt��processor_id2�ł���
						Teacher t = SearchTeacher(processor_id2);
						if(t.isQualification(qualification_id1)){
							//processor_id2��period_id1�ɉ\
							if(t.isPeriod(period_id1)){
								//�u�t�͓������Ԃɕ����̍u�`�����Ȃ�
								if(isNotTasksb(processor_id2,period_id1)){
									//(task_id2,period_id2,processor_id1)���L�����ǂ���
									//processor_id1��period_id2�ɒS���\���ǂ����H
									//�u�tt2��processor_id1�ł���
									Teacher t2 = SearchTeacher(processor_id1);
									if(t2.isQualification(qualification_id2)){
										//processor_id1��period_id2�ɉ\
										if(t2.isPeriod(period_id2)){
											//�u�t�͓������Ԃɕ����̍u�`�����Ȃ�
											if(isNotTasksb(processor_id1,period_id2)){
												//(task_id1,period_id1,processor_id2)�̒ǉ�
											    insertTimeTableRow(task_id1,period_id1,processor_id2);
												//(task_id2,period_id2,processor_id1)�̒ǉ�
											    insertTimeTableRow(task_id2,period_id2,processor_id1);
												new_off = count_offence2("timetableSQL");
												//new_val = valuate("t_timetable");
												new_val = valuateTimeTableSQLFile();
												if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
												{
													//�ω������������Ă���
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
												//�ǉ�����(task_id1,period_id1,processor_id2)�̍폜
												deleteTimeTableRow(task_id1,period_id1,processor_id2);
												//�ǉ�����(task_id2,period_id2,processor_id1)�̍폜
												deleteTimeTableRow(task_id2,period_id2,processor_id1);
											}
										}
									}
								}
							}
						}
						//��菜����(task_id1,period_id1,processor_id1)��(task_id2,period_id2,processor_id2)��ǉ�����
						insertTimeTableRow(task_id1,period_id1,processor_id1);
						insertTimeTableRow(task_id2,period_id2,processor_id2);
					}
				}
				next++;
				rs2.absolute(next);
				//���߂ɖ߂�
				//rs2.beforeFirst();
			}
			//�������P���ꂽ��
			if(change){
				updateProcessorTimeTable(new_processor_id2,new_processor_id1,new_task_id1,new_period_id1);
				updateProcessorTimeTable(new_processor_id1,new_processor_id2,new_task_id2,new_period_id2);
				//�ύX�̕\��
				System.out.println("(task_id,processor_id,period_id)");
				System.out.println("swap:("+new_task_id1+","+new_processor_id1+","+new_period_id1+")��("+new_task_id2+","+new_processor_id2+","+new_period_id2+")");
			}
		}catch(SQLException e) {
			e.printStackTrace();
			System.err.println(e);
			throw e;
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs2!=null){
				rs2.close();
				rs2 = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt2!=null){
				smt2.close();
				smt2 = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		return change;
	}
	
	//�ߖT����Swap:�قȂ�2�̒S���u�t�����ւ���(���\�b�h�g�p)
	//�ߖTswap��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_swap2() throws Exception{
		Connection con = null;
		ResultSet rs = null;
		ResultSet rs2 = null;
		Statement smt = null;
		Statement smt2 = null;
		boolean change = false;
		int min_val,min_off,new_val,new_off;
		int new_task_id1 = 0,new_processor_id1 = 0,new_period_id1 = 0,new_task_id2 = 0,new_processor_id2 = 0,new_period_id2 = 0;
		int next = 0;
		//�l�̕]��
		min_val = valuateTimeTableSQLFile();
		min_off = count_offence2("timetableSQL");
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			smt2 = con.createStatement();
			//����p�̃e�[�u�����쐬
			createTTimeTable();
			//���݂̎��Ԋ��̎擾
			String sql = "select * from timetableSQL";
			//���݂̎��Ԋ��̎擾
			rs = smt.executeQuery(sql);
			rs2 = smt2.executeQuery(sql);
			while(rs.next()){
				int task_id1 = rs.getInt("task_id");
				int period_id1 = rs.getInt("period_id");
				int processor_id1 = rs.getInt("processor_id");
				while(rs2.next()){
					int task_id2 = rs2.getInt("task_id");
					int period_id2 = rs2.getInt("period_id");
					int processor_id2 = rs2.getInt("processor_id");
					int qualification_id1=0,qualification_id2=0;
					//�قȂ�u�`���ǂ���
					if(task_id1!=task_id2){
						//(task_id1,period_id1,processor_id1)��(task_id2,period_id2,processor_id2)����菜��
						deleteTTimeTableRow(task_id1,period_id1,processor_id1);
						deleteTTimeTableRow(task_id2,period_id2,processor_id2);
						//task_id1��task_id2�̎�ނ𒲂ׂ�
						Lecture l1 = SearchLecture(task_id1);
						qualification_id1 = l1.getQualification_id();
						Lecture l2 = SearchLecture(task_id2);
						qualification_id2 = l2.getQualification_id();
						//(task_id1,period_id1,processor_id2)���L�����ǂ���
						//processor_id2��period_id1�ɉ\���ǂ����H
						//�u�tt��processor_id2�ł���
						Teacher t = SearchTeacher(processor_id2);
						if(t.isQualification(qualification_id1)){
							//processor_id2��period_id1�ɉ\
							if(t.isPeriod(period_id1)){
								//�u�t�͓������Ԃɕ����̍u�`�����Ȃ�
								if(isNotTasks(processor_id2,period_id1)){
									//(task_id2,period_id2,processor_id1)���L�����ǂ���
									//processor_id1��period_id2�ɒS���\���ǂ����H
									//�u�tt2��processor_id1�ł���
									Teacher t2 = SearchTeacher(processor_id1);
									if(t2.isQualification(qualification_id2)){
										//processor_id1��period_id2�ɉ\
										if(t2.isPeriod(period_id2)){
											//�u�t�͓������Ԃɕ����̍u�`�����Ȃ�
											if(isNotTasks(processor_id1,period_id2)){
												//(task_id1,period_id1,processor_id2)�̒ǉ�
											    insertTTimeTableRow(task_id1,period_id1,processor_id2);
												//(task_id2,period_id2,processor_id1)�̒ǉ�
											    insertTTimeTableRow(task_id2,period_id2,processor_id1);
												//new_off = count_offence("t_timetable");
												new_off = count_offence2("t_timetable");
												//new_val = valuate("t_timetable");
												new_val = valuateTTimeTableFile();
												if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
												{
													//�ω������������Ă���
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
												//�ǉ�����(task_id1,period_id1,processor_id2)�̍폜
												deleteTTimeTableRow(task_id1,period_id1,processor_id2);
												//�ǉ�����(task_id2,period_id2,processor_id1)�̍폜
												deleteTTimeTableRow(task_id2,period_id2,processor_id1);
											}
										}
									}
								}
							}
						}
						//��菜����(task_id1,period_id1,processor_id1)��(task_id2,period_id2,processor_id2)��ǉ�����
						insertTTimeTableRow(task_id1,period_id1,processor_id1);
						insertTTimeTableRow(task_id2,period_id2,processor_id2);
					}
				}
				next++;
				rs2.absolute(next);
				//���߂ɖ߂�
				//rs2.beforeFirst();
			}
			//����p�e�[�u���̍폜
			deleteTTimeTable();
			//�������P���ꂽ��
			if(change){
				updateProcessorTimeTable(new_processor_id2,new_processor_id1,new_task_id1,new_period_id1);
				updateProcessorTimeTable(new_processor_id1,new_processor_id2,new_task_id2,new_period_id2);
				//�ύX�̕\��
				System.out.println("(task_id,processor_id,period_id)");
				System.out.println("swap:("+new_task_id1+","+new_processor_id1+","+new_period_id1+")��("+new_task_id2+","+new_processor_id2+","+new_period_id2+")");
			}
		}catch(SQLException e) {
			e.printStackTrace();
			System.err.println(e);
			throw e;
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs2!=null){
				rs2.close();
				rs2 = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt2!=null){
				smt2.close();
				smt2 = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		return change;
	}
	
	//�ߖT����Swap:�قȂ�2�̒S���u�t�����ւ���(���\�b�h�g�p)
	//�ߖTswap��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_swap3() throws Exception{
		ArrayList t_timetable = null;
		boolean change = false;
		int min_val,min_off,new_val,new_off;
		int new_task_id1 = 0,new_processor_id1 = 0,new_period_id1 = 0,new_task_id2 = 0,new_processor_id2 = 0,new_period_id2 = 0;
		int next = 0;
		timetable.loadTimeTable();
		//�l�̕]��
		min_val = valuateTimeTableSQLFile();
		min_off = timetable.countOffence();
		t_timetable = timetable.getTimeTableRowsCopy();
		//����p�̃e�[�u�����쐬
		createTTimeTable();
		//���݂̎��Ԋ��̎擾
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
				//�قȂ�u�`���ǂ���
				if(task_id1!=task_id2){
					//(task_id1,period_id1,processor_id1)��(task_id2,period_id2,processor_id2)����菜��
					deleteTTimeTableRow(task_id1,period_id1,processor_id1);
					deleteTTimeTableRow(task_id2,period_id2,processor_id2);
					timetable.deleteTimeTableRow(task_id1,period_id1,processor_id1);
					timetable.deleteTimeTableRow(task_id2,period_id2,processor_id2);
					//task_id1��task_id2�̎�ނ𒲂ׂ�
					Lecture l1 = SearchLecture(task_id1);
					qualification_id1 = l1.getQualification_id();
					Lecture l2 = SearchLecture(task_id2);
					qualification_id2 = l2.getQualification_id();
					//(task_id1,period_id1,processor_id2)���L�����ǂ���
					//processor_id2��period_id1�ɉ\���ǂ����H
					//�u�tt��processor_id2�ł���
					Teacher t = SearchTeacher(processor_id2);
					if(t.isQualification(qualification_id1)){
						//processor_id2��period_id1�ɉ\
						if(t.isPeriod(period_id1)){
							//�u�t�͓������Ԃɕ����̍u�`�����Ȃ�
							if(timetable.isNotTasks(processor_id2,period_id1)){
								//(task_id2,period_id2,processor_id1)���L�����ǂ���
								//processor_id1��period_id2�ɒS���\���ǂ����H
								//�u�tt2��processor_id1�ł���
								Teacher t2 = SearchTeacher(processor_id1);
								if(t2.isQualification(qualification_id2)){
									//processor_id1��period_id2�ɉ\
									if(t2.isPeriod(period_id2)){
										//�u�t�͓������Ԃɕ����̍u�`�����Ȃ�
										if(timetable.isNotTasks(processor_id1,period_id2)){
											//(task_id1,period_id1,processor_id2)�̒ǉ�
											insertTTimeTableRow(task_id1,period_id1,processor_id2);
											//(task_id2,period_id2,processor_id1)�̒ǉ�
											insertTTimeTableRow(task_id2,period_id2,processor_id1);
											timetable.insertTimeTableRow(task_id1,period_id1,processor_id2);
											timetable.insertTimeTableRow(task_id2,period_id2,processor_id1);
											
											new_off = timetable.countOffence();
											new_val = valuateTTimeTableFile();
											if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
											{
												//�ω������������Ă���
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
											//�ǉ�����(task_id1,period_id1,processor_id2)�̍폜
											deleteTTimeTableRow(task_id1,period_id1,processor_id2);
											//�ǉ�����(task_id2,period_id2,processor_id1)�̍폜
											deleteTTimeTableRow(task_id2,period_id2,processor_id1);
											timetable.deleteTimeTableRow(task_id1,period_id1,processor_id2);
											timetable.deleteTimeTableRow(task_id2,period_id2,processor_id1);
											}
									}
								}
							}
						}
					}
					//��菜����(task_id1,period_id1,processor_id1)��(task_id2,period_id2,processor_id2)��ǉ�����
					insertTTimeTableRow(task_id1,period_id1,processor_id1);
					insertTTimeTableRow(task_id2,period_id2,processor_id2);
					timetable.insertTimeTableRow(task_id1,period_id1,processor_id1);
					timetable.insertTimeTableRow(task_id2,period_id2,processor_id2);
				}
			}
			next++;
		}
		//����p�e�[�u���̍폜
		deleteTTimeTable();
		//�������P���ꂽ��
		if(change){
			System.out.println("min_val:"+min_val);
			updateProcessorTimeTable(new_processor_id2,new_processor_id1,new_task_id1,new_period_id1);
			//timetable.updateProcessorTimeTable(new_processor_id2,new_processor_id1,new_task_id1,new_period_id1);
			updateProcessorTimeTable(new_processor_id1,new_processor_id2,new_task_id2,new_period_id2);
			//timetable.updateProcessorTimeTable(new_processor_id1,new_processor_id2,new_task_id2,new_period_id2);
			//�ύX�̕\��
			System.out.println("(task_id,processor_id,period_id)");
			System.out.println("swap:("+new_task_id1+","+new_processor_id1+","+new_period_id1+")��("+new_task_id2+","+new_processor_id2+","+new_period_id2+")");
		}
		return change;
	}
	//(�v���O�������̃e�[�u�������������Ȃ�)	
	//�ߖT����Swap:�قȂ�2�̒S���u�t�����ւ���(���\�b�h�g�p)
	//�ߖTswap��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_swap3b() throws Exception{
		ArrayList t_timetable = null;
		boolean change = false;
		int min_val,min_off,new_val,new_off;
		int new_task_id1 = 0,new_processor_id1 = 0,new_period_id1 = 0,new_task_id2 = 0,new_processor_id2 = 0,new_period_id2 = 0;
		int next = 0;
		//�l�̕]��
		min_val = valuateTimeTableSQLFile();
		min_off = timetable.countOffence();
		t_timetable = timetable.getTimeTableRowsCopy();
		//����p�̃e�[�u�����쐬
		createTTimeTable();
		//���݂̎��Ԋ��̎擾
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
				//�قȂ�u�`���ǂ���
				if(task_id1!=task_id2){
					//(task_id1,period_id1,processor_id1)��(task_id2,period_id2,processor_id2)����菜��
					deleteTTimeTableRow(task_id1,period_id1,processor_id1);
					deleteTTimeTableRow(task_id2,period_id2,processor_id2);
					timetable.deleteTimeTableRow(task_id1,period_id1,processor_id1);
					timetable.deleteTimeTableRow(task_id2,period_id2,processor_id2);
					//task_id1��task_id2�̎�ނ𒲂ׂ�
					Lecture l1 = SearchLecture(task_id1);
					qualification_id1 = l1.getQualification_id();
					Lecture l2 = SearchLecture(task_id2);
					qualification_id2 = l2.getQualification_id();
					//(task_id1,period_id1,processor_id2)���L�����ǂ���
					//processor_id2��period_id1�ɉ\���ǂ����H
					//�u�tt��processor_id2�ł���
					Teacher t = SearchTeacher(processor_id2);
					if(t.isQualification(qualification_id1)){
						//processor_id2��period_id1�ɉ\
						if(t.isPeriod(period_id1)){
							//�u�t�͓������Ԃɕ����̍u�`�����Ȃ�
							if(timetable.isNotTasks(processor_id2,period_id1)){
								//(task_id2,period_id2,processor_id1)���L�����ǂ���
								//processor_id1��period_id2�ɒS���\���ǂ����H
								//�u�tt2��processor_id1�ł���
								Teacher t2 = SearchTeacher(processor_id1);
								if(t2.isQualification(qualification_id2)){
									//processor_id1��period_id2�ɉ\
									if(t2.isPeriod(period_id2)){
										//�u�t�͓������Ԃɕ����̍u�`�����Ȃ�
										if(timetable.isNotTasks(processor_id1,period_id2)){
											//(task_id1,period_id1,processor_id2)�̒ǉ�
											insertTTimeTableRow(task_id1,period_id1,processor_id2);
											//(task_id2,period_id2,processor_id1)�̒ǉ�
											insertTTimeTableRow(task_id2,period_id2,processor_id1);
											timetable.insertTimeTableRow(task_id1,period_id1,processor_id2);
											timetable.insertTimeTableRow(task_id2,period_id2,processor_id1);
											new_off = timetable.countOffence();
											new_val = valuateTTimeTableFile();
											if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
											{
												//�ω������������Ă���
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
											//�ǉ�����(task_id1,period_id1,processor_id2)�̍폜
											deleteTTimeTableRow(task_id1,period_id1,processor_id2);
											//�ǉ�����(task_id2,period_id2,processor_id1)�̍폜
											deleteTTimeTableRow(task_id2,period_id2,processor_id1);
											timetable.deleteTimeTableRow(task_id1,period_id1,processor_id2);
											timetable.deleteTimeTableRow(task_id2,period_id2,processor_id1);
											}
									}
								}
							}
						}
					}
					//��菜����(task_id1,period_id1,processor_id1)��(task_id2,period_id2,processor_id2)��ǉ�����
					insertTTimeTableRow(task_id1,period_id1,processor_id1);
					insertTTimeTableRow(task_id2,period_id2,processor_id2);
					timetable.insertTimeTableRow(task_id1,period_id1,processor_id1);
					timetable.insertTimeTableRow(task_id2,period_id2,processor_id2);
				}
			}
			next++;
		}
		//����p�e�[�u���̍폜
		deleteTTimeTable();
		//�������P���ꂽ��
		if(change){
			System.out.println("min_val:"+min_val);
			updateProcessorTimeTable(new_processor_id2,new_processor_id1,new_task_id1,new_period_id1);
			timetable.updateProcessorTimeTable(new_processor_id2,new_processor_id1,new_task_id1,new_period_id1);
			updateProcessorTimeTable(new_processor_id1,new_processor_id2,new_task_id2,new_period_id2);
			timetable.updateProcessorTimeTable(new_processor_id1,new_processor_id2,new_task_id2,new_period_id2);
			//�ύX�̕\��
			System.out.println("(task_id,processor_id,period_id)");
			System.out.println("swap:("+new_task_id1+","+new_processor_id1+","+new_period_id1+")��("+new_task_id2+","+new_processor_id2+","+new_period_id2+")");
		}
		return change;
	}

	//(�v���O�������̃e�[�u�������������Ȃ�)	
	//�ߖT����Swap:�قȂ�2�̒S���u�t�����ւ���(���\�b�h�g�p)
	//�ߖTswap��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_swap3c() throws Exception{
		ArrayList t_timetable = null;
		boolean change = false;
		int min_val,min_off,new_val,new_off;
		int new_task_id1 = 0,new_processor_id1 = 0,new_period_id1 = 0,new_task_id2 = 0,new_processor_id2 = 0,new_period_id2 = 0;
		int next = 0;
		//�l�̕]��
		min_val = valuateTimeTableSQLFile();
		min_off = timetable.countOffence();
		t_timetable = timetable.getTimeTableRowsCopy();
		//���݂̎��Ԋ��̎擾
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
				//�قȂ�u�`���ǂ���
				if(task_id1!=task_id2){
					//(task_id1,period_id1,processor_id1)��(task_id2,period_id2,processor_id2)����菜��
					deleteTimeTableRow(task_id1,period_id1,processor_id1);
					deleteTimeTableRow(task_id2,period_id2,processor_id2);
					timetable.deleteTimeTableRow(task_id1,period_id1,processor_id1);
					timetable.deleteTimeTableRow(task_id2,period_id2,processor_id2);
					//task_id1��task_id2�̎�ނ𒲂ׂ�
					Lecture l1 = SearchLecture(task_id1);
					qualification_id1 = l1.getQualification_id();
					Lecture l2 = SearchLecture(task_id2);
					qualification_id2 = l2.getQualification_id();
					//(task_id1,period_id1,processor_id2)���L�����ǂ���
					//processor_id2��period_id1�ɉ\���ǂ����H
					//�u�tt��processor_id2�ł���
					Teacher t = SearchTeacher(processor_id2);
					if(t.isQualification(qualification_id1)){
						//processor_id2��period_id1�ɉ\
						if(t.isPeriod(period_id1)){
							//�u�t�͓������Ԃɕ����̍u�`�����Ȃ�
							if(timetable.isNotTasks(processor_id2,period_id1)){
								//(task_id2,period_id2,processor_id1)���L�����ǂ���
								//processor_id1��period_id2�ɒS���\���ǂ����H
								//�u�tt2��processor_id1�ł���
								Teacher t2 = SearchTeacher(processor_id1);
								if(t2.isQualification(qualification_id2)){
									//processor_id1��period_id2�ɉ\
									if(t2.isPeriod(period_id2)){
										//�u�t�͓������Ԃɕ����̍u�`�����Ȃ�
										if(timetable.isNotTasks(processor_id1,period_id2)){
											//(task_id1,period_id1,processor_id2)�̒ǉ�
											insertTimeTableRow(task_id1,period_id1,processor_id2);
											//(task_id2,period_id2,processor_id1)�̒ǉ�
											insertTimeTableRow(task_id2,period_id2,processor_id1);
											timetable.insertTimeTableRow(task_id1,period_id1,processor_id2);
											timetable.insertTimeTableRow(task_id2,period_id2,processor_id1);
											new_off = timetable.countOffence();
											new_val = valuateTimeTableSQLFile();
											if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
											{
												//�ω������������Ă���
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
											//�ǉ�����(task_id1,period_id1,processor_id2)�̍폜
											deleteTimeTableRow(task_id1,period_id1,processor_id2);
											//�ǉ�����(task_id2,period_id2,processor_id1)�̍폜
											deleteTimeTableRow(task_id2,period_id2,processor_id1);
											timetable.deleteTimeTableRow(task_id1,period_id1,processor_id2);
											timetable.deleteTimeTableRow(task_id2,period_id2,processor_id1);
											}
									}
								}
							}
						}
					}
					//��菜����(task_id1,period_id1,processor_id1)��(task_id2,period_id2,processor_id2)��ǉ�����
					insertTimeTableRow(task_id1,period_id1,processor_id1);
					insertTimeTableRow(task_id2,period_id2,processor_id2);
					timetable.insertTimeTableRow(task_id1,period_id1,processor_id1);
					timetable.insertTimeTableRow(task_id2,period_id2,processor_id2);
				}
			}
			next++;
		}
		//�������P���ꂽ��
		if(change){
			updateProcessorTimeTable(new_processor_id2,new_processor_id1,new_task_id1,new_period_id1);
			timetable.updateProcessorTimeTable(new_processor_id2,new_processor_id1,new_task_id1,new_period_id1);
			updateProcessorTimeTable(new_processor_id1,new_processor_id2,new_task_id2,new_period_id2);
			timetable.updateProcessorTimeTable(new_processor_id1,new_processor_id2,new_task_id2,new_period_id2);
			//�ύX�̕\��
			System.out.println("(task_id,processor_id,period_id)");
			System.out.println("swap:("+new_task_id1+","+new_processor_id1+","+new_period_id1+")��("+new_task_id2+","+new_processor_id2+","+new_period_id2+")");
		}
		return change;
	}
	
	//�ߖT����add:�u�`�̒S���u�t��ǉ�����
	//�ߖTadd��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_add() throws Exception{
		Connection con = null;
		ResultSet rs = null;
		ResultSet rs2 = null;
		Statement smt = null;
		Statement smt2 = null;
		Statement smt3 = null;
		boolean change = false;
		ArrayList new_teacher_ids = null;
		int min_val,min_off,new_val,new_off;
		int new_task_id = 0,new_period_id = 0;
		//�l�̕]��
		min_val = valuateTimeTableSQLFile();
		min_off = count_offence("timetableSQL");
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			smt2 = con.createStatement();
			smt3 = con.createStatement();
			//����p�̃e�[�u�����쐬
			String sql = "create table t_timetable as select * from timetableSQL";
			smt.executeUpdate(sql);
			//���݂̎��Ԋ��̍u�`�Ƃ��̒S���l���ƊJ�u���Ԃ��擾
			sql = "select task_id , period_id, count(*) as num from timetableSQL group by task_id ,period_id order by task_id";
			rs = smt.executeQuery(sql);
			int task_id;
			int qualification_id = 0;
			int period_id;
			int num;
			int low;
			while(rs.next()) {
				task_id = rs.getInt("task_id");
				period_id = rs.getInt("period_id");
				num = rs.getInt("num");
				//task_id�ł���u�`�̎擾
				Lecture l = SearchLecture(task_id);
				qualification_id = l.getQualification_id();
				//period_id�ɒS���\�ȍu�t��id�����Ă���
				ArrayList t_teachers2 = null;
				//task_id�̒S���l������ɑ���Ȃ��l���̎擾
				low = l.getRequired_processors_ub()-num;
				//System.out.println("low:"+low);
				//�����u�`�̒S���l������ɑ���Ă��Ȃ��Ȃ炻�̍u�`��S���ł���S�u�t��processor_id���擾
				if(low>0){
					t_teachers2 = new ArrayList();
					Iterator i3 = teachers.iterator();
					while(i3.hasNext()){
						Teacher t = (Teacher)i3.next();
						if(t.isQualification(qualification_id)){
							//t��period_id�ŒS���\
							if(t.isPeriod(period_id)){
								int processor_id = t.getProcessor_id();
								//�u�t�͓������Ԃɕ����̍u�`�����Ȃ�
								String sql2 = "select * from t_timetable where processor_id='"+processor_id+"' and period_id = '" + period_id +"'";
								rs2 = smt2.executeQuery(sql2);
								if(rs2.next()==false){
									t_teachers2.add(new Integer(processor_id));	
									//System.out.println("�ǉ��\:"+processor_id);
								}
							}
						}
					}
				}
				//�S���l���̑���
				for(int j = 0;j<low;j++){
					//System.out.println("task_id:"+task_id);
					ArrayList t_teachers = null;
					Enumeration e = new CombEnum(t_teachers2.toArray(), j+1);
					//���̒ǉ�����u�t�̑g�ݍ��킹�͂��邩
					while(e.hasMoreElements()) {
						t_teachers = new ArrayList();
						Object[] a = (Object[])e.nextElement();
						//System.out.println("�ǉ�����l��:"+(j+1));
						for(int num2=0; num2<a.length; num2++){
							int processor_id = ((Integer)a[num2]).intValue();
							//�u�t�̒ǉ�
							String sql3="insert into t_timetable(task_id,period_id , processor_id) values("+ "'" + task_id + "','" +period_id + "','" + processor_id+"')";
							smt3.executeUpdate(sql3);
							t_teachers.add(new Integer(processor_id));
						}
						//�l�̕]��
						new_off = count_offence("t_timetable");
						new_val = valuateTTimeTableFile();
						if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
						{
							//�ω������������Ă���
							new_teacher_ids = t_teachers;
							new_task_id = task_id;
							new_period_id = period_id;
							min_off = new_off;
							min_val = new_val;
							change = true;
						}
						//�ǉ������u�t�̍폜
						Iterator i5 = t_teachers.iterator();
						while(i5.hasNext()){
							Integer t = (Integer)i5.next();
							String sql2="delete from t_timetable where task_id ='"+task_id+"' and period_id = '"+period_id+"' and processor_id = '"+t.intValue()+"'";
							smt2.executeUpdate(sql2);
						}
					}
				}
			}
			//����p�e�[�u���̍폜
			sql = "drop table t_timetable";
			smt.executeUpdate(sql);
			//�������P���ꂽ��
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
		}catch(SQLException e) {
			System.err.println(e);
			throw e;
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs2!=null){
				rs2.close();
				rs2 = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt2!=null){
				smt2.close();
				smt2 = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt3!=null){
				smt3.close();
				smt3 = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		return change;		
	}

	//�ߖT����add:�u�`�̒S���u�t��ǉ�����
	//�ߖTadd��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_add1() throws Exception{
		Connection con = null;
		ResultSet rs = null;
		ResultSet rs2 = null;
		Statement smt = null;
		Statement smt2 = null;
		Statement smt3 = null;
		boolean change = false;
		ArrayList new_teacher_ids = null;
		int min_val,min_off,new_val,new_off;
		int new_task_id = 0,new_period_id = 0;
		//�l�̕]��
		min_val = valuateTimeTableSQLFile();
		min_off = count_offence2("timetableSQL");
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			smt2 = con.createStatement();
			smt3 = con.createStatement();
			//����p�̃e�[�u�����쐬
			String sql = "create table t_timetable as select * from timetableSQL";
			smt.executeUpdate(sql);
			//���݂̎��Ԋ��̍u�`�Ƃ��̒S���l���ƊJ�u���Ԃ��擾
			sql = "select task_id , period_id, count(*) as num from timetableSQL group by task_id ,period_id order by task_id";
			rs = smt.executeQuery(sql);
			int task_id;
			int qualification_id = 0;
			int period_id;
			int num;
			int low;
			while(rs.next()) {
				task_id = rs.getInt("task_id");
				period_id = rs.getInt("period_id");
				num = rs.getInt("num");
				//task_id�ł���u�`�̎擾
				Lecture l = SearchLecture(task_id);
				qualification_id = l.getQualification_id();
				//period_id�ɒS���\�ȍu�t��id�����Ă���
				ArrayList t_teachers2 = null;
				//task_id�̒S���l������ɑ���Ȃ��l���̎擾
				low = l.getRequired_processors_ub()-num;
				//System.out.println("low:"+low);
				//�����u�`�̒S���l������ɑ���Ă��Ȃ��Ȃ炻�̍u�`��S���ł���S�u�t��processor_id���擾
				if(low>0){
					t_teachers2 = new ArrayList();
					Iterator i3 = teachers.iterator();
					while(i3.hasNext()){
						Teacher t = (Teacher)i3.next();
						if(t.isQualification(qualification_id)){
							//t��period_id�ŒS���\
							if(t.isPeriod(period_id)){
								int processor_id = t.getProcessor_id();
								//�u�t�͓������Ԃɕ����̍u�`�����Ȃ�
								String sql2 = "select * from t_timetable where processor_id='"+processor_id+"' and period_id = '" + period_id +"'";
								rs2 = smt2.executeQuery(sql2);
								if(rs2.next()==false){
									t_teachers2.add(new Integer(processor_id));	
									//System.out.println("�ǉ��\:"+processor_id);
								}
							}
						}
					}
				}
				//�S���l���̑���
				for(int j = 0;j<low;j++){
					//System.out.println("task_id:"+task_id);
					ArrayList t_teachers = null;
					Enumeration e = new CombEnum(t_teachers2.toArray(), j+1);
					//���̒ǉ�����u�t�̑g�ݍ��킹�͂��邩
					while(e.hasMoreElements()) {
						t_teachers = new ArrayList();
						Object[] a = (Object[])e.nextElement();
						//System.out.println("�ǉ�����l��:"+(j+1));
						for(int num2=0; num2<a.length; num2++){
							int processor_id = ((Integer)a[num2]).intValue();
							//�u�t�̒ǉ�
							String sql3="insert into t_timetable(task_id,period_id , processor_id) values("+ "'" + task_id + "','" +period_id + "','" + processor_id+"')";
							smt3.executeUpdate(sql3);
							t_teachers.add(new Integer(processor_id));
							//System.out.println("�ǉ�:"+processor_id);
						}
						//�l�̕]��
						//new_off = count_offence("t_timetable");
						new_off = count_offence2("t_timetable");
						//new_val = valuate("t_timetable");
						new_val = valuateTTimeTableFile();
						if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
						{
							//�ω������������Ă���
							new_teacher_ids = t_teachers;
							new_task_id = task_id;
							new_period_id = period_id;
							min_off = new_off;
							min_val = new_val;
							change = true;
						}
						//�ǉ������u�t�̍폜
						Iterator i5 = t_teachers.iterator();
						while(i5.hasNext()){
							Integer t = (Integer)i5.next();
							String sql2="delete from t_timetable where task_id ='"+task_id+"' and period_id = '"+period_id+"' and processor_id = '"+t.intValue()+"'";
							smt2.executeUpdate(sql2);
						}
					}
				}
			}
			//����p�e�[�u���̍폜
			sql = "drop table t_timetable";
			smt.executeUpdate(sql);
			//�������P���ꂽ��
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
		}catch(SQLException e) {
			System.err.println(e);
			throw e;
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs2!=null){
				rs2.close();
				rs2 = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt2!=null){
				smt2.close();
				smt2 = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt3!=null){
				smt3.close();
				smt3 = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		return change;		
	}
	//����p�̃e�[�u�����g�p���Ȃ�
	//�ߖT����add:�u�`�̒S���u�t��ǉ�����(���\�b�h�g�p)
	//�ߖTadd��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_add2b() throws Exception{
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		boolean change = false;
		ArrayList new_teacher_ids = null;
		int min_val,min_off,new_val,new_off;
		int new_task_id = 0,new_period_id = 0;
		//�l�̕]��
		min_val = valuateTimeTableSQLFile();
		min_off = count_offence2("timetableSQL");
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			//���݂̎��Ԋ��̍u�`�Ƃ��̒S���l���ƊJ�u���Ԃ��擾
			String sql = "select task_id , period_id, count(*) as num from timetableSQL group by task_id ,period_id order by task_id";
			rs = smt.executeQuery(sql);
			int task_id;
			int qualification_id = 0;
			int period_id;
			int num;
			int low;
			while(rs.next()) {
				task_id = rs.getInt("task_id");
				period_id = rs.getInt("period_id");
				num = rs.getInt("num");
				//task_id�ł���u�`�̎擾
				Lecture l = SearchLecture(task_id);
				qualification_id = l.getQualification_id();
				//period_id�ɒS���\�ȍu�t��id�����Ă���
				ArrayList t_teachers2 = null;
				//task_id�̒S���l������ɑ���Ȃ��l���̎擾
				low = l.getRequired_processors_ub()-num;
				//System.out.println("low:"+low);
				//�����u�`�̒S���l������ɑ���Ă��Ȃ��Ȃ炻�̍u�`��S���ł���S�u�t��processor_id���擾
				if(low>0){
					t_teachers2 = new ArrayList();
					Iterator i3 = teachers.iterator();
					while(i3.hasNext()){
						Teacher t = (Teacher)i3.next();
						if(t.isQualification(qualification_id)){
							//t��period_id�ŒS���\
							if(t.isPeriod(period_id)){
								int processor_id = t.getProcessor_id();
								//�u�t�͓������Ԃɕ����̍u�`�����Ȃ�
								if(isNotTasksb(processor_id,period_id)){
									t_teachers2.add(new Integer(processor_id));	
									//System.out.println("�ǉ��\:"+processor_id);
								}
							}
						}
					}
				}
				//�S���l���̑���
				for(int j = 0;j<low;j++){
					//System.out.println("task_id:"+task_id);
					ArrayList t_teachers = null;
					Enumeration e = new CombEnum(t_teachers2.toArray(), j+1);
					//���̒ǉ�����u�t�̑g�ݍ��킹�͂��邩
					while(e.hasMoreElements()) {
						t_teachers = new ArrayList();
						Object[] a = (Object[])e.nextElement();
						//System.out.println("�ǉ�����l��:"+(j+1));
						for(int num2=0; num2<a.length; num2++){
							int processor_id = ((Integer)a[num2]).intValue();
							//�u�t�̒ǉ�
							insertTimeTableRow(task_id,period_id,processor_id);
							t_teachers.add(new Integer(processor_id));
							//System.out.println("�ǉ�:"+processor_id);
						}
						//�l�̕]��
						new_off = count_offence2("t_timetableSQL");
						new_val = valuateTimeTableSQLFile();
						if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
						{
							//�ω������������Ă���
							new_teacher_ids = t_teachers;
							new_task_id = task_id;
							new_period_id = period_id;
							min_off = new_off;
							min_val = new_val;
							change = true;
						}
						//�ǉ������u�t�̍폜
						Iterator i5 = t_teachers.iterator();
						while(i5.hasNext()){
							Integer t = (Integer)i5.next();
							deleteTimeTableRow(task_id,period_id,t.intValue());
						}
					}
				}
			}
			//�������P���ꂽ��
			if(change){
				//�ύX�̕\��
				System.out.println("(task_id,processor_id,period_id)");
				Iterator i = new_teacher_ids.iterator();
				int processor_id;
				while(i.hasNext()){
					processor_id = ((Integer)i.next()).intValue();
					System.out.println("add:("+new_task_id+","+processor_id+","+new_period_id+")");
					insertTimeTableRow(new_task_id,new_period_id,processor_id);
				}
			}
		}catch(SQLException e) {
			System.err.println(e);
			throw e;
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		return change;		
	}
	
	//�ߖT����add:�u�`�̒S���u�t��ǉ�����(���\�b�h�g�p)
	//�ߖTadd��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_add2() throws Exception{
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		boolean change = false;
		ArrayList new_teacher_ids = null;
		int min_val,min_off,new_val,new_off;
		int new_task_id = 0,new_period_id = 0;
		//�l�̕]��
		//min_val = valuate("timetableSQL");
		min_val = valuateTimeTableSQLFile();
		min_off = count_offence2("timetableSQL");
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			//����p�̃e�[�u�����쐬
			createTTimeTable();
			//���݂̎��Ԋ��̍u�`�Ƃ��̒S���l���ƊJ�u���Ԃ��擾
			String sql = "select task_id , period_id, count(*) as num from timetableSQL group by task_id ,period_id order by task_id";
			rs = smt.executeQuery(sql);
			int task_id;
			int qualification_id = 0;
			int period_id;
			int num;
			int low;
			while(rs.next()) {
				task_id = rs.getInt("task_id");
				period_id = rs.getInt("period_id");
				num = rs.getInt("num");
				//task_id�ł���u�`�̎擾
				Lecture l = SearchLecture(task_id);
				qualification_id = l.getQualification_id();
				//period_id�ɒS���\�ȍu�t��id�����Ă���
				ArrayList t_teachers2 = null;
				//task_id�̒S���l������ɑ���Ȃ��l���̎擾
				low = l.getRequired_processors_ub()-num;
				//System.out.println("low:"+low);
				//�����u�`�̒S���l������ɑ���Ă��Ȃ��Ȃ炻�̍u�`��S���ł���S�u�t��processor_id���擾
				if(low>0){
					t_teachers2 = new ArrayList();
					Iterator i3 = teachers.iterator();
					while(i3.hasNext()){
						Teacher t = (Teacher)i3.next();
						if(t.isQualification(qualification_id)){
							//t��period_id�ŒS���\
							if(t.isPeriod(period_id)){
								int processor_id = t.getProcessor_id();
								//�u�t�͓������Ԃɕ����̍u�`�����Ȃ�
								if(isNotTasks(processor_id,period_id)){
									t_teachers2.add(new Integer(processor_id));	
									//System.out.println("�ǉ��\:"+processor_id);
								}
							}
						}
					}
				}
				//�S���l���̑���
				for(int j = 0;j<low;j++){
					//System.out.println("task_id:"+task_id);
					ArrayList t_teachers = null;
					Enumeration e = new CombEnum(t_teachers2.toArray(), j+1);
					//���̒ǉ�����u�t�̑g�ݍ��킹�͂��邩
					while(e.hasMoreElements()) {
						t_teachers = new ArrayList();
						Object[] a = (Object[])e.nextElement();
						//System.out.println("�ǉ�����l��:"+(j+1));
						for(int num2=0; num2<a.length; num2++){
							int processor_id = ((Integer)a[num2]).intValue();
							//�u�t�̒ǉ�
							insertTTimeTableRow(task_id,period_id,processor_id);
							t_teachers.add(new Integer(processor_id));
							//System.out.println("�ǉ�:"+processor_id);
						}
						//�l�̕]��
						//new_off = count_offence("t_timetable");
						new_off = count_offence2("t_timetable");
						//new_val = valuate("t_timetable");
						new_val = valuateTTimeTableFile();
						if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
						{
							//�ω������������Ă���
							new_teacher_ids = t_teachers;
							new_task_id = task_id;
							new_period_id = period_id;
							min_off = new_off;
							min_val = new_val;
							change = true;
						}
						//�ǉ������u�t�̍폜
						Iterator i5 = t_teachers.iterator();
						while(i5.hasNext()){
							Integer t = (Integer)i5.next();
							deleteTTimeTableRow(task_id,period_id,t.intValue());
						}
					}
				}
			}
			//����p�e�[�u���̍폜
			deleteTTimeTable();
			//�������P���ꂽ��
			if(change){
				//�ύX�̕\��
				System.out.println("(task_id,processor_id,period_id)");
				Iterator i = new_teacher_ids.iterator();
				int processor_id;
				while(i.hasNext()){
					processor_id = ((Integer)i.next()).intValue();
					System.out.println("add:("+new_task_id+","+processor_id+","+new_period_id+")");
					insertTimeTableRow(new_task_id,new_period_id,processor_id);
				}
			}
		}catch(SQLException e) {
			System.err.println(e);
			throw e;
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		return change;		
	}
	
	//�ߖT����add:�u�`�̒S���u�t��ǉ�����(���\�b�h�g�p)
	//�ߖTadd��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_add3() throws Exception{
		boolean change = false;
		ArrayList new_teacher_ids = null;
		int min_val,min_off,new_val,new_off;
		int new_task_id = 0,new_period_id = 0;
		timetable.loadTimeTable();
		//�l�̕]��
		min_val = valuateTimeTableSQLFile();
		min_off = timetable.countOffence();
		//����p�̃e�[�u�����쐬
		createTTimeTable();
		int task_id;
		int qualification_id = 0;
		int period_id;
		int num;
		int low;
		//���݂̎��Ԋ��̍u�`�Ƃ��̒S���l���ƊJ�u���Ԃ��擾
		ArrayList list = timetable.getTimeTableTasksNum();
		Iterator i_list = list.iterator();
		while(i_list.hasNext()){
			TaskPeriodNum tpn = (TaskPeriodNum)i_list.next();
			task_id = tpn.getTask_id();
			period_id = tpn.getPeriod_id();
			num = tpn.getNum();
			//task_id�ł���u�`�̎擾
			Lecture l = SearchLecture(task_id);
			qualification_id = l.getQualification_id();
			//period_id�ɒS���\�ȍu�t��id�����Ă���
			ArrayList t_teachers2 = null;
			//task_id�̒S���l������ɑ���Ȃ��l���̎擾
			low = l.getRequired_processors_ub()-num;
			//System.out.println("low:"+low);
			//�����u�`�̒S���l������ɑ���Ă��Ȃ��Ȃ炻�̍u�`��S���ł���S�u�t��processor_id���擾
			if(low>0){
				t_teachers2 = new ArrayList();
				Iterator i3 = teachers.iterator();
				while(i3.hasNext()){
					Teacher t = (Teacher)i3.next();
					if(t.isQualification(qualification_id)){
						//t��period_id�ŒS���\
						if(t.isPeriod(period_id)){
							int processor_id = t.getProcessor_id();
							//�u�t�͓������Ԃɕ����̍u�`�����Ȃ�
							if(timetable.isNotTasks(processor_id,period_id)){
								t_teachers2.add(new Integer(processor_id));	
								//System.out.println("�ǉ��\:"+processor_id);
							}
						}
					}
				}
			}
			//�S���l���̑���
			for(int j = 0;j<low;j++){
				//System.out.println("task_id:"+task_id);
				ArrayList t_teachers = null;
				Enumeration e = new CombEnum(t_teachers2.toArray(), j+1);
				//���̒ǉ�����u�t�̑g�ݍ��킹�͂��邩
				while(e.hasMoreElements()) {
					t_teachers = new ArrayList();
					Object[] a = (Object[])e.nextElement();
					//System.out.println("�ǉ�����l��:"+(j+1));
					for(int num2=0; num2<a.length; num2++){
						int processor_id = ((Integer)a[num2]).intValue();
						//�u�t�̒ǉ�
						insertTTimeTableRow(task_id,period_id,processor_id);
						timetable.insertTimeTableRow(task_id,period_id,processor_id);
						t_teachers.add(new Integer(processor_id));
					}
					//�l�̕]��
					new_off = timetable.countOffence();
					new_val = valuateTTimeTableFile();
					if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
					{
						//�ω������������Ă���
						new_teacher_ids = t_teachers;
						new_task_id = task_id;
						new_period_id = period_id;
						min_off = new_off;
						min_val = new_val;
						change = true;
					}
					//�ǉ������u�t�̍폜
					Iterator i5 = t_teachers.iterator();
					while(i5.hasNext()){
						Integer t = (Integer)i5.next();
						deleteTTimeTableRow(task_id,period_id,t.intValue());
						timetable.deleteTimeTableRow(task_id,period_id,t.intValue());
					}
				}
			}
		}
		//����p�e�[�u���̍폜
		deleteTTimeTable();
		//�������P���ꂽ��
		if(change){
			//�ύX�̕\��
			System.out.println("(task_id,processor_id,period_id)");
			Iterator i = new_teacher_ids.iterator();
			int processor_id;
			while(i.hasNext()){
				processor_id = ((Integer)i.next()).intValue();
				System.out.println("add:("+new_task_id+","+processor_id+","+new_period_id+")");
				insertTimeTableRow(new_task_id,new_period_id,processor_id);
				//timetable.insertTimeTableRow(new_task_id,new_period_id,processor_id);
			}
		}
		return change;		
	}
	//(�v���O�������̃e�[�u�������������Ȃ�)	
	//�ߖT����add:�u�`�̒S���u�t��ǉ�����(���\�b�h�g�p)
	//�ߖTadd��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_add3b() throws Exception{
		boolean change = false;
		ArrayList new_teacher_ids = null;
		int min_val,min_off,new_val,new_off;
		int new_task_id = 0,new_period_id = 0;
		//�l�̕]��
		min_val = valuateTimeTableSQLFile();
		min_off = timetable.countOffence();
		//����p�̃e�[�u�����쐬
		createTTimeTable();
		int task_id;
		int qualification_id = 0;
		int period_id;
		int num;
		int low;
		//���݂̎��Ԋ��̍u�`�Ƃ��̒S���l���ƊJ�u���Ԃ��擾
		ArrayList list = timetable.getTimeTableTasksNum();
		Iterator i_list = list.iterator();
		while(i_list.hasNext()){
			TaskPeriodNum tpn = (TaskPeriodNum)i_list.next();
			task_id = tpn.getTask_id();
			period_id = tpn.getPeriod_id();
			num = tpn.getNum();
			//task_id�ł���u�`�̎擾
			Lecture l = SearchLecture(task_id);
			qualification_id = l.getQualification_id();
			//period_id�ɒS���\�ȍu�t��id�����Ă���
			ArrayList t_teachers2 = null;
			//task_id�̒S���l������ɑ���Ȃ��l���̎擾
			low = l.getRequired_processors_ub()-num;
			//System.out.println("low:"+low);
			//�����u�`�̒S���l������ɑ���Ă��Ȃ��Ȃ炻�̍u�`��S���ł���S�u�t��processor_id���擾
			if(low>0){
				t_teachers2 = new ArrayList();
				Iterator i3 = teachers.iterator();
				while(i3.hasNext()){
					Teacher t = (Teacher)i3.next();
					if(t.isQualification(qualification_id)){
						//t��period_id�ŒS���\
						if(t.isPeriod(period_id)){
							int processor_id = t.getProcessor_id();
							//�u�t�͓������Ԃɕ����̍u�`�����Ȃ�
							if(timetable.isNotTasks(processor_id,period_id)){
								t_teachers2.add(new Integer(processor_id));	
								//System.out.println("�ǉ��\:"+processor_id);
							}
						}
					}
				}
			}
			//�S���l���̑���
			for(int j = 0;j<low;j++){
				//System.out.println("task_id:"+task_id);
				ArrayList t_teachers = null;
				Enumeration e = new CombEnum(t_teachers2.toArray(), j+1);
				//���̒ǉ�����u�t�̑g�ݍ��킹�͂��邩
				while(e.hasMoreElements()) {
					t_teachers = new ArrayList();
					Object[] a = (Object[])e.nextElement();
					//System.out.println("�ǉ�����l��:"+(j+1));
					for(int num2=0; num2<a.length; num2++){
						int processor_id = ((Integer)a[num2]).intValue();
						//�u�t�̒ǉ�
						insertTTimeTableRow(task_id,period_id,processor_id);
						timetable.insertTimeTableRow(task_id,period_id,processor_id);
						t_teachers.add(new Integer(processor_id));
					}
					//�l�̕]��
					new_off = timetable.countOffence();
					new_val = valuateTTimeTableFile();
					if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
					{
						//�ω������������Ă���
						new_teacher_ids = t_teachers;
						new_task_id = task_id;
						new_period_id = period_id;
						min_off = new_off;
						min_val = new_val;
						change = true;
					}
					//�ǉ������u�t�̍폜
					Iterator i5 = t_teachers.iterator();
					while(i5.hasNext()){
						Integer t = (Integer)i5.next();
						deleteTTimeTableRow(task_id,period_id,t.intValue());
						timetable.deleteTimeTableRow(task_id,period_id,t.intValue());
					}
				}
			}
		}
		//����p�e�[�u���̍폜
		deleteTTimeTable();
		//�������P���ꂽ��
		if(change){
			//�ύX�̕\��
			System.out.println("(task_id,processor_id,period_id)");
			Iterator i = new_teacher_ids.iterator();
			int processor_id;
			while(i.hasNext()){
				processor_id = ((Integer)i.next()).intValue();
				System.out.println("add:("+new_task_id+","+processor_id+","+new_period_id+")");
				insertTimeTableRow(new_task_id,new_period_id,processor_id);
				timetable.insertTimeTableRow(new_task_id,new_period_id,processor_id);
			}
		}
		return change;		
	}

	//(�v���O�������̃e�[�u�������������Ȃ�)	
	//�ߖT����add:�u�`�̒S���u�t��ǉ�����(���\�b�h�g�p)
	//�ߖTadd��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_add3c() throws Exception{
		boolean change = false;
		ArrayList new_teacher_ids = null;
		int min_val,min_off,new_val,new_off;
		int new_task_id = 0,new_period_id = 0;
		//�l�̕]��
		min_val = valuateTimeTableSQLFile();
		min_off = timetable.countOffence();
		int task_id;
		int qualification_id = 0;
		int period_id;
		int num;
		int low;
		//���݂̎��Ԋ��̍u�`�Ƃ��̒S���l���ƊJ�u���Ԃ��擾
		ArrayList list = timetable.getTimeTableTasksNum();
		Iterator i_list = list.iterator();
		while(i_list.hasNext()){
			TaskPeriodNum tpn = (TaskPeriodNum)i_list.next();
			task_id = tpn.getTask_id();
			period_id = tpn.getPeriod_id();
			num = tpn.getNum();
			//task_id�ł���u�`�̎擾
			Lecture l = SearchLecture(task_id);
			qualification_id = l.getQualification_id();
			//period_id�ɒS���\�ȍu�t��id�����Ă���
			ArrayList t_teachers2 = null;
			//task_id�̒S���l������ɑ���Ȃ��l���̎擾
			low = l.getRequired_processors_ub()-num;
			//System.out.println("low:"+low);
			//�����u�`�̒S���l������ɑ���Ă��Ȃ��Ȃ炻�̍u�`��S���ł���S�u�t��processor_id���擾
			if(low>0){
				t_teachers2 = new ArrayList();
				Iterator i3 = teachers.iterator();
				while(i3.hasNext()){
					Teacher t = (Teacher)i3.next();
					if(t.isQualification(qualification_id)){
						//t��period_id�ŒS���\
						if(t.isPeriod(period_id)){
							int processor_id = t.getProcessor_id();
							//�u�t�͓������Ԃɕ����̍u�`�����Ȃ�
							if(timetable.isNotTasks(processor_id,period_id)){
								t_teachers2.add(new Integer(processor_id));	
								//System.out.println("�ǉ��\:"+processor_id);
							}
						}
					}
				}
			}
			//�S���l���̑���
			for(int j = 0;j<low;j++){
				//System.out.println("task_id:"+task_id);
				ArrayList t_teachers = null;
				Enumeration e = new CombEnum(t_teachers2.toArray(), j+1);
				//���̒ǉ�����u�t�̑g�ݍ��킹�͂��邩
				while(e.hasMoreElements()) {
					t_teachers = new ArrayList();
					Object[] a = (Object[])e.nextElement();
					//System.out.println("�ǉ�����l��:"+(j+1));
					for(int num2=0; num2<a.length; num2++){
						int processor_id = ((Integer)a[num2]).intValue();
						//�u�t�̒ǉ�
						insertTimeTableRow(task_id,period_id,processor_id);
						timetable.insertTimeTableRow(task_id,period_id,processor_id);
						t_teachers.add(new Integer(processor_id));
					}
					//�l�̕]��
					new_off = timetable.countOffence();
					new_val = valuateTimeTableSQLFile();
					if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
					{
						//�ω������������Ă���
						new_teacher_ids = t_teachers;
						new_task_id = task_id;
						new_period_id = period_id;
						min_off = new_off;
						min_val = new_val;
						change = true;
					}
					//�ǉ������u�t�̍폜
					Iterator i5 = t_teachers.iterator();
					while(i5.hasNext()){
						Integer t = (Integer)i5.next();
						deleteTimeTableRow(task_id,period_id,t.intValue());
						timetable.deleteTimeTableRow(task_id,period_id,t.intValue());
					}
				}
			}
		}
		//�������P���ꂽ��
		if(change){
			//�ύX�̕\��
			System.out.println("(task_id,processor_id,period_id)");
			Iterator i = new_teacher_ids.iterator();
			int processor_id;
			while(i.hasNext()){
				processor_id = ((Integer)i.next()).intValue();
				System.out.println("add:("+new_task_id+","+processor_id+","+new_period_id+")");
				insertTimeTableRow(new_task_id,new_period_id,processor_id);
				timetable.insertTimeTableRow(new_task_id,new_period_id,processor_id);
			}
		}
		return change;		
	}

	//�ߖT����leave:�u�`�̒S���u�t�����炷
	//�ߖTleave��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_leave() throws Exception{
		Connection con = null;
		ResultSet rs = null;
		ResultSet rs2 = null;
		Statement smt = null;
		Statement smt2 = null;
		Statement smt3 = null;
		boolean change = false;
		ArrayList new_teacher_ids = null;
		int min_val,min_off,new_val,new_off;
		int new_task_id = 0,new_period_id = 0;
		//�l�̕]��
		min_val = valuateTimeTableSQLFile();
		min_off = count_offence2("timetableSQL");
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			smt2 = con.createStatement();
			smt3 = con.createStatement();
			//����p�̃e�[�u�����쐬
			String sql = "create table t_timetable as select * from timetableSQL";
			smt.executeUpdate(sql);
			//���݂̎��Ԋ��̍u�`�Ƃ��̒S���l���ƊJ�u���Ԃ��擾
			sql = "select task_id , period_id, count(*) as num from timetableSQL group by task_id order by task_id";
			rs = smt.executeQuery(sql);
			int task_id;
			int period_id;
			int num;
			while(rs.next()) {
				task_id = rs.getInt("task_id");
				period_id = rs.getInt("period_id");
				num = rs.getInt("num");
				//task_id�ł���u�`�̎擾
				Lecture l = SearchLecture(task_id);
				//task_id�̒S���l�������ȏ�̐l���̎擾
				int up = num-l.getRequired_processors_lb();
				ArrayList t_teachers2 = null;
				//�����u�`�̒S���l�������ȏ�Ȃ�΂��̍u�`��S�����Ă���S�u�t��processor_id���擾
				if(up>0){
					t_teachers2 = new ArrayList();
					//period_id��task_id��S������u�t���擾
					String sql2 = "select processor_id from t_timetable where task_id='"+task_id+"' and period_id = '" + period_id +"'";
					rs2 = smt2.executeQuery(sql2);
					while(rs2.next()){
						t_teachers2.add(new Integer(rs2.getInt("processor_id")));
					}
				}
				//�S���l���̌���
				for(int j=0;j<up;j++){
					//�폜����processor_id��ۑ�����
					ArrayList t_teachers = null;
					Enumeration e = new CombEnum(t_teachers2.toArray(), j+1);
					//���̍폜����u�t�̑g�ݍ��킹�͂��邩
					while(e.hasMoreElements()) {
						t_teachers = new ArrayList();
						Object[] a = (Object[])e.nextElement();
						//System.out.println("�폜����l��:"+(j+1));
						for(int num2=0; num2<a.length; num2++){
							int processor_id = ((Integer)a[num2]).intValue();
							//�u�t�̍폜
							String sql3="delete from t_timetable where task_id ='"+task_id+"' and period_id = '"+period_id+"' and processor_id = '"+processor_id+"'";
							smt3.executeUpdate(sql3);
							t_teachers.add(new Integer(processor_id));
						}
						//�l�̕]��
						new_off = count_offence("t_timetable");
						new_val = valuateTTimeTableFile();
						if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
						{
							//�ω������������Ă���
							new_teacher_ids = t_teachers;
							new_task_id = task_id;
							new_period_id = period_id;
							min_off = new_off;
							min_val = new_val;
							change = true;
						}
						//�폜�����u�t��ǉ�
						Iterator i5 = t_teachers.iterator();
						while(i5.hasNext()){
							String sql3="insert into t_timetable(task_id,period_id , processor_id) values("+ "'" + task_id+ "','" +period_id + "','" + ((Integer)i5.next()).intValue()+"')";
							smt3.executeUpdate(sql3);
						}
					}
				}
			}
			//����p�e�[�u���̍폜
			sql = "drop table t_timetable";
			smt.executeUpdate(sql);
			//�������P���ꂽ��
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
			throw e;
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs2!=null){
				rs2.close();
				rs2 = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt2!=null){
				smt2.close();
				smt2 = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt3!=null){
				smt3.close();
				smt3 = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		return change;		
	}

	//�ߖT����leave:�u�`�̒S���u�t�����炷
	//�ߖTleave��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_leave1() throws Exception{
		Connection con = null;
		ResultSet rs = null;
		ResultSet rs2 = null;
		Statement smt = null;
		Statement smt2 = null;
		Statement smt3 = null;
		boolean change = false;
		ArrayList new_teacher_ids = null;
		int min_val,min_off,new_val,new_off;
		int new_task_id = 0,new_period_id = 0;
		//�l�̕]��
		min_val = valuateTimeTableSQLFile();
		min_off = count_offence2("timetableSQL");
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			smt2 = con.createStatement();
			smt3 = con.createStatement();
			//����p�̃e�[�u�����쐬
			String sql = "create table t_timetable as select * from timetableSQL";
			smt.executeUpdate(sql);
			//���݂̎��Ԋ��̍u�`�Ƃ��̒S���l���ƊJ�u���Ԃ��擾
			sql = "select task_id , period_id, count(*) as num from timetableSQL group by task_id order by task_id";
			rs = smt.executeQuery(sql);
			int task_id;
			int period_id;
			int num;
			while(rs.next()) {
				task_id = rs.getInt("task_id");
				period_id = rs.getInt("period_id");
				num = rs.getInt("num");
				//task_id�ł���u�`�̎擾
				Lecture l = SearchLecture(task_id);
				//task_id�̒S���l�������ȏ�̐l���̎擾
				int up = num-l.getRequired_processors_lb();
				ArrayList t_teachers2 = null;
				//System.out.println("up:"+up);
				//�����u�`�̒S���l�������ȏ�Ȃ�΂��̍u�`��S�����Ă���S�u�t��processor_id���擾
				if(up>0){
					t_teachers2 = new ArrayList();
					//period_id��task_id��S������u�t���擾
					String sql2 = "select processor_id from t_timetable where task_id='"+task_id+"' and period_id = '" + period_id +"'";
					rs2 = smt2.executeQuery(sql2);
					//System.out.println("task_id:"+task_id);
					while(rs2.next()){
						//System.out.println("�S�����Ă���u�t:"+rs2.getInt("processor_id"));
						t_teachers2.add(new Integer(rs2.getInt("processor_id")));
					}
				}
				//�S���l���̌���
				for(int j=0;j<up;j++){
					//�폜����processor_id��ۑ�����
					ArrayList t_teachers = null;
					Enumeration e = new CombEnum(t_teachers2.toArray(), j+1);
					//���̍폜����u�t�̑g�ݍ��킹�͂��邩
					while(e.hasMoreElements()) {
						t_teachers = new ArrayList();
						Object[] a = (Object[])e.nextElement();
						//System.out.println("�폜����l��:"+(j+1));
						for(int num2=0; num2<a.length; num2++){
							int processor_id = ((Integer)a[num2]).intValue();
							//�u�t�̍폜
							String sql3="delete from t_timetable where task_id ='"+task_id+"' and period_id = '"+period_id+"' and processor_id = '"+processor_id+"'";
							smt3.executeUpdate(sql3);
							t_teachers.add(new Integer(processor_id));
							//System.out.println("�폜:"+processor_id);
						}
						//�l�̕]��
						//new_off = count_offence("t_timetable");
						new_off = count_offence2("t_timetable");
						//new_val = valuate("t_timetable");
						new_val = valuateTTimeTableFile();
						if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
						{
							//�ω������������Ă���
							new_teacher_ids = t_teachers;
							new_task_id = task_id;
							new_period_id = period_id;
							min_off = new_off;
							min_val = new_val;
							change = true;
						}
						//�폜�����u�t��ǉ�
						Iterator i5 = t_teachers.iterator();
						while(i5.hasNext()){
							String sql3="insert into t_timetable(task_id,period_id , processor_id) values("+ "'" + task_id+ "','" +period_id + "','" + ((Integer)i5.next()).intValue()+"')";
							smt3.executeUpdate(sql3);
						}
					}
				}
			}
			//����p�e�[�u���̍폜
			sql = "drop table t_timetable";
			smt.executeUpdate(sql);
			//�������P���ꂽ��
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
			throw e;
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs2!=null){
				rs2.close();
				rs2 = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt2!=null){
				smt2.close();
				smt2 = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt3!=null){
				smt3.close();
				smt3 = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		return change;		
	}
	
	//�ߖT����leave:�u�`�̒S���u�t�����炷(���\�b�h�g�p)
	//�ߖTleave��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_leave2() throws Exception{
		Connection con = null;
		ResultSet rs = null;
		ResultSet rs2 = null;
		Statement smt = null;
		Statement smt2 = null;
		boolean change = false;
		ArrayList new_teacher_ids = null;
		int min_val,min_off,new_val,new_off;
		int new_task_id = 0,new_period_id = 0;
		//�l�̕]��
		//min_val = valuate("timetableSQL");
		min_val = valuateTimeTableSQLFile();
		min_off = count_offence2("timetableSQL");
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			smt2 = con.createStatement();
			//����p�̃e�[�u�����쐬
			createTTimeTable();
			//���݂̎��Ԋ��̍u�`�Ƃ��̒S���l���ƊJ�u���Ԃ��擾
			String sql = "select task_id , period_id, count(*) as num from timetableSQL group by task_id order by task_id";
			rs = smt.executeQuery(sql);
			int task_id;
			int period_id;
			int num;
			while(rs.next()) {
				task_id = rs.getInt("task_id");
				period_id = rs.getInt("period_id");
				num = rs.getInt("num");
				//task_id�ł���u�`�̎擾
				Lecture l = SearchLecture(task_id);
				//task_id�̒S���l�������ȏ�̐l���̎擾
				int up = num-l.getRequired_processors_lb();
				ArrayList t_teachers2 = null;
				//System.out.println("up:"+up);
				//�����u�`�̒S���l�������ȏ�Ȃ�΂��̍u�`��S�����Ă���S�u�t��processor_id���擾
				if(up>0){
					t_teachers2 = new ArrayList();
					//period_id��task_id��S������u�t���擾
					String sql2 = "select processor_id from t_timetable where task_id='"+task_id+"' and period_id = '" + period_id +"'";
					rs2 = smt2.executeQuery(sql2);
					//System.out.println("task_id:"+task_id);
					while(rs2.next()){
						//System.out.println("�S�����Ă���u�t:"+rs2.getInt("processor_id"));
						t_teachers2.add(new Integer(rs2.getInt("processor_id")));
					}
				}
				//�S���l���̌���
				for(int j=0;j<up;j++){
					//�폜����processor_id��ۑ�����
					ArrayList t_teachers = null;
					Enumeration e = new CombEnum(t_teachers2.toArray(), j+1);
					//���̍폜����u�t�̑g�ݍ��킹�͂��邩
					while(e.hasMoreElements()) {
						t_teachers = new ArrayList();
						Object[] a = (Object[])e.nextElement();
						//System.out.println("�폜����l��:"+(j+1));
						for(int num2=0; num2<a.length; num2++){
							int processor_id = ((Integer)a[num2]).intValue();
							//�u�t�̍폜
							deleteTTimeTableRow(task_id,period_id,processor_id);
							t_teachers.add(new Integer(processor_id));
							//System.out.println("�폜:"+processor_id);
						}
						//�l�̕]��
						//new_off = count_offence("t_timetable");
						new_off = count_offence2("t_timetable");
						//new_val = valuate("t_timetable");
						new_val = valuateTTimeTableFile();
						if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
						{
							//�ω������������Ă���
							new_teacher_ids = t_teachers;
							new_task_id = task_id;
							new_period_id = period_id;
							min_off = new_off;
							min_val = new_val;
							change = true;
						}
						//�폜�����u�t��ǉ�
						Iterator i5 = t_teachers.iterator();
						while(i5.hasNext()){
							insertTTimeTableRow(task_id,period_id,((Integer)i5.next()).intValue());
						}
					}
				}
			}
			//����p�e�[�u���̍폜
			deleteTTimeTable();
			//�������P���ꂽ��
			if(change){
				//�ύX�̕\��
				System.out.println("(task_id,processor_id,period_id)");
				Iterator i = new_teacher_ids.iterator();
				int processor_id;
				while(i.hasNext()){
					processor_id = ((Integer)i.next()).intValue();
					System.out.println("leave:("+new_task_id+","+processor_id+","+new_period_id+")");
					deleteTimeTableRow(new_task_id,new_period_id,processor_id);
				}
			}
		}catch(SQLException e) {
			System.err.println(e);
			throw e;
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs2!=null){
				rs2.close();
				rs2 = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt2!=null){
				smt2.close();
				smt2 = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		return change;		
	}
	
	//����p�̃e�[�u�����g�p���Ȃ�
	//�ߖT����leave:�u�`�̒S���u�t�����炷(���\�b�h�g�p)
	//�ߖTleave��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_leave2b() throws Exception{
		Connection con = null;
		ResultSet rs = null;
		ResultSet rs2 = null;
		Statement smt = null;
		Statement smt2 = null;
		boolean change = false;
		ArrayList new_teacher_ids = null;
		int min_val,min_off,new_val,new_off;
		int new_task_id = 0,new_period_id = 0;
		//�l�̕]��
		min_val = valuateTimeTableSQLFile();
		min_off = count_offence2("timetableSQL");
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			smt2 = con.createStatement();
			//���݂̎��Ԋ��̍u�`�Ƃ��̒S���l���ƊJ�u���Ԃ��擾
			String sql = "select task_id , period_id, count(*) as num from timetableSQL group by task_id order by task_id";
			rs = smt.executeQuery(sql);
			int task_id;
			int period_id;
			int num;
			while(rs.next()) {
				task_id = rs.getInt("task_id");
				period_id = rs.getInt("period_id");
				num = rs.getInt("num");
				//task_id�ł���u�`�̎擾
				Lecture l = SearchLecture(task_id);
				//task_id�̒S���l�������ȏ�̐l���̎擾
				int up = num-l.getRequired_processors_lb();
				ArrayList t_teachers2 = null;
				//System.out.println("up:"+up);
				//�����u�`�̒S���l�������ȏ�Ȃ�΂��̍u�`��S�����Ă���S�u�t��processor_id���擾
				if(up>0){
					t_teachers2 = new ArrayList();
					//period_id��task_id��S������u�t���擾
					String sql2 = "select processor_id from t_timetable where task_id='"+task_id+"' and period_id = '" + period_id +"'";
					rs2 = smt2.executeQuery(sql2);
					//System.out.println("task_id:"+task_id);
					while(rs2.next()){
						//System.out.println("�S�����Ă���u�t:"+rs2.getInt("processor_id"));
						t_teachers2.add(new Integer(rs2.getInt("processor_id")));
					}
				}
				//�S���l���̌���
				for(int j=0;j<up;j++){
					//�폜����processor_id��ۑ�����
					ArrayList t_teachers = null;
					Enumeration e = new CombEnum(t_teachers2.toArray(), j+1);
					//���̍폜����u�t�̑g�ݍ��킹�͂��邩
					while(e.hasMoreElements()) {
						t_teachers = new ArrayList();
						Object[] a = (Object[])e.nextElement();
						//System.out.println("�폜����l��:"+(j+1));
						for(int num2=0; num2<a.length; num2++){
							int processor_id = ((Integer)a[num2]).intValue();
							//�u�t�̍폜
							deleteTimeTableRow(task_id,period_id,processor_id);
							t_teachers.add(new Integer(processor_id));
							//System.out.println("�폜:"+processor_id);
						}
						//�l�̕]��
						new_off = count_offence2("timetableSQL");
						new_val = valuateTimeTableSQLFile();
						if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
						{
							//�ω������������Ă���
							new_teacher_ids = t_teachers;
							new_task_id = task_id;
							new_period_id = period_id;
							min_off = new_off;
							min_val = new_val;
							change = true;
						}
						//�폜�����u�t��ǉ�
						Iterator i5 = t_teachers.iterator();
						while(i5.hasNext()){
							insertTimeTableRow(task_id,period_id,((Integer)i5.next()).intValue());
						}
					}
				}
			}
			//�������P���ꂽ��
			if(change){
				//�ύX�̕\��
				System.out.println("(task_id,processor_id,period_id)");
				Iterator i = new_teacher_ids.iterator();
				int processor_id;
				while(i.hasNext()){
					processor_id = ((Integer)i.next()).intValue();
					System.out.println("leave:("+new_task_id+","+processor_id+","+new_period_id+")");
					deleteTimeTableRow(new_task_id,new_period_id,processor_id);
				}
			}
		}catch(SQLException e) {
			System.err.println(e);
			throw e;
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs2!=null){
				rs2.close();
				rs2 = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt2!=null){
				smt2.close();
				smt2 = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		return change;		
	}
	
	//�ߖT����leave:�u�`�̒S���u�t�����炷(���\�b�h�g�p)
	//�ߖTleave��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_leave3() throws Exception{
		boolean change = false;
		ArrayList new_teacher_ids = null;
		int min_val,min_off,new_val,new_off;
		int new_task_id = 0,new_period_id = 0;
		timetable.loadTimeTable();
		//�l�̕]��
		min_val = valuateTimeTableSQLFile();
		min_off = timetable.countOffence();
		//����p�̃e�[�u�����쐬
		createTTimeTable();
		//���݂̎��Ԋ��̍u�`�Ƃ��̒S���l���ƊJ�u���Ԃ��擾
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
			//task_id�ł���u�`�̎擾
			Lecture l = SearchLecture(task_id);
			//task_id�̒S���l�������ȏ�̐l���̎擾
			int up = num-l.getRequired_processors_lb();
			ArrayList t_teachers2 = null;
			//�����u�`�̒S���l�������ȏ�Ȃ�΂��̍u�`��S�����Ă���S�u�t��processor_id���擾
			if(up > 0){
				//period_id��task_id��S������u�t���擾
				t_teachers2 = timetable.getProcessors(task_id,period_id);
			}
			//�S���l���̌���
			for(int j=0;j<up;j++){
				//�폜����processor_id��ۑ�����
				ArrayList t_teachers = null;
				Enumeration e = new CombEnum(t_teachers2.toArray(), j+1);
				//���̍폜����u�t�̑g�ݍ��킹�͂��邩
				while(e.hasMoreElements()) {
					t_teachers = new ArrayList();
					Object[] a = (Object[])e.nextElement();
					//System.out.println("�폜����l��:"+(j+1));
					for(int num2=0; num2<a.length; num2++){
						int processor_id = ((Integer)a[num2]).intValue();
						//�u�t�̍폜
						deleteTTimeTableRow(task_id,period_id,processor_id);
						timetable.deleteTimeTableRow(task_id,period_id,processor_id);
						t_teachers.add(new Integer(processor_id));
					}
					//�l�̕]��
					new_off = timetable.countOffence();
					new_val = valuateTTimeTableFile();
					if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
					{
						//�ω������������Ă���
						new_teacher_ids = t_teachers;
						new_task_id = task_id;
						new_period_id = period_id;
						min_off = new_off;
						min_val = new_val;
						change = true;
					}
					//�폜�����u�t��ǉ�
					Iterator i5 = t_teachers.iterator();
					while(i5.hasNext()){
						insertTTimeTableRow(task_id,period_id,((Integer)i5.next()).intValue());
						timetable.insertTimeTableRow(task_id,period_id,((Integer)i5.next()).intValue());
					}
				}
			}
		}
		//����p�e�[�u���̍폜
		deleteTTimeTable();
		//�������P���ꂽ��
		if(change){
			//�ύX�̕\��
			System.out.println("(task_id,processor_id,period_id)");
			Iterator i = new_teacher_ids.iterator();
			int processor_id;
			while(i.hasNext()){
				processor_id = ((Integer)i.next()).intValue();
				System.out.println("leave:("+new_task_id+","+processor_id+","+new_period_id+")");
				deleteTimeTableRow(new_task_id,new_period_id,processor_id);
				//timetable.deleteTimeTableRow(new_task_id,new_period_id,processor_id);
			}
		}
		return change;		
	}
	//(�v���O�������̃e�[�u�������������Ȃ�)	
	//�ߖT����leave:�u�`�̒S���u�t�����炷(���\�b�h�g�p)
	//�ߖTleave��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_leave3b() throws Exception{
		boolean change = false;
		ArrayList new_teacher_ids = null;
		int min_val,min_off,new_val,new_off;
		int new_task_id = 0,new_period_id = 0;
		//�l�̕]��
		min_val = valuateTimeTableSQLFile();
		min_off = timetable.countOffence();
		//����p�̃e�[�u�����쐬
		createTTimeTable();
		//���݂̎��Ԋ��̍u�`�Ƃ��̒S���l���ƊJ�u���Ԃ��擾
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
			//task_id�ł���u�`�̎擾
			Lecture l = SearchLecture(task_id);
			//task_id�̒S���l�������ȏ�̐l���̎擾
			int up = num-l.getRequired_processors_lb();
			ArrayList t_teachers2 = null;
			//�����u�`�̒S���l�������ȏ�Ȃ�΂��̍u�`��S�����Ă���S�u�t��processor_id���擾
			if(up > 0){
				//period_id��task_id��S������u�t���擾
				t_teachers2 = timetable.getProcessors(task_id,period_id);
			}
			//�S���l���̌���
			for(int j=0;j<up;j++){
				//�폜����processor_id��ۑ�����
				ArrayList t_teachers = null;
				Enumeration e = new CombEnum(t_teachers2.toArray(), j+1);
				//���̍폜����u�t�̑g�ݍ��킹�͂��邩
				while(e.hasMoreElements()) {
					t_teachers = new ArrayList();
					Object[] a = (Object[])e.nextElement();
					//System.out.println("�폜����l��:"+(j+1));
					for(int num2=0; num2<a.length; num2++){
						int processor_id = ((Integer)a[num2]).intValue();
						//�u�t�̍폜
						deleteTTimeTableRow(task_id,period_id,processor_id);
						timetable.deleteTimeTableRow(task_id,period_id,processor_id);
						t_teachers.add(new Integer(processor_id));
					}
					//�l�̕]��
					new_off = timetable.countOffence();
					new_val = valuateTTimeTableFile();
					if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
					{
						//�ω������������Ă���
						new_teacher_ids = t_teachers;
						new_task_id = task_id;
						new_period_id = period_id;
						min_off = new_off;
						min_val = new_val;
						change = true;
					}
					//�폜�����u�t��ǉ�
					Iterator i5 = t_teachers.iterator();
					while(i5.hasNext()){
						insertTTimeTableRow(task_id,period_id,((Integer)i5.next()).intValue());
						timetable.insertTimeTableRow(task_id,period_id,((Integer)i5.next()).intValue());
					}
				}
			}
		}
		//����p�e�[�u���̍폜
		deleteTTimeTable();
		//�������P���ꂽ��
		if(change){
			//�ύX�̕\��
			System.out.println("(task_id,processor_id,period_id)");
			Iterator i = new_teacher_ids.iterator();
			int processor_id;
			while(i.hasNext()){
				processor_id = ((Integer)i.next()).intValue();
				System.out.println("leave:("+new_task_id+","+processor_id+","+new_period_id+")");
				deleteTimeTableRow(new_task_id,new_period_id,processor_id);
				timetable.deleteTimeTableRow(new_task_id,new_period_id,processor_id);
			}
		}
		return change;		
	}

	//(�v���O�������̃e�[�u�������������Ȃ�)	
	//�ߖT����leave:�u�`�̒S���u�t�����炷(���\�b�h�g�p)
	//�ߖTleave��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_leave3c() throws Exception{
		boolean change = false;
		ArrayList new_teacher_ids = null;
		int min_val,min_off,new_val,new_off;
		int new_task_id = 0,new_period_id = 0;
		//�l�̕]��
		min_val = valuateTimeTableSQLFile();
		min_off = timetable.countOffence();
		//���݂̎��Ԋ��̍u�`�Ƃ��̒S���l���ƊJ�u���Ԃ��擾
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
			//task_id�ł���u�`�̎擾
			Lecture l = SearchLecture(task_id);
			//task_id�̒S���l�������ȏ�̐l���̎擾
			int up = num-l.getRequired_processors_lb();
			ArrayList t_teachers2 = null;
			//�����u�`�̒S���l�������ȏ�Ȃ�΂��̍u�`��S�����Ă���S�u�t��processor_id���擾
			if(up > 0){
				//period_id��task_id��S������u�t���擾
				t_teachers2 = timetable.getProcessors(task_id,period_id);
			}
			//�S���l���̌���
			for(int j=0;j<up;j++){
				//�폜����processor_id��ۑ�����
				ArrayList t_teachers = null;
				Enumeration e = new CombEnum(t_teachers2.toArray(), j+1);
				//���̍폜����u�t�̑g�ݍ��킹�͂��邩
				while(e.hasMoreElements()) {
					t_teachers = new ArrayList();
					Object[] a = (Object[])e.nextElement();
					//System.out.println("�폜����l��:"+(j+1));
					for(int num2=0; num2<a.length; num2++){
						int processor_id = ((Integer)a[num2]).intValue();
						//�u�t�̍폜
						deleteTimeTableRow(task_id,period_id,processor_id);
						timetable.deleteTimeTableRow(task_id,period_id,processor_id);
						t_teachers.add(new Integer(processor_id));
					}
					//�l�̕]��
					new_off = timetable.countOffence();
					new_val = valuateTimeTableSQLFile();
					if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
					{
						//�ω������������Ă���
						new_teacher_ids = t_teachers;
						new_task_id = task_id;
						new_period_id = period_id;
						min_off = new_off;
						min_val = new_val;
						change = true;
					}
					//�폜�����u�t��ǉ�
					Iterator i5 = t_teachers.iterator();
					while(i5.hasNext()){
						insertTimeTableRow(task_id,period_id,((Integer)i5.next()).intValue());
						timetable.insertTimeTableRow(task_id,period_id,((Integer)i5.next()).intValue());
					}
				}
			}
		}
		//�������P���ꂽ��
		if(change){
			//�ύX�̕\��
			System.out.println("(task_id,processor_id,period_id)");
			Iterator i = new_teacher_ids.iterator();
			int processor_id;
			while(i.hasNext()){
				processor_id = ((Integer)i.next()).intValue();
				System.out.println("leave:("+new_task_id+","+processor_id+","+new_period_id+")");
				deleteTimeTableRow(new_task_id,new_period_id,processor_id);
				timetable.deleteTimeTableRow(new_task_id,new_period_id,processor_id);
			}
		}
		return change;		
	}
	
	//���̕]���l��v�]�ʂɐ����č��v��Ԃ�
	public int valuate(String table) throws Exception{
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		int penalties = 0;
		int timetable = 0;
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			//���Ԋ��̌��̌��𒲂ׂ�
			String sql = "select count(*) as num from "+ table ;
			rs = smt.executeQuery(sql);
			if(rs.next()){
				timetable = rs.getInt("num");
			}
			//�e�u�`�̒S���u�t�̐l���̏�������邩�ǂ����H
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
			//�e�u�`�̒S���u�t�̐l���̉���������邩�ǂ����H
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
			//�e�u�t�ɂ��āA�u�`��S������񐔂���������邩�ǂ����H
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
			
			//�e�u�t�ɂ��āA�u�`��S������񐔂�����������邩�ǂ����H
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
			
			//���΍u�t�S���̍u�`�S���񐔂̍��v�Ə���l�Ƃ̍�(����l:40)
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
			
			//�e�u�t�ɂ��āA�u�`��S�������������������邩�H
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
			
			//�p��ōs���e�u�`�ɂ��āA�S���u�t�̂����p��𗬒��ɘb���Ȃ����̐l��
			//���{��ōs���e�u�`�ɂ��āA�S���u�t�̂������{��𗬒��ɘb���Ȃ����̐l��
			sql = "select count(*) as num from "
				+table+" a , processor_qualification b , task_properties c where c.qualification_id = b.qualification_id and a.processor_id = b.processor_id and a.task_id = c.task_id";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("num");
				num = timetable - num;
				penalties = penalties + (num * 1000);
			}
			
			//�e�u�t�A�e�j���ɂ��āA�S������u�`�̊Ԃɂ��邠��󂫎��Ԃ̐�
			sql = "create table t as select a.task_id,a.period_id,a.processor_id,c.employment,b.day_id from "
				+table+" a,period_properties b ,processor_properties c where a.period_id = b.period_id and a.processor_id = c.processor_id";
			smt.executeUpdate(sql);			
			//�󂫎���1�̎�
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
			//�󂫎���2�̎�
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
			//�󂫎���3�̎�
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
			
			//�e�u�t�ɂ��āA�u�`��S������u�`���Ԃɂ�����S���s���x
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
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		return penalties;
	}
	//	���̕]���l��v�]�ʂɐ����č��v��Ԃ�(�ڍ�)
	public int valuateDetail(String table) throws Exception{
		Connection con = null;
		ResultSet rs = null;
		ResultSet rs2 = null;
		Statement smt = null;
		Statement smt2 = null;
		int penalties = 0;
		int timetable = 0;
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			smt2 = con.createStatement();
			//���Ԋ��̌��̌��𒲂ׂ�
			String sql = "select count(*) as num from "+ table ;
			rs = smt.executeQuery(sql);
			if(rs.next()){
				timetable = rs.getInt("num");
			}
			//�e�u�`�̒S���u�t�̐l���̏�������邩�ǂ����H
			sql = "create temporary table t select a.task_id,count(*),b.required_processors_ub,count(*)-CAST(b.required_processors_ub AS SIGNED) as num from "
				+table+" a, task_properties b where a.task_id = b.task_id group by a.task_id,b.required_processors_lb having count(*)-CAST(b.required_processors_ub AS SIGNED) > 0";
			smt.executeUpdate(sql);
			sql = "select sum(num) as penalties from t";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("penalties");
				penalties = num * 10000;
				System.out.println("�e�u�`�̒S���u�t�̐l���̏�������邩�ǂ����H:"+num);
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

			//�e�u�`�̒S���u�t�̐l���̉���������邩�ǂ����H
			sql = "create temporary table t select a.task_id,count(*),b.required_processors_lb,CAST(b.required_processors_lb AS SIGNED)-count(*) as num from "
				+table+" a, task_properties b where a.task_id = b.task_id group by a.task_id,b.required_processors_ub having CAST(b.required_processors_lb AS SIGNED)-count(*) > 0";
			smt.executeUpdate(sql);
			sql = "select sum(num) as penalties from t";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("penalties");
				penalties = penalties + (num * 10000);
				System.out.println("�e�u�`�̒S���u�t�̐l���̉���������邩�ǂ����H:"+num);
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

			//�e�u�t�ɂ��āA�u�`��S������񐔂���������邩�ǂ����H
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
						System.out.println("�e�u�t�ɂ��āA�u�`��S������񐔂���������邩�ǂ����H:"+num2);
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

			//�e�u�t�ɂ��āA�u�`��S������񐔂�����������邩�ǂ����H
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
						System.out.println("�e�u�t�ɂ��āA�u�`��S������񐔂�����������邩�ǂ����H:"+num2);
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

			//���΍u�t�S���̍u�`�S���񐔂̍��v�Ə���l�Ƃ̍�(����l:40)
			sql = "select count(*) as num from "
				+table+ " a,processor_properties b where b.employment = 2 and a.processor_id = b.processor_id";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("num");
				int temp = num - 40;
				if(temp > 0){
					penalties = penalties + ( temp * 1000);
					System.out.println("���΍u�t�S���̍u�`�S���񐔂̍��v�Ə���l�Ƃ̍�(����l:40):"+temp);
				}
			}

			//�e�u�t�ɂ��āA�u�`��S�������������������邩�H
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
						System.out.println("�e�u�t�ɂ��āA�u�`��S�������������������邩�H:"+num2);
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

			//�p��ōs���e�u�`�ɂ��āA�S���u�t�̂����p��𗬒��ɘb���Ȃ����̐l��
			//���{��ōs���e�u�`�ɂ��āA�S���u�t�̂������{��𗬒��ɘb���Ȃ����̐l��
			sql = "select count(*) as num from "
				+table+" a , processor_qualification b , task_properties c where c.qualification_id = b.qualification_id and a.processor_id = b.processor_id and a.task_id = c.task_id";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("num");
				num = timetable - num;
				penalties = penalties + (num * 1000);
				System.out.println("�S���u�t�̂��������ɘb���Ȃ����̐l��:"+num);
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

			//�e�u�t�A�e�j���ɂ��āA�S������u�`�̊Ԃɂ��邠��󂫎��Ԃ̐�
			sql = "create table t as select a.task_id,a.period_id,a.processor_id,c.employment,b.day_id from "
				+table+" a,period_properties b ,processor_properties c where a.period_id = b.period_id and a.processor_id = c.processor_id";
			smt.executeUpdate(sql);
			//�󂫎���1�̎�
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
						System.out.println("�󂫎���1�̎�:"+num2);
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
			//�󂫎���2�̎�
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
						System.out.println("�󂫎���2�̎�:"+num2);
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
			//�󂫎���3�̎�
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
						System.out.println("�󂫎���3�̎�:"+num2);
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

			//�e�u�t�ɂ��āA�u�`��S������u�`���Ԃɂ�����S���s���x
			sql = "create temporary table t select a.processor_id,sum(b.preferred_level_proc) as num from "
				+table+" a,processor_schedules b where a.period_id = b.period_id and a.processor_id = b.processor_id group by a.processor_id";
			smt.executeUpdate(sql);
			sql = "select sum(num) as penalties from t";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("penalties");
				penalties = penalties + num;
				System.out.println("�e�u�t�ɂ��āA�u�`��S������u�`���Ԃɂ�����S���s���x:"+num);
			}
			sql = "drop table t";
			smt.executeUpdate(sql);
		}catch(SQLException e) {
			System.out.println(e);
			throw e;
		}catch(Exception e){
			e.printStackTrace();
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			if(rs2!=null){
				rs2.close();
				rs2 = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			if(smt2!=null){
				smt2.close();
				smt2 = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		return penalties;
	}
	
	//���̕]���l��v�]�ʂɐ����č��v��Ԃ�(t_timetable�ɑ΂��čs��)
	public int valuateTTimeTableFile() throws Exception{
		if(SWflag){
			SWflag = false;
			vsw.start();
		}
		else vsw.resume();
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		int penalties = 0;
		ValuateSQL t = null;
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			Iterator i = t_v_sqls.iterator();
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
		}catch(Exception e){
			e.printStackTrace();
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		vsw.suspend();
		return penalties;
	}
	//���̕]���l��v�]�ʂɐ����č��v��Ԃ�(t_timetable�ɑ΂��čs��)
	//�o�b�`����(�t�@�C�� t_valuate4.txt)
	public int valuateTTimeTableFile3() throws Exception{
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		int penalties = 0;
		if(SWflag){
			SWflag = false;
			vsw.start();
		}
		else vsw.resume();
		ValuateSQL t = null;
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			con.setAutoCommit(false);
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			Iterator i = t_v_sqls.iterator();
			while(i.hasNext()){
				t = (ValuateSQL)i.next();
				switch(t.getType()){
				case 1:
					smt.addBatch(t.getSql());
					break;
				case 2:
					smt.executeBatch();
					rs = smt.executeQuery(t.getSql());
					if(rs.next()){
						penalties = rs.getInt("penalties");
					}
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
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		vsw.suspend();
		return penalties;
	}
	
	//���̕]���l��v�]�ʂɐ����č��v��Ԃ�(timetableSQL�ɑ΂��čs��)
	public int valuateTimeTableSQLFile() throws Exception{
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		int penalties = 0;
		ValuateSQL t = null;
		if(SWflag){
			SWflag = false;
			vsw.start();
		}
		else vsw.resume();
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
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
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		vsw.suspend();
		return penalties;
	}
	
	//���̕]���l��v�]�ʂɐ����č��v��Ԃ�(timetableSQL�ɑ΂��čs��)
	//�o�b�`����(�t�@�C�� t_valuate4.txt)	
	public int valuateTimeTableSQLFile3() throws Exception{
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		int penalties = 0;
		ValuateSQL t = null;
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			con.setAutoCommit(false);
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			Iterator i = v_sqls.iterator();
			while(i.hasNext()){
				t = (ValuateSQL)i.next();
				switch(t.getType()){
				case 1:
					smt.addBatch(t.getSql());
					break;
				case 2:
					smt.executeBatch();
					rs = smt.executeQuery(t.getSql());
					if(rs.next()){
						penalties = rs.getInt("penalties");
					}
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
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		return penalties;
	}
	
	//���̐���ᔽ����ޕʂɐ����č��v��Ԃ�(�ꎞ�e�[�u���쐬)
	public int count_offence(String table) throws Exception{
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		int timetable  = 0;
		int penalties = 0;
		int t_task_num = 0;
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			//���Ԋ��̌��̌��𒲂ׂ�
			String sql = "select count(*) as num from "+table;
			rs = smt.executeQuery(sql);
			if(rs.next()){
				timetable = rs.getInt("num");
			}
			//�e�u�t�͒S���\���Ԃɍu�`��S��(�S���\�ȍu�`���ǂ����͒��ׂĂ��Ȃ�)
			sql = "create temporary table t select a.processor_id,a.task_id,a.period_id from "
				+table+" a,processor_schedules b where a.processor_id = b.processor_id and a.period_id = b.period_id";
			smt.executeUpdate(sql);
			sql = "select count(*) as num from t";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("num");
				penalties = timetable - num;
				//System.out.println("�e�u�t�͒S���\���Ԃɍu�`��S��"+penalties);
				
			}
			sql = "drop table t";
			smt.executeUpdate(sql);
			//�e�u�t�́A�������Ԃɕ����̍u�`��S�����Ȃ�
			sql = "create temporary table t select processor_id,period_id, count(*)-1 as num from "
				+table+" group by processor_id,period_id having count(*)-1 > 0";
			smt.executeUpdate(sql);
			sql = "select sum(num) as num2 from t";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("num2");
				penalties += num;
				//System.out.println("�e�u�t�́A�������Ԃɕ����̍u�`��S�����Ȃ�"+num);
			}
			sql = "drop table t";
			smt.executeUpdate(sql);
			//�e�u�`�́A�e�u�`�ŗL�̊J�u�\���Ԃ̂�����ɊJ�u�����
			//1.�e�u�`�͊J�u�\���ԂɊJ�u�����
			sql = "create temporary table t select a.processor_id,a.task_id,a.period_id from "
				+table+" a,task_opportunities b where a.task_id = b.task_id and a.period_id = b.period_id;";
			smt.executeUpdate(sql);
			sql = "select count(*) as num from t";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("num");
				penalties = penalties + (timetable - num);
				//System.out.println("�e�u�`�͊J�u�\���ԂɊJ�u�����"+(timetable - num));
			}
			sql = "drop table t";
			smt.executeUpdate(sql);
			//2.�e�u�`�͊J�u�\�ł���ΕK���J�u�����
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
				//System.out.println("�e�u�`�͊J�u�\�ł���ΕK���J�u�����"+(num - t_task_num));
			}
			sql = "drop table t";
			smt.executeUpdate(sql);
			//�e�u�`�͂���������J�u�����
			sql = "create temporary table t select count(*) as num from "
				+table+" group by task_id,period_id";
			smt.executeUpdate(sql);
			sql = "select count(*) as num2 from t";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("num2");
				penalties = penalties + (num - t_task_num);
				//System.out.println("�e�u�`�͂���������J�u�����"+(num - t_task_num));
			}
			sql = "drop table t";
			smt.executeUpdate(sql);
		}catch(SQLException e) {
			System.err.println(e);
			throw e;
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		return penalties;
	}
	
	//���̐���ᔽ����ޕʂɐ����č��v��Ԃ�(�]��SQL�̕ύXver1)
	public int count_offence2(String table) throws Exception{
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		int timetable  = 0;
		int penalties = 0;
		int t_task_num = 0;
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			//���Ԋ��̌��̌��𒲂ׂ�
			String sql = "select count(*) as num from "+table;
			rs = smt.executeQuery(sql);
			if(rs.next()){
				timetable = rs.getInt("num");
			}
			//�e�u�t�͒S���\���Ԃɍu�`��S��(�S���\�ȍu�`���ǂ����͒��ׂĂ��Ȃ�)
			sql = "select count(*) as num from (select a.processor_id,a.task_id,a.period_id from "
				+table+" a,processor_schedules b where a.processor_id = b.processor_id and a.period_id = b.period_id) as t";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("num");
				penalties = timetable - num;
				//System.out.println("�e�u�t�͒S���\���Ԃɍu�`��S��"+penalties);
				
			}
			//�e�u�t�́A�������Ԃɕ����̍u�`��S�����Ȃ�
			sql = "select sum(num) as num2 from (select processor_id,period_id, count(*)-1 as num from "
				+table+" group by processor_id,period_id having count(*)-1 > 0) as t";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("num2");
				penalties += num;
				//System.out.println("�e�u�t�́A�������Ԃɕ����̍u�`��S�����Ȃ�"+num);
			}
			//�e�u�`�́A�e�u�`�ŗL�̊J�u�\���Ԃ̂�����ɊJ�u�����
			//1.�e�u�`�͊J�u�\���ԂɊJ�u�����
			sql = "select count(*) as num from (select a.processor_id,a.task_id,a.period_id from "
				+table+" a,task_opportunities b where a.task_id = b.task_id and a.period_id = b.period_id) as t";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("num");
				penalties = penalties + (timetable - num);
				//System.out.println("�e�u�`�͊J�u�\���ԂɊJ�u�����"+(timetable - num));
			}
			//2.�e�u�`�͊J�u�\�ł���ΕK���J�u�����
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
				//System.out.println("�e�u�`�͊J�u�\�ł���ΕK���J�u�����"+(num - t_task_num));
			}
			//�e�u�`�͂���������J�u�����
			sql = "select count(*) as num2 from (select count(*) as num from "
				+table+" group by task_id,period_id) as t";
			rs = smt.executeQuery(sql);
			if(rs.next()){
				int num = rs.getInt("num2");
				penalties = penalties + (num - t_task_num);
				//System.out.println("�e�u�`�͂���������J�u�����"+(num - t_task_num));
			}
		}catch(SQLException e) {
			System.err.println(e);
			throw e;
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		return penalties;
	}
	
	//table��timetableSQL�ɃR�s�[
	public void LoadTimetable(String table) throws Exception{
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			//��DB��ɂ��łɑ��݂��Ă���ꍇ���̃e�[�u�����폜����
			String sql = "drop table if exists timetableSQL";
			smt.executeUpdate(sql);
			sql = "create table timetableSQL as select * from "+table;
			smt.executeUpdate(sql);
		}catch(SQLException e) {
			System.out.println(e);				
			throw e;
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}
	//DB��Ɍ��݂�timetableSQL��table���ŕۑ�����
	public void SaveTimetable(String table) throws Exception{
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			//��DB��ɂ��łɑ��݂��Ă���ꍇ���̃e�[�u�����폜����
			String sql = "drop table if exists " + table;
			smt.executeUpdate(sql);
			sql = "create table " + table + " as select * from timetableSQL";
			smt.executeUpdate(sql);
		}catch(SQLException e) {
			System.out.println(e);				
			throw e;
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}
	//task_id�ł���u�`���擾����
	private Lecture SearchLecture(int task_id){
		Iterator i = lectures.iterator();
		while(i.hasNext()){
			Lecture l = (Lecture)i.next();
			//�u�`l��task_id�ł���
			if(l.getTask_id()==task_id){
				return l;
			}
		}
		return null;
	}
	
	//processor_id�ł���u�t���擾����
	private Teacher SearchTeacher(int processor_id){
		Iterator i = teachers.iterator();
		while(i.hasNext()){
			Teacher t = (Teacher)i.next();
			//�u�tt��processor_id�ł���
			if(t.getProcessor_id()==processor_id){
				return t;
			}
		}
		return null;
	}
	
	//�u�tprocessor_id��period_id�ɕ����̍u�`��S�����Ȃ����ǂ���(�S�����Ȃ�:true �S������:false)
	//(�e�[�u��t_timetable)
	private boolean isNotTasks(int processor_id, int period_id) throws Exception{
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		boolean flag = false;
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			//�u�t�͓������Ԃɕ����̍u�`�����Ȃ����𒲂ׂ�
			String sql = "select * from t_timetable where processor_id='"+processor_id+"' and period_id = '" + period_id +"'";
			rs = smt.executeQuery(sql);
			if(rs.next()==false){
				flag = true;
			}
		}catch(SQLException e) {
			System.out.println(e);				
			throw e;
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		return flag;
	}
	
	//�u�tprocessor_id��period_id�ɕ����̍u�`��S�����Ȃ����ǂ���(�S�����Ȃ�:true �S������:false)
	//(�e�[�u��t_timetable)
	private boolean isNotTasksb(int processor_id, int period_id) throws Exception{
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		boolean flag = false;
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			//�u�t�͓������Ԃɕ����̍u�`�����Ȃ����𒲂ׂ�
			String sql = "select * from timetableSQL where processor_id='"+processor_id+"' and period_id = '" + period_id +"'";
			rs = smt.executeQuery(sql);
			if(rs.next()==false){
				flag = true;
			}
		}catch(SQLException e) {
			System.out.println(e);				
			throw e;
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		return flag;
	}
	
	//(task_id,period_id,processor_id)�����Ԋ�t_timetable�ɒǉ�
	private void insertTTimeTableRow(int task_id,int period_id,int processor_id) throws Exception{
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			//�ǉ�
			String sql="insert into t_timetable(task_id,period_id,processor_id) values('" + task_id + "','" +period_id + "','" + processor_id+"')";
			smt.executeUpdate(sql);
		}catch(SQLException e) {
			System.out.println(e);				
			throw e;
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}
	
	//(task_id,period_id,processor_id)�����Ԋ�timetableSQL�ɒǉ�
	private void insertTimeTableRow(int task_id,int period_id,int processor_id) throws Exception{
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			//�ǉ�
			String sql="insert into timetableSQL(task_id,period_id,processor_id) values('" + task_id + "','" +period_id + "','" + processor_id+"')";
			smt.executeUpdate(sql);
		}catch(SQLException e) {
			System.out.println(e);				
			throw e;
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}
	//(task_id,period_id,processor_id)�����Ԋ�t_timetable����폜
	private void deleteTTimeTableRow(int task_id,int period_id, int processor_id) throws Exception{
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			//�폜
			String sql = "delete from t_timetable where processor_id ='" + processor_id+"' and period_id = '" + period_id+"' and task_id = '"+task_id +"'";
			smt.executeUpdate(sql);
		}catch(SQLException e) {
			System.out.println(e);				
			throw e;
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}
	
	//(task_id,period_id,processor_id)�����Ԋ�timetableSQL����폜
	private void deleteTimeTableRow(int task_id,int period_id, int processor_id) throws Exception{
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			//�폜
			String sql = "delete from timetableSQL where processor_id ='" + processor_id+"' and period_id = '" + period_id+"' and task_id = '"+task_id +"'";
			smt.executeUpdate(sql);
		}catch(SQLException e) {
			System.out.println(e);				
			throw e;
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}
	
	//�u�`task_id�����Ԋ�t_timetable����폜
	private void deleteTaskTimeTable(int task_id, int period_id) throws Exception{
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			//�폜
			String sql = "delete from timetableSQL where task_id = '"+task_id +"' and period_id ='"+ period_id+"'";
			smt.executeUpdate(sql);
		}catch(SQLException e) {
			System.out.println(e);				
			throw e;
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}
	
	//�u�`task_id�����Ԋ�t_timetable����폜
	private void deleteTaskTTimeTable(int task_id, int period_id) throws Exception{
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			//�폜
			String sql = "delete from t_timetable where task_id = '"+task_id +"' and period_id ='"+ period_id+"'";
			smt.executeUpdate(sql);
		}catch(SQLException e) {
			System.out.println(e);				
			throw e;
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}
	
	//�u�`task_id�����Ԋ�t_timetable����폜
	private void deleteTaskTTimeTable(int task_id) throws Exception{
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			//�폜
			String sql = "delete from t_timetable where task_id = '"+task_id +"'";
			smt.executeUpdate(sql);
		}catch(SQLException e) {
			System.out.println(e);				
			throw e;
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}
	
	//�u�`task_id�����Ԋ�timetableSQL����폜
	private void deleteTaskTimeTable(int task_id) throws Exception{
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			//�폜
			String sql = "delete from timetableSQL where task_id = '"+task_id +"'";
			smt.executeUpdate(sql);
		}catch(SQLException e) {
			System.out.println(e);				
			throw e;
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}
	//�ꎞ�e�[�u��t_timetable�̍쐬
	private void createTTimeTable()  throws Exception{
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			String sql = "create table t_timetable as select * from timetableSQL";
			smt.executeUpdate(sql);
		}catch(SQLException e) {
			System.out.println(e);				
			throw e;
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}
	
	//�ꎞ�e�[�u��t_timetable�̍폜
	private void deleteTTimeTable() throws Exception{
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			String sql = "drop table t_timetable";
			smt.executeUpdate(sql);
		}catch(SQLException e) {
			System.out.println(e);				
			throw e;
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}
	
	//
	private void updateProcessorTimeTable(int new_processor_id,int old_processor_id, int old_task_id, int old_period_id) throws Exception{
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			String sql = "update timetableSQL set processor_id ='"+new_processor_id+"' where processor_id = '"+old_processor_id+"' and task_id = '" + old_task_id + "' and period_id = '" + old_period_id + "'";
			smt.executeUpdate(sql);
		}catch(SQLException e) {
			System.out.println(e);				
			throw e;
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}
}