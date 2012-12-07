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
 * �쐬��: 2006/12/11
 *
 * TODO ���̐������ꂽ�t�@�C���̃e���v���[�g��ύX����ɂ͎��փW�����v:
 * �E�B���h�E - �ݒ� - Java - �R�[�h�E�X�^�C�� - �R�[�h�E�e���v���[�g
 */

/**
 * @author masayoshi
 *
 * TODO ���Ԋ��쐬
 *
 */
public class TimeTableSchedulingTaskSeries {
	//�S�u�t(���X�g�̒��gObject:Teacher)
	private ArrayList teachers;
	//�S�u�`(���X�g�̒��gObject:Lecture)
	private ArrayList lectures;
	//�S���Ԃ̏��(���X�g�̒��gObject:DayPeriod)
	private OrderDayPeriods day_periods;
	//SQL�ɂ��v�]�]����
	private ArrayList v_sqls;
	//�f�[�^�x�[�X�̃R�l�N�V�����v�[��
	private DBConnectionPool cmanager;
	//�u�`���Ԋ���
	private TimeTable timetable;
	//�]���̕��@ 1:�t�@�C�� 2:DB
	private int valuateType;
	
	//�����p�X�g�b�v�E�H�b�`
	private StopWatch vsw;
	private boolean SWflag = true;
	private boolean debug = false;
	
	//�R���X�g���N�^(�v���W�F�N�g�����݂��Ȃ�)
	public TimeTableSchedulingTaskSeries(String host,String db,String user, String pass, String file, int readType) throws Exception{
		cmanager = new DBConnectionPool("jdbc:mysql://"+host+"/"+db+"?useUnicode=true&characterEncoding=sjis",user,pass);
		init(file,readType);
		timetable = new TimeTable(cmanager,teachers,lectures);
		if(debug)vsw = new StopWatch();
	}
	//�R���X�g���N�^
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
			if(debug) System.out.println("�u�t�̏������I��");
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
			if(debug) System.out.println("�u�`�̏������I��");
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
	
	//������(�v���W�F�N�g�Ǘ����Ă���f�[�^�x�[�X�g�p)
	private void initAddProject(String file,int readType,int project_id) throws Exception{
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
			sql = "drop table if exists t_task_relation";
			smt.executeUpdate(sql);
			sql = "drop table if exists t_processor_relation";
			smt.executeUpdate(sql);
			//�u�t�̏�����
			//���������͂���Ă���S�u�t���擾
			sql = "select a.processor_id from processors a, processor_properties b where a.processor_id = b.processor_id and a.project_id ="+project_id+" order by a.processor_id";
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
			if(debug) System.out.println("�u�t�̏������I��");
			//�u�`�̏�����
			//���������͂���Ă���S�u�`�̎擾
			sql = "select * from tasks a, task_properties b where a.task_id = b.task_id and a.project_id = "+project_id+" order by a.task_id";
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
			if(debug) System.out.println("�u�`�̏������I��");
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
	//������(�v���W�F�N�g�Ǘ����Ă���f�[�^�x�[�X�g�p)
	//�A������u�`�ɑΉ�
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
			sql = "drop table if exists t_task_relation";
			smt.executeUpdate(sql);
			sql = "drop table if exists t_processor_relation";
			smt.executeUpdate(sql);
			//�u�t�̏��̏�����
			//���������͂���Ă���u�t��processor_id���擾
			sql = "select a.processor_id from processors a, processor_properties b where a.processor_id = b.processor_id and a.project_id ="+project_id+" order by a.processor_id";
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
			if(debug) System.out.println("�u�t�̏������I��");
			//�u�`�̏��̏�����
			//���������͂���Ă���u�`�̏��̎擾
			sql = "select * from tasks a, task_properties b where a.task_id = b.task_id and a.project_id = "+project_id+" order by a.task_id";
			rs = smt.executeQuery(sql);
			while(rs.next()){
				Lecture l = new Lecture();
				ArrayList pds = new ArrayList();
				l.setTask_id(rs.getInt("task_id"));
				l.setRequired_processors_lb(rs.getInt("required_processors_lb"));
				l.setRequired_processors_ub(rs.getInt("required_processors_ub"));
				l.setQualification_id(rs.getInt("qualification_id"));
				//�A������u�`
				l.setTask_series_num(rs.getInt("task_series_num"));
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
			if(debug){
				System.out.println("�u�`�̏������I��");
				/*
				Iterator i = lectures.iterator();
				while(i.hasNext()){
					Lecture l = (Lecture)i.next();
					System.out.println(l.toString());
				}
				*/
				
			}
			//���Ԃ̏�����
			sql = "select pp.day_id , pp.period_id from period_properties pp, days d, hours h where pp.day_id = d.day_id and pp.hour_id = h.hour_id and pp.project_id = "+project_id+" order by d.number,d.day_id,h.number,h.hour_id";
			rs = smt.executeQuery(sql);
			while(rs.next()){
				DayPeriod dp = new DayPeriod(rs.getInt("day_id"),rs.getInt("period_id"));
				list.add(dp);
			}
			day_periods = new OrderDayPeriods(list);
			if(debug) System.out.println("���Ԃ̏������I��");
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
	
	//�t�@�C������]��SQL���擾����(�ǂݍ��ނr�p�k�ɔԍ�)
	//�����߂��Ȃ�
	private void readerFileValuateNumSQL(String file) throws Exception{
		//�t�@�C������]��SQL���擾����
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
			System.out.println("�t�@�C����������܂���");
			throw e;
		}
	}
	
	//	�t�@�C������]��SQL���擾����
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
			System.out.println("�t�@�C����������܂���");
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

	//�f�[�^�x�[�X����]��SQL���擾����
	private void readerDBValuateSQL(String table) throws Exception{
		v_sqls = new ArrayList();
		ArrayList create_list = new ArrayList();
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
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

	//���̕]���l���t�@�C���Ɏc��
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
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
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
				bw.write("�]��SQL");
				bw.newLine();
				v_sqls_p = (ValuateSQLsPenalty)i3.next();
				Iterator i4 = v_sqls_p.getV_sqls().getV_sqls().iterator();
				while(i4.hasNext()){
					t2 = (ValuateSQL)i4.next();
					bw.write(t2.getSql());
					bw.newLine();
				}
				bw.write("�]���l:"+v_sqls_p.getPenalty());
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
		if(debug) vsw.suspend();
	}

	//�R�l�N�V�����̉��
	public void close() throws Exception{
		cmanager.release();
	}

	/**
	 * @return timetable ��߂��܂��B
	 */
	public TimeTable getTimetable() {
		return timetable;
	}

	//�������쐬(�ύX�\��)
	public void initsol() throws Exception{
		ResultSet rs = null;
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
			String sql = "create table t_task_relation as select a.task_id, b.period_id, a.qualification_id, a.required_processors_lb, a.required_processors_ub from task_properties a, task_opportunities b where a.task_id = b.task_id order by task_id, period_id";
			smt.executeUpdate(sql);
			sql = "create table t_processor_relation as select a.processor_id, c.period_id, b.qualification_id from processor_properties a, processor_qualification b, processor_schedules c where a.processor_id = b.processor_id and a.processor_id = c.processor_id order by processor_id, period_id";
			smt.executeUpdate(sql);
			//���Ԋ��̏�����
			sql = "delete from timetablesql";
			smt.executeUpdate(sql);
			while(true){
				//�u�`�Ƃ��̍u�`�J�u�\���Ԃ������_���ɑI��
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
					//���̑���ɂ��e�u�`�́A�e�u�`�ŗL�̊J�u�\���Ԃ̂�����ɊJ�u�����B
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
			System.err.println("���R�F" + e.toString());
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
		timetable.loadTimeTable();
	}

	//�������쐬(�ύX�\��)
	public void initsolAddProject(int project_id) throws Exception{
		ResultSet rs = null;
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
			String sql = "create table t_task_relation as select a.task_id, b.period_id, a.qualification_id, a.required_processors_lb, a.required_processors_ub from task_properties a, task_opportunities b where a.task_id = b.task_id and a.project_id ="+project_id+" order by task_id, period_id";
			smt.executeUpdate(sql);
			sql = "create table t_processor_relation as select a.processor_id, c.period_id, b.qualification_id from processor_properties a, processor_qualification b, processor_schedules c where a.processor_id = b.processor_id and a.processor_id = c.processor_id  and a.project_id ="+project_id+" order by processor_id, period_id";
			smt.executeUpdate(sql);
			//���Ԋ��̏�����
			sql = "delete from timetablesql";
			smt.executeUpdate(sql);
			while(true){
				//�u�`�Ƃ��̍u�`�J�u�\���Ԃ������_���ɑI��
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
					//���̑���ɂ��e�u�`�́A�e�u�`�ŗL�̊J�u�\���Ԃ̂�����ɊJ�u�����B
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
			System.err.println("���R�F" + e.toString());
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
		timetable.loadTimeTable();
	}

	//�������쐬(�ύX�\��)
	//�A������u�`�Ή�
	public void initsolTaskSeriesAddProject(int project_id) throws Exception{
		timetable.clearTimeTable();
		boolean debug2=false;
		Connection con = null;
		Statement smt = null;
		Statement smt2 = null;
		ResultSet rs = null;
		//�V�[�h��^���āARandom�N���X�̃C���X�^���X�𐶐�����B
		Random rand = new Random(Calendar.getInstance().getTimeInMillis());
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			smt2 = con.createStatement();
			String sql = "create table t_task_relation as select a.task_id, b.period_id, a.qualification_id, a.required_processors_lb,a.required_processors_ub,a.task_series_num from task_properties a, task_opportunities b where a.task_id = b.task_id and a.project_id ="+project_id+" order by task_id, period_id";
			smt.executeUpdate(sql);
			sql = "create table t_processor_relation as select a.processor_id, c.period_id, b.qualification_id from processor_properties a, processor_qualification b, processor_schedules c where a.processor_id = b.processor_id and a.processor_id = c.processor_id  and a.project_id ="+project_id+" order by processor_id, period_id";
			smt.executeUpdate(sql);
			//���Ԋ��̏�����
			sql = "delete from timetablesql";
			smt.executeUpdate(sql);
			while(true){
				//�u�`�Ƃ��̍u�`�J�u�\���Ԃ������_���ɑI��
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
							//���̑���ɂ��e�u�`�́A�e�u�`�ŗL�̊J�u�\���Ԃ̂�����ɊJ�u�����B
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
						//���̑���ɂ��e�u�`�́A�e�u�`�ŗL�̊J�u�\���Ԃ̂�����ɊJ�u�����B
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
			System.err.println("���R�F" + e.toString());
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
		//timetable.initTimetable();
	}

	//�������쐬
	public void initsolNotRandom(int seed) throws Exception{
		ResultSet rs = null;
		Connection con = null;
		Statement smt = null;
		Statement smt2 = null;
		//Random�N���X�̃C���X�^���X�𐶐�����B
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
			sql = "delete from timetablesql";
			smt.executeUpdate(sql);
			while(true){
				//�u�`�Ƃ��̍u�`�J�u�\���Ԃ������_���ɑI��
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
					//���̑���ɂ��e�u�`�́A�e�u�`�ŗL�̊J�u�\���Ԃ̂�����ɊJ�u�����B
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
			System.err.println("���R�F" + e.toString());
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

//	�������쐬(�ύX�\��)
	//�A������u�`�Ή�
	public void initsolNotRandomTaskSeriesAddProject(int seed,int project_id) throws Exception{
		timetable.clearTimeTable();
		boolean debug2=false;
		ResultSet rs = null;
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
			String sql = "create table t_task_relation as select a.task_id, b.period_id, a.qualification_id, a.required_processors_lb,a.required_processors_ub,a.task_series_num from task_properties a, task_opportunities b where a.task_id = b.task_id and a.project_id ="+project_id+" order by task_id, period_id";
			smt.executeUpdate(sql);
			sql = "create table t_processor_relation as select a.processor_id, c.period_id, b.qualification_id from processor_properties a, processor_qualification b, processor_schedules c where a.processor_id = b.processor_id and a.processor_id = c.processor_id  and a.project_id ="+project_id+" order by processor_id, period_id";
			smt.executeUpdate(sql);
			//���Ԋ��̏�����
			sql = "delete from timetablesql";
			smt.executeUpdate(sql);
			while(true){
				//�u�`�Ƃ��̍u�`�J�u�\���Ԃ������_���ɑI��
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
							//���̑���ɂ��e�u�`�́A�e�u�`�ŗL�̊J�u�\���Ԃ̂�����ɊJ�u�����B
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
						//���̑���ɂ��e�u�`�́A�e�u�`�ŗL�̊J�u�\���Ԃ̂�����ɊJ�u�����B
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
			System.err.println("���R�F" + e.toString());
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
		//timetable.initTimetable();
	}

	//�Ǐ��T��(���P�P,�Q)(�v���O�������̃e�[�u�������������Ȃ�)(����p�e�[�u�����g�p���Ȃ�)
	public int local_search() throws Exception
	{
		//if(debug) SWflag = true;
		int i;
		for (i=1; i<=50; i++)
		{
			if(debug) System.out.println(i+"���");
			if (!search_add2()&&!search_move()&&!search_change()&&!search_swap()&&!search_add()&&!search_leave()){
				if(debug) System.out.println("����ȏ㖳��");
				break;
			}
			if(debug){
				System.out.println("count_offence:"+count_offence());
				System.out.println("valuate:"+valuateTimeTableSQL());
			}
		}
		if(debug) System.out.println("�]���ɂ�����������:"+vsw);
		return i;
	}

	//�Ǐ��T��(���P�P,�Q)
	//(�v���O�������̃e�[�u�������������Ȃ�)
	//(����p�e�[�u�����g�p���Ȃ�)
	//�A�����鎞�Ԋ��ɑΉ�
	public int task_series_local_search() throws Exception
	{
		//if(debug) SWflag = true;
		int i;
		for (i=1; i<=50; i++)
		{
			if(debug) System.out.println(i+"���");
			if(!search_move_task_series()&&!search_change_task_series()&&!search_swap_task_series()&&!search_add_task_series()&&!search_leave_task_series()){
			//if (!search_change_task_series()&&!search_swap_task_series()&&!search_add_task_series()&&!search_leave_task_series()){
				if(debug) System.out.println("����ȏ㖳��");
				break;
			}
			if(debug){
				//System.out.println("count_offence:"+task_series_count_offence());
				//System.out.println("count_offenceDetail:"+task_series_count_offenceDetail());
				//System.out.println("valuate:"+valuateTimeTableSQL());
				//System.out.println("valuate:"+valuateTimeTableSQLDBDetail());
			}
		}
		if(debug) System.out.println("�]���ɂ�����������:"+vsw);
		//printInsertTimeTable();
		return i;
	}
	
	//(�v���O�������̃e�[�u�������������Ȃ�)
	//�ߖT����move:�u�`���s�����Ԃ�ύX����(���ׂĒ��ׂ�)(���\�b�h�g�p)
	//�ߖTmove��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_move() throws Exception{
		boolean change = false;
		int min_val,min_off,new_val,new_off;
		ArrayList new_teacher_ids = null;
		int new_task_id = 0,new_period_id = 0,del_period_id = 0;
		//�l�̕]��
		min_val = valuateTimeTableSQL();
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
			Lecture l = searchLecture(task_id);
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
							new_val = valuateTimeTableSQL();
							if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
							{
								//�ω������������Ă���
								new_teacher_ids = t_teachers;
								new_task_id = task_id;
								new_period_id = period_id2;
								del_period_id = period_id;
								min_off = new_off;
								min_val = new_val;
								//if(debug)System.out.println(""+min_val);
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
						int processor_id = ((Integer)i4.next()).intValue();
						insertTimeTableRow(task_id,period_id,processor_id);
						timetable.insertTimeTableRow(task_id,period_id,processor_id);
					}
				}
			}
		}
		//�������P���ꂽ��
		if(change){
			//�ύX�̕\��
			if(debug)System.out.println("move:task_id:"+new_task_id+"��period_id:"+del_period_id+"����period_id:"+new_period_id+"�Ɉړ�");
			deleteTaskTimeTable(new_task_id);
			timetable.deleteTaskTimeTable(new_task_id);
			Iterator i = new_teacher_ids.iterator();
			int processor_id;
			while(i.hasNext()){
				processor_id = ((Integer)i.next()).intValue();
				if(debug)System.out.println("�S���u�t:"+processor_id);
				insertTimeTableRow(new_task_id,new_period_id,processor_id);
				timetable.insertTimeTableRow(new_task_id,new_period_id,processor_id);
			}
			if(debug)printInsertTimeTable();
		}
		return change;
	}

	//(�v���O�������̃e�[�u�������������Ȃ�)
	//�ߖT����move:�u�`���s�����Ԃ�ύX����(���ׂĒ��ׂ�)(���\�b�h�g�p)
	//�ߖTmove��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_move_task_series() throws Exception{
		if(debug)System.out.println("move start");
		boolean change = false;
		int min_val,min_off,new_val,new_off;
		ArrayList new_teacher_ids = null;
		ArrayList new_n_period_ids = null;
		int new_task_id = 0,new_period_id = 0,new_task_series_num=1,del_period_id = 0;
		//�l�̕]��
		min_val = valuateTimeTableSQL();
		min_off = timetable.taskSeriesCountOffence();
		//���ݎ��Ԋ��ɓ����Ă���u�`���擾
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
								new_off = timetable.taskSeriesCountOffence();
								new_val = valuateTimeTableSQL();
								if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
								{
									//�ω������������Ă���
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
								//�ǉ������u�`�̍폜
								deleteTaskTimeTable(task_id);
								timetable.deleteTaskTimeTable(task_id);
							}
						}
						//������task_id�̒ǉ�
						Iterator i4 = t_teacher3.iterator();
						while(i4.hasNext()){
							int processor_id = ((Integer)i4.next()).intValue();
							insertTimeTableRow(task_id,period_id,processor_id);
							timetable.insertTimeTableRow(task_id,period_id,processor_id);
						}
					}
				}
			}else{
				//task_id�̊J�u�\���Ԃ̎擾
				Iterator i = l.getPeriods().iterator();
				while(i.hasNext()){
					PeriodDesire lpd = (PeriodDesire)i.next();
					int period_id2 = lpd.getPeriod_id();
					//�u�`�̊J�u���Ԃ����݊J�u����Ă��鎞�ԂłȂ�
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
						//task_id��S�����Ă���u�t������
						//period_id��task_id��S�����Ă���u�t���擾
						ArrayList t_teacher3 = timetable.getProcessors(task_id,period_id);
						//task_id������
						//deleteTaskTimeTable(task_id,period_id);
						deleteTaskTimeTable(task_id);
						timetable.deleteTaskTimeTable(task_id,period_id);
						//�V����task_id��S������u�t
						ArrayList t_teachers2 = new ArrayList();
						Iterator i2 = teachers.iterator();
						while(i2.hasNext()){
							Teacher t = (Teacher)i2.next();
							//�u�tt���u�`��S���ł��邩
							if(t.isQualification(qualification_id)){
								if(t.isPeriod(period_id2)){
									//�u�t�͓������Ԃɕ����̍u�`�����Ȃ����Ƃ𒲂ׂ�
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
									Iterator iterator = t_periods.iterator();
									while(iterator.hasNext()){
										int n_period = ((Integer)iterator.next()).intValue();
										insertTimeTableRow(task_id,n_period,processor_id);
									}
								}
								//�l�̕]��
								new_off = timetable.taskSeriesCountOffence();
								new_val = valuateTimeTableSQL();
								if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
								{
									//�ω������������Ă���
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
								//�ǉ������u�`�̍폜
								deleteTaskTimeTable(task_id);
								timetable.deleteTaskTimeTable(task_id);
							}
						}
						//������task_id�̒ǉ�
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
		//�������P���ꂽ��
		if(change){
			//�ύX�̕\��
			if(debug)System.out.println("move:task_id:"+new_task_id+"��period_id:"+del_period_id+"����period_id:"+new_period_id+"�Ɉړ�");
			deleteTaskTimeTable(new_task_id);
			timetable.deleteTaskTimeTable(new_task_id);
			Iterator i = new_teacher_ids.iterator();
			int processor_id;
			if(new_task_series_num==1){
				while(i.hasNext()){
					processor_id = ((Integer)i.next()).intValue();
					if(debug)System.out.println("�S���u�t:"+processor_id);
					insertTimeTableRow(new_task_id,new_period_id,processor_id);
					timetable.insertTimeTableRow(new_task_id,new_period_id,processor_id);
				}
			}else{
				while(i.hasNext()){
					processor_id = ((Integer)i.next()).intValue();
					if(debug)System.out.println("�S���u�t:"+processor_id);
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

	//(�v���O�������̃e�[�u�������������Ȃ�)
	//�ߖT����move:�u�`���s�����Ԃ�ύX����(���ׂĒ��ׂ�)(���\�b�h�g�p)
	//�ߖTmove��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_move_task_series2() throws Exception{
		boolean change = false;
		int min_val,min_off,new_val,new_off;
		ArrayList new_teacher_ids = null;
		ArrayList new_n_period_ids = null;
		int new_task_id = 0,new_period_id = 0,new_task_series_num=1,del_period_id = 0;
		//�l�̕]��
		min_val = valuateTimeTableSQL();
		min_off = timetable.taskSeriesCountOffence();
		//���ݎ��Ԋ��ɓ����Ă���u�`���擾
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
								new_off = timetable.taskSeriesCountOffence();
								new_val = valuateTimeTableSQL();
								if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
								{
									//�ω������������Ă���
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
								//�ǉ������u�`�̍폜
								deleteTaskTimeTable(task_id);
								timetable.deleteTaskTimeTable(task_id);
							}
						}
						//������task_id�̒ǉ�
						Iterator i4 = t_teacher3.iterator();
						while(i4.hasNext()){
							int processor_id = ((Integer)i4.next()).intValue();
							insertTimeTableRow(task_id,period_id,processor_id);
							timetable.insertTimeTableRow(task_id,period_id,processor_id);
						}
					}
				}
			}else{
				//task_id�̊J�u�\���Ԃ̎擾
				Iterator i = l.getPeriods().iterator();
				while(i.hasNext()){
					PeriodDesire lpd = (PeriodDesire)i.next();
					int period_id2 = lpd.getPeriod_id();
					//�u�`�̊J�u���Ԃ����݊J�u����Ă��鎞�ԂłȂ�
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
						//task_id��S�����Ă���u�t������
						//period_id��task_id��S�����Ă���u�t���擾
						ArrayList t_teacher3 = timetable.getProcessors(task_id,period_id);
						//task_id������
						//deleteTaskTimeTable(task_id,period_id);
						deleteTaskTimeTable(task_id);
						timetable.deleteTaskTimeTable(task_id,period_id);
						//�V����task_id��S������u�t
						ArrayList t_teachers2 = new ArrayList();
						Iterator i2 = teachers.iterator();
						while(i2.hasNext()){
							Teacher t = (Teacher)i2.next();
							//�u�tt���u�`��S���ł��邩
							if(t.isQualification(qualification_id)){
								if(t.isPeriod(period_id2)){
									//�u�t�͓������Ԃɕ����̍u�`�����Ȃ����Ƃ𒲂ׂ�
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
									Iterator iterator = t_periods.iterator();
									while(iterator.hasNext()){
										int n_period = ((Integer)iterator.next()).intValue();
										insertTimeTableRow(task_id,n_period,processor_id);
									}
								}
								//�l�̕]��
								new_off = timetable.taskSeriesCountOffence();
								new_val = valuateTimeTableSQL();
								if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
								{
									//�ω������������Ă���
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
								//�ǉ������u�`�̍폜
								deleteTaskTimeTable(task_id);
								timetable.deleteTaskTimeTable(task_id);
							}
						}
						//������task_id�̒ǉ�
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
		//�������P���ꂽ��
		if(change){
			//�ύX�̕\��
			if(debug)System.out.println("move:task_id:"+new_task_id+"��period_id:"+del_period_id+"����period_id:"+new_period_id+"�Ɉړ�");
			deleteTaskTimeTable(new_task_id);
			timetable.deleteTaskTimeTable(new_task_id);
			Iterator i = new_teacher_ids.iterator();
			int processor_id;
			if(new_task_series_num==1){
				while(i.hasNext()){
					processor_id = ((Integer)i.next()).intValue();
					if(debug)System.out.println("�S���u�t:"+processor_id);
					insertTimeTableRow(new_task_id,new_period_id,processor_id);
					timetable.insertTimeTableRow(new_task_id,new_period_id,processor_id);
				}
			}else{
				while(i.hasNext()){
					processor_id = ((Integer)i.next()).intValue();
					if(debug)System.out.println("�S���u�t:"+processor_id);
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

	//(�v���O�������̃e�[�u�������������Ȃ�)	
	//�ߖT����Change:�u�`�̒S���u�t��ύX����(���\�b�h�g�p)
	//�ߖTchange��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_change() throws Exception{
		ArrayList t_timetable = null;
		boolean change = false;
		int min_val,min_off,new_val,new_off;
		int del_processor_id = 0,new_task_id = 0,new_processor_id = 0,new_period_id = 0;
		//�l�̕]��
		min_val = valuateTimeTableSQL();
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
			Lecture l = searchLecture(task_id);
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
								new_val = valuateTimeTableSQL();
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
			if(debug){
				System.out.println("task_id:"+new_task_id+" period_id:"+new_period_id);
				System.out.println("change:processor_id:"+del_processor_id+"��processor_id:"+new_processor_id+"�ɕω�");
			}
		}
		return change;
	}

	//(�v���O�������̃e�[�u�������������Ȃ�)	
	//�ߖT����Change:�u�`�̒S���u�t��ύX����(���\�b�h�g�p)
	//�ߖTchange��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_change_task_series() throws Exception{
		if(debug)System.out.println("change start");
		//printInsertTimeTable();
		ArrayList t_timetable = null;
		boolean change = false;
		int min_val,min_off,new_val,new_off;
		int del_processor_id = 0,new_task_id = 0,new_task_series_num=1,new_processor_id = 0,new_period_id = 0;
		//�l�̕]��
		min_val = valuateTimeTableSQL();
		min_off = timetable.taskSeriesCountOffence();
		t_timetable = timetable.getTimeTableRowsCopy();
		//���݂̎��Ԋ��̎擾
		Iterator i_t_timetable = t_timetable.iterator();
		int task_id, period_id, processor_id,qualification_id,task_series_num;
		while(i_t_timetable.hasNext()) {
			TimeTableRow ttr = (TimeTableRow)i_t_timetable.next();
			task_id = ttr.getTask_id();
			period_id = ttr.getPeriod_id();
			processor_id = ttr.getProcessor_id();
			//task_id�̍u�`��ނ𒲂ׂ�
			Lecture l = searchLecture(task_id);
			qualification_id = l.getQualification_id();
			task_series_num = l.getTask_series_num();
			//(task,period,processor)����菜��
			deleteTimeTableRow(task_id,period_id,processor_id);
			timetable.deleteTimeTableRow(task_id, period_id, processor_id);
			if(task_series_num>1){
				int n_period = period_id;
				for(int num=1;num<task_series_num;num++){
					n_period = day_periods.getNextPeriods(n_period);
					deleteTimeTableRow(task_id,n_period,processor_id);
				}
			}
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
								//(task_id,period_id,processor_id2)�̒ǉ�
								insertTimeTableRow(task_id,period_id,processor_id2);
								timetable.insertTimeTableRow(task_id,period_id,processor_id2);
								if(task_series_num>1){
									int n_period = period_id;
									for(int num=1;num<task_series_num;num++){
										n_period = day_periods.getNextPeriods(n_period);
										insertTimeTableRow(task_id,n_period,processor_id2);
									}
								}
								//�l�̕]��
								new_off = timetable.taskSeriesCountOffence();
								new_val = valuateTimeTableSQL();
								if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
								{
									//�ω������������Ă���
									new_task_id = task_id;
									new_task_series_num=task_series_num;
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
			//��菜����(task_id,period_id,processor_id)�̒ǉ�
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
		//�������P���ꂽ��
		if(change){
			//���܂ł̉���ύX����
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
			//�ύX�̕\��
			if(debug){
				System.out.println("task_id:"+new_task_id+" period_id:"+new_period_id);
				System.out.println("change:processor_id:"+del_processor_id+"��processor_id:"+new_processor_id+"�ɕω�");
			}
		}
		return change;
	}
	
//(�v���O�������̃e�[�u�������������Ȃ�)	
	//�ߖT����Swap:�قȂ�2�̒S���u�t�����ւ���(���\�b�h�g�p)
	//�ߖTswap��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_swap() throws Exception{
		ArrayList t_timetable = null;
		boolean change = false;
		int min_val,min_off,new_val,new_off;
		int new_task_id1 = 0,new_processor_id1 = 0,new_period_id1 = 0,new_task_id2 = 0,new_processor_id2 = 0,new_period_id2 = 0;
		int next = 0;
		//�l�̕]��
		min_val = valuateTimeTableSQL();
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
					Lecture l1 = searchLecture(task_id1);
					qualification_id1 = l1.getQualification_id();
					Lecture l2 = searchLecture(task_id2);
					qualification_id2 = l2.getQualification_id();
					//(task_id1,period_id1,processor_id2)���L�����ǂ���
					//processor_id2��period_id1�ɉ\���ǂ����H
					//�u�tt��processor_id2�ł���
					Teacher t = searchTeacher(processor_id2);
					if(t.isQualification(qualification_id1)){
						//processor_id2��period_id1�ɉ\
						if(t.isPeriod(period_id1)){
							//�u�t�͓������Ԃɕ����̍u�`�����Ȃ�
							if(timetable.isNotTasks(processor_id2,period_id1)){
								//(task_id2,period_id2,processor_id1)���L�����ǂ���
								//processor_id1��period_id2�ɒS���\���ǂ����H
								//�u�tt2��processor_id1�ł���
								Teacher t2 = searchTeacher(processor_id1);
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
											new_val = valuateTimeTableSQL();
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
			if(debug){
				System.out.println("(task_id,processor_id,period_id)");
				System.out.println("swap:("+new_task_id1+","+new_processor_id1+","+new_period_id1+")��("+new_task_id2+","+new_processor_id2+","+new_period_id2+")");
			}
			
		}
		return change;
	}

	//(�v���O�������̃e�[�u�������������Ȃ�)	
	//�ߖT����Swap:�قȂ�2�̒S���u�t�����ւ���(���\�b�h�g�p)
	//�ߖTswap��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_swap_task_series() throws Exception{
		if(debug)System.out.println("swap start");
		//printInsertTimeTable();
		ArrayList t_timetable = null;
		boolean change = false;
		int min_val,min_off,new_val,new_off;
		int new_task_id1 = 0,new_processor_id1 = 0,new_period_id1 = 0,new_task_id2 = 0,new_processor_id2 = 0,new_period_id2 = 0,new_task_series_num1=1,new_task_series_num2=1;
		int next = 0;
		//�l�̕]��
		min_val = valuateTimeTableSQL();
		min_off = timetable.taskSeriesCountOffence();
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
				int qualification_id1,qualification_id2,task_series_num1,task_series_num2;
				//�قȂ�u�`���ǂ���
				if(task_id1!=task_id2){
					//task_id1��task_id2�̎�ނ𒲂ׂ�
					Lecture l1 = searchLecture(task_id1);
					qualification_id1 = l1.getQualification_id();
					task_series_num1 = l1.getTask_series_num();
					Lecture l2 = searchLecture(task_id2);
					qualification_id2 = l2.getQualification_id();
					task_series_num2 = l2.getTask_series_num();
					//(task_id1,period_id1,processor_id1)��(task_id2,period_id2,processor_id2)����菜��
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
					//(task_id1,period_id1,processor_id2)���L�����ǂ���
					//processor_id2��period_id1�ɉ\���ǂ����H
					//�u�tt��processor_id2�ł���
					Teacher t = searchTeacher(processor_id2);
					if(t.isQualification(qualification_id1)){
						//processor_id2��period_id1�ɉ\
						if(t.isPeriod(period_id1)){
							//�u�t�͓������Ԃɕ����̍u�`�����Ȃ�
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
									//��菜����(task_id1,period_id1,processor_id1)��(task_id2,period_id2,processor_id2)��ǉ�����
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
								//(task_id2,period_id2,processor_id1)���L�����ǂ���
								//processor_id1��period_id2�ɒS���\���ǂ����H
								//�u�tt2��processor_id1�ł���
								Teacher t2 = searchTeacher(processor_id1);
								if(t2.isQualification(qualification_id2)){
									//processor_id1��period_id2�ɉ\
									if(t2.isPeriod(period_id2)){
										//�u�t�͓������Ԃɕ����̍u�`�����Ȃ�
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
												//��菜����(task_id1,period_id1,processor_id1)��(task_id2,period_id2,processor_id2)��ǉ�����
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
											//(task_id1,period_id1,processor_id2)�̒ǉ�
											insertTimeTableRow(task_id1,period_id1,processor_id2);
											//(task_id2,period_id2,processor_id1)�̒ǉ�
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
												//�ω������������Ă���
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
											//�ǉ�����(task_id1,period_id1,processor_id2)�̍폜
											deleteTimeTableRow(task_id1,period_id1,processor_id2);
											//�ǉ�����(task_id2,period_id2,processor_id1)�̍폜
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
					//��菜����(task_id1,period_id1,processor_id1)��(task_id2,period_id2,processor_id2)��ǉ�����
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
		//�������P���ꂽ��
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
			//�ύX�̕\��
			if(debug){
				System.out.println("(task_id,processor_id,period_id)");
				System.out.println("swap:("+new_task_id1+","+new_processor_id1+","+new_period_id1+")��("+new_task_id2+","+new_processor_id2+","+new_period_id2+")");
			}
			if(debug)printTimeTable();
		}
		return change;
	}
	
	//(�v���O�������̃e�[�u�������������Ȃ�)	
	//�ߖT����add:�u�`�̒S���u�t��ǉ�����(���\�b�h�g�p)
	//�ߖTadd��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_add() throws Exception{
		boolean change = false;
		ArrayList new_teacher_ids = null;
		int min_val,min_off,new_val,new_off;
		int new_task_id = 0,new_period_id = 0;
		//�l�̕]��
		min_val = valuateTimeTableSQL();
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
			Lecture l = searchLecture(task_id);
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
					new_val = valuateTimeTableSQL();
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

	//(�v���O�������̃e�[�u�������������Ȃ�)	
	//�ߖT����add:�u�`�̒S���u�t��ǉ�����(���\�b�h�g�p)
	//�ߖTadd��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_add_task_series() throws Exception{
		if(debug)System.out.println("add start");
		//printInsertTimeTable();
		boolean change = false;
		ArrayList new_teacher_ids = null;
		int min_val,min_off,new_val,new_off;
		int new_task_id = 0,new_period_id = 0,new_task_series_num=1;
		//�l�̕]��
		min_val = valuateTimeTableSQL();
		min_off = timetable.taskSeriesCountOffence();
		int task_id,qualification_id,period_id,task_num,low,task_series_num;
		//���݂̎��Ԋ��̍u�`�Ƃ��̒S���l���ƊJ�u���Ԃ��擾
		ArrayList list = timetable.getTimeTableTasksNum();
		Iterator i_list = list.iterator();
		while(i_list.hasNext()){
			TaskPeriodNum tpn = (TaskPeriodNum)i_list.next();
			task_id = tpn.getTask_id();
			period_id = tpn.getPeriod_id();
			task_num = tpn.getNum();
			//task_id�ł���u�`�̎擾
			Lecture l = searchLecture(task_id);
			qualification_id = l.getQualification_id();
			task_series_num = l.getTask_series_num();
			//period_id�ɒS���\�ȍu�t��id�����Ă���
			ArrayList t_teachers2 = null;
			//task_id�̒S���l������ɑ���Ȃ��l���̎擾
			low = l.getRequired_processors_ub()-task_num;
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
						if(task_series_num>1){
							int n_period = period_id;
							for(int num=1;num<task_series_num;num++){
								n_period = day_periods.getNextPeriods(n_period);
								insertTimeTableRow(task_id,n_period,processor_id);
							}
						}
						t_teachers.add(new Integer(processor_id));
					}
					//�l�̕]��
					new_off = timetable.taskSeriesCountOffence();
					new_val = valuateTimeTableSQL();
					if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
					{
						//�ω������������Ă���
						new_task_series_num=task_series_num;
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
		//�������P���ꂽ��
		if(change){
			//�ύX�̕\��
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

	//(�v���O�������̃e�[�u�������������Ȃ�)
	//�ߖT����add:�u�`���J�u����ĂȂ����ǉ�(���\�b�h�g�p)
	//�ߖTadd��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_add2() throws Exception{
		boolean change = false;
		ArrayList new_teacher_ids = null;
		int min_val,min_off,new_val,new_off;
		int new_task_id = 0,new_period_id = 0;
		//�l�̕]��
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
				//task_id�̒S���l������ɑ���Ȃ��l���̎擾
				low = l.getRequired_processors_ub()-num;
				//task_id�̊J�u�\���Ԃ̎擾
				Iterator i = l.getPeriods().iterator();
				while(i.hasNext()){
					PeriodDesire lpd = (PeriodDesire)i.next();
					period_id = lpd.getPeriod_id();
					//period_id�ɒS���\�ȍu�t��id�����Ă���
					ArrayList t_teachers2 = new ArrayList();
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
							new_val = valuateTimeTableSQL();
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
			}
		}
		//�������P���ꂽ��
		if(change){
			//�ύX�̕\��
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

	//(�v���O�������̃e�[�u�������������Ȃ�)
	//�ߖT����add:�u�`���J�u����ĂȂ����ǉ�(���\�b�h�g�p)
	//�ߖTadd��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_add2_task_series() throws Exception{
		boolean change = false;
		ArrayList new_teacher_ids = null;
		int min_val,min_off,new_val,new_off;
		int new_task_id = 0,new_period_id = 0,new_task_series_num=1;
		//�l�̕]��
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
				//task_id�̒S���l������ɑ���Ȃ��l���̎擾
				low = l.getRequired_processors_ub()-task_num;
				//task_id�̊J�u�\���Ԃ̎擾
				Iterator i = l.getPeriods().iterator();
				while(i.hasNext()){
					PeriodDesire lpd = (PeriodDesire)i.next();
					period_id = lpd.getPeriod_id();
					//period_id�ɒS���\�ȍu�t��id�����Ă���
					ArrayList t_teachers2 = new ArrayList();
					Iterator i3 = teachers.iterator();
					while(i3.hasNext()){
						Teacher t = (Teacher)i3.next();
						if(t.isQualification(qualification_id)){
							//t��period_id�ŒS���\
							if(t.isPeriod(period_id)){
								int processor_id = t.getProcessor_id();
								//�u�t�͓������Ԃɕ����̍u�`�����Ȃ�
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
									//System.out.println("�ǉ��\:"+processor_id);
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
								if(task_series_num>1){
									int n_period = period_id;
									for(int num=1;num<task_series_num;num++){
										n_period = day_periods.getNextPeriods(n_period);
										insertTimeTableRow(task_id,n_period,processor_id);
									}
								}
								t_teachers.add(new Integer(processor_id));
							}
							//�l�̕]��
							new_off = timetable.taskSeriesCountOffence();
							new_val = valuateTimeTableSQL();
							if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
							{
								//�ω������������Ă���
								new_task_series_num=task_series_num;
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
		//�������P���ꂽ��
		if(change){
			//�ύX�̕\��
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
	//(�v���O�������̃e�[�u�������������Ȃ�)	
	//�ߖT����leave:�u�`�̒S���u�t�����炷(���\�b�h�g�p)
	//�ߖTleave��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_leave() throws Exception{
		boolean change = false;
		ArrayList new_teacher_ids = null;
		int min_val,min_off,new_val,new_off;
		int new_task_id = 0,new_period_id = 0;
		//�l�̕]��
		min_val = valuateTimeTableSQL();
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
			Lecture l = searchLecture(task_id);
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
					new_val = valuateTimeTableSQL();
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
						int processor_id = ((Integer)i5.next()).intValue();
						insertTimeTableRow(task_id,period_id,processor_id);
						timetable.insertTimeTableRow(task_id,period_id,processor_id);
					}
				}
			}
		}
		//�������P���ꂽ��
		if(change){
			//�ύX�̕\��
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

	//(�v���O�������̃e�[�u�������������Ȃ�)	
	//�ߖT����leave:�u�`�̒S���u�t�����炷(���\�b�h�g�p)
	//�ߖTleave��T�����A���P��������΂��̒��ōŗǂ̂��̂Ɉړ���true��Ԃ��B�Ȃ����false��Ԃ��B
	public boolean search_leave_task_series() throws Exception{
		if(debug)System.out.println("leave start");
		//printInsertTimeTable();
		boolean change = false;
		ArrayList new_teacher_ids = null;
		int min_val,min_off,new_val,new_off;
		int new_task_id = 0,new_period_id = 0,new_task_series_num=1;
		//�l�̕]��
		min_val = valuateTimeTableSQL();
		min_off = timetable.taskSeriesCountOffence();
		//���݂̎��Ԋ��̍u�`�Ƃ��̒S���l���ƊJ�u���Ԃ��擾
		ArrayList list = timetable.getTimeTableTasksNum();
		Iterator i_list = list.iterator();
		int task_id,period_id,task_num,task_series_num;
		while(i_list.hasNext()){
			TaskPeriodNum tpn = (TaskPeriodNum)i_list.next();
			task_id = tpn.getTask_id();
			period_id = tpn.getPeriod_id();
			task_num = tpn.getNum();
			//task_id�ł���u�`�̎擾
			Lecture l = searchLecture(task_id);
			//task_id�̒S���l�������ȏ�̐l���̎擾
			int up = task_num-l.getRequired_processors_lb();
			task_series_num=l.getTask_series_num();
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
						if(task_series_num>1){
							int n_period = period_id;
							for(int num=1;num<task_series_num;num++){
								n_period = day_periods.getNextPeriods(n_period);
								deleteTimeTableRow(task_id,n_period,processor_id);
							}
						}
						t_teachers.add(new Integer(processor_id));
					}
					//�l�̕]��
					new_off = timetable.taskSeriesCountOffence();
					new_val = valuateTimeTableSQL();
					if ((new_off < min_off) || ((new_off == min_off) && (new_val < min_val)))
					{
						//�ω������������Ă���
						new_teacher_ids = t_teachers;
						new_task_series_num=task_series_num;
						new_task_id = task_id;
						new_period_id = period_id;
						min_off = new_off;
						min_val = new_val;
						change = true;
					}
					//�폜�����u�t��ǉ�
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
		//�������P���ꂽ��
		if(change){
			//�ύX�̕\��
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
	
	//���̕]���l��v�]�ʂɐ����č��v��Ԃ�
	public int valuateTimeTableSQL() throws Exception{
		if(valuateType==1) return valuateTimeTableSQLFile();
		else return valuateTimeTableSQLDB();
	}
	
	//���̕]���l��v�]�ʂɐ����č��v��Ԃ�(timetablesql�ɑ΂��čs��)
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
		if(debug)vsw.suspend();
		return penalties;
	}
	
	//���̕]���l��v�]�ʂɐ����č��v��Ԃ�(timetablesql�ɑ΂��čs��)
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
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
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
		if(debug) vsw.suspend();
		return penalties;
	}

	//���̕]���l��v�]�ʂɐ����č��v��Ԃ�(timetablesql�ɑ΂��čs��)
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
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
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
					System.out.println("�]��SQL");
					Iterator i4 = v_sqls_p.getV_sqls().getV_sqls().iterator();
					while(i4.hasNext()){
						t2 = (ValuateSQL)i4.next();
						System.out.println(t2.getSql());
					}
					System.out.println("�]���l:"+v_sqls_p.getPenalty());
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
		if(debug) vsw.suspend();
		return penalties;
	}

	//���̐���ᔽ����ޕʂɐ����č��v��Ԃ�
	public int count_offence(){
		return timetable.countOffence();
	}
	//���̐���ᔽ����ޕʂɐ����č��v��Ԃ�
	//�A������u�`�ɑΉ�
	public int task_series_count_offence(){
		return timetable.taskSeriesCountOffence();
	}
	//���̐���ᔽ����ޕʂɐ����č��v��Ԃ�(�ڍ׏��\��)
	public int count_offenceDetail(){
		return timetable.countOffenceDetail();
	}
	//���̐���ᔽ����ޕʂɐ����č��v��Ԃ�(�ڍ׏��\��)
	//�A������u�`�ɑΉ�
	public int task_series_count_offenceDetail(){
		return timetable.taskSeriesCountOffenceDetail();
	}
	
	//�e�[�u��table(�t�B�[���hproject_id�ǉ�)��timetablesql�ɓǂݍ���
	public void loadTimeTableAddProject(String table,int project_id) throws Exception{
		Connection con = null;
		Statement smt = null;
		try{
			con = cmanager.getConnection();
			smt = con.createStatement();
			//�폜
			String sql = "delete from timetablesql";
			smt.executeUpdate(sql);
			sql = "insert into timetablesql(task_id,period_id,processor_id) select task_id , period_id , processor_id from "+ table+ " where project_id="+project_id;
			smt.executeUpdate(sql);
		}catch(SQLException e) {
			System.out.println(e);throw e;
		}finally{
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

	//�e�[�u��table��timetablesql�ɓǂݍ���
	public void loadTimeTable(String table) throws Exception{
		Connection con = null;
		Statement smt = null;
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			//��DB��ɂ��łɑ��݂��Ă���ꍇ���̃e�[�u�����폜����
			String sql = "drop table if exists timetablesql";
			smt.executeUpdate(sql);
			sql = "create table timetablesql as select * from "+table;
			smt.executeUpdate(sql);
		}catch(SQLException e) {
			System.out.println(e);				
			throw e;
		}finally{
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
	
	//DB���timetablesql�̃f�[�^��table���ŕۑ�����
	//�v���W�F�N�g�Ǘ����ĂȂ���
	public void saveTimeTable(String table) throws Exception{
		Connection con = null;
		Statement smt = null;
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			//DB��ɂ��łɑ��݂��Ă���ꍇ���̃e�[�u�����폜����
			String sql = "drop table if exists " + table;
			smt.executeUpdate(sql);
			sql = "create table " + table + " as select * from timetablesql";
			smt.executeUpdate(sql);
		}catch(SQLException e) {
			System.out.println(e);				
			throw e;
		}finally{
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

	//�o�O������
	//DB���timetablesql�̃f�[�^��table(�t�B�[���h:projectu_id�ǉ�)�ŕۑ�����
	public void saveTimeTableAddProject(String table,int project_id) throws Exception{
		Connection con = null;
		PreparedStatement psmt = null;
		Statement smt = null;
		ResultSet rs = null;
		try{
			con = cmanager.getConnection();
			smt = con.createStatement();
			//DB��ɂ��łɑ��݂��Ă���ꍇ,�폜����
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
			//�R�l�N�V�����v�[���֕ԋp
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}
	//DB���timetablesql�̃f�[�^��table(�t�B�[���h:projectu_id�ǉ�)�ŕۑ�����
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
			//�R�l�N�V�����v�[���֕ԋp
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}
	//DB���table���̃e�[�u���̗v�f��project_id�ł�����̂��폜
	public void deleteTimetable(String table,int project_id) throws Exception{
		Connection con = null;
		Statement smt = null;
		try{
			con = cmanager.getConnection();
			smt = con.createStatement();
			//DB��ɂ��łɑ��݂��Ă���ꍇ���̃e�[�u�����폜����
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
			//�R�l�N�V�����v�[���֕ԋp
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}
	//���X�g:lectures����task_id�ł���u�`���擾����
	private Lecture searchLecture(int task_id){
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
	
	//���X�g:teachers����processor_id�ł���u�t���擾����
	private Teacher searchTeacher(int processor_id){
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
	
	//(task_id,period_id,processor_id)�����Ԋ�timetablesql�ɒǉ�
	private void insertTimeTableRow(int task_id,int period_id,int processor_id) throws Exception{
		Connection con = null;
		Statement smt = null;
		try{
			con = cmanager.getConnection();
			smt = con.createStatement();
			//�ǉ�
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
			//�R�l�N�V�����v�[���֕ԋp
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}
	//(task_id,period_id,processor_id)�����Ԋ�timetablesql����폜
	private void deleteTimeTableRow(int task_id,int period_id, int processor_id) throws Exception{
		Connection con = null;
		Statement smt = null;
		try{
			con = cmanager.getConnection();
			smt = con.createStatement();
			//�폜
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
			//�R�l�N�V�����v�[���֕ԋp
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}
	
	//��Ǝ���period_id�̍u�`task_id�����Ԋ�timetablesql����폜
	private void deleteTaskTimeTable(int task_id, int period_id) throws Exception{
		Connection con = null;
		Statement smt = null;
		try{
			con = cmanager.getConnection();
			smt = con.createStatement();
			//�폜
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
			//�R�l�N�V�����v�[���֕ԋp
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}
	
	//�u�`task_id�����Ԋ�timetablesql����폜
	private void deleteTaskTimeTable(int task_id) throws Exception{
		Connection con = null;
		Statement smt = null;
		try{
			con = cmanager.getConnection();
			smt = con.createStatement();
			//�폜
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
			//�R�l�N�V�����v�[���֕ԋp
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}
	//���Ԋ�timetablesql��(old_task_id,old_period_id,old_processor_id)���u�tnew_processor_id�ɕύX
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
			//�R�l�N�V�����v�[���֕ԋp
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}
	
	//���݂̎��Ԋ��̏��(�v���O������)��\��
	public void printTimeTable(){
		timetable.printTimeTable();
	}

	//insert���Ō��݂̎��Ԋ��̏��(�v���O������)��\��
	public void printInsertTimeTable(){
		timetable.printInsertTimeTable();
	}
	/**
	 * @return cmanager ��߂��܂��B
	 */
	public DBConnectionPool getCmanager() {
		return cmanager;
	}
	/**
	 * @param cmanager cmanager ��ݒ�B
	 */
	public void setCmanager(DBConnectionPool cmanager) {
		this.cmanager = cmanager;
	}
}
