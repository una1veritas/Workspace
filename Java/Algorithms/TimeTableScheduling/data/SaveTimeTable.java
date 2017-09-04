package data;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Iterator;

import common.DBConnectionPool;

/**
 * @author �Y�����Y
 *
 * TODO ���Ԋ��̏��ƕ]���l���Ǘ�(�ꎞ�I)
 *
 */
public class SaveTimeTable implements Comparable{
	private int count_offence;
	private int valuate;
	private ArrayList timetablerows;
	
	public SaveTimeTable(ArrayList timetablerows,int count_offence,int valuate){
		this.timetablerows = timetablerows;
		this.count_offence = count_offence;
		this.valuate = valuate;
	}
	/**
	 * @return count_offence ��߂��܂��B
	 */
	public int getCount_offence() {
		return count_offence;
	}
	/**
	 * @param count_offence count_offence ��ݒ�B
	 */
	public void setCount_offence(int count_offence) {
		this.count_offence = count_offence;
	}
	/**
	 * @return timetablerows ��߂��܂��B
	 */
	public ArrayList getTimetablerows() {
		return timetablerows;
	}
	/**
	 * @param timetablerows timetablerows ��ݒ�B
	 */
	public void setTimetablerows(ArrayList timetablerows) {
		this.timetablerows = timetablerows;
	}
	/**
	 * @return valuate ��߂��܂��B
	 */
	public int getValuate() {
		return valuate;
	}
	/**
	 * @param valuate valuate ��ݒ�B
	 */
	public void setValuate(int valuate) {
		this.valuate = valuate;
	}
	
	public int compareTo(Object o){
		SaveTimeTable stt = (SaveTimeTable)o;
		int temp = 0;
		if(stt.count_offence<count_offence)temp = 1;
		else if(stt.count_offence>count_offence)temp = -1;
		else{
			if(stt.valuate<valuate)temp = 1;
			else if(stt.valuate>valuate)temp = -1;
		}
		return temp;
	}
	
	public boolean equals(Object o){
		if(!(o instanceof SaveTimeTable))
			return false;
		SaveTimeTable stt = (SaveTimeTable)o;
		return (stt.count_offence==count_offence&&stt.valuate==valuate);
	}
	
	public void saveTimetablesql(DBConnectionPool cmanager) throws Exception{
		Connection con = null;
		PreparedStatement psmt = null;
		Statement smt = null;
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			smt.executeUpdate("delete from timetablesql");
			psmt = con.prepareStatement("insert into timetablesql(task_id,period_id,processor_id) values( ? , ? , ? )");
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
	//DB���table�Ɏ��Ԋ��̏����R�s�[
	public void saveTimetable(DBConnectionPool cmanager,String table) throws Exception{
		Connection con = null;
		PreparedStatement psmt = null;
		Statement smt = null;
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			//DB��ɂ��łɑ��݂��Ă���ꍇ���̃e�[�u�����폜����
			String sql = "drop table if exists " + table;
			smt.executeUpdate(sql);
			//DB��Ƀe�[�u��table���쐬
			sql = "create table " + table + " as select * from timetablesql";
			smt.executeUpdate(sql);
			smt.executeUpdate("delete from " +table);
			psmt = con.prepareStatement("insert into "+table+"(task_id,period_id,processor_id) values( ? , ? , ? )");
			Iterator i = timetablerows.iterator();
			while(i.hasNext()){
				TimeTableRow t = (TimeTableRow)i.next();
				psmt.setInt(1,t.getTask_id());
				psmt.setInt(2,t.getPeriod_id());
				psmt.setInt(3,t.getProcessor_id());
				psmt.executeUpdate();
				
			}
		}catch(SQLException e) {
			System.out.println(e);throw e;
		}finally{
			if(psmt!=null){
				smt.close();
				smt = null;
			}
			if(smt!=null){
				smt.close();
				smt = null;
			}
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}
	//DB���table�Ɏ��Ԋ��̏����R�s�[(�t�B�[���hproject_id�ǉ�)
	public void saveTimeTableAddProject(DBConnectionPool cmanager,String table,int project_id) throws Exception{
		Connection con = null;
		PreparedStatement psmt = null;
		ResultSet rs = null;
		Statement smt = null;
		try{
			con = cmanager.getConnection();
			smt = con.createStatement();
			String sql = "delete from " + table + " where project_id="+project_id;
			smt.executeUpdate(sql);
			psmt = con.prepareStatement("insert into "+table+"(task_id,period_id,processor_id,project_id) values( ? , ? , ? ,?)");
			Iterator i = timetablerows.iterator();
			while(i.hasNext()){
				TimeTableRow t = (TimeTableRow)i.next();
				psmt.setInt(1,t.getTask_id());
				psmt.setInt(2,t.getPeriod_id());
				psmt.setInt(3,t.getProcessor_id());
				psmt.setInt(4,project_id);
				psmt.executeUpdate();
				
			}
		}catch(SQLException e) {
			System.out.println(e);throw e;
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
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}
}
