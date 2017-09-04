package sql_test;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

import org.apache.commons.lang.time.StopWatch;

import common.ConnectionManager;


public class SQLTest {
	public static void main(String args[]) throws Exception{
		ConnectionManager cmanager = new ConnectionManager("jdbc:mysql://localhost/schedule?useUnicode=true&characterEncoding=sjis","yukiko","nishimura");
		StopWatch sp = new StopWatch();
		StopWatch spDrop = new StopWatch();
		spDrop.start();
		spDrop.suspend();
		StopWatch spCreate = new StopWatch();
		spCreate.start();
		spCreate.suspend();
		StopWatch spSelect = new StopWatch();
		spSelect.start();
		spSelect.suspend();
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		int penalties = 0;
		int mode = 1;
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			sp.start();
			for(int n = 0; n < 10000; n++){
				if(mode==1){
					//�e�u�`�̒S���u�t�̐l���̏�������邩�ǂ���?
					spCreate.resume();
					String sql = "create temporary table t select a.task_id,count(*),b.required_processors_ub,count(*)-CAST(b.required_processors_ub AS SIGNED) as num from timetableSQL a, task_properties b where a.task_id = b.task_id group by a.task_id,b.required_processors_lb having count(*)-CAST(b.required_processors_ub AS SIGNED) > 0";
					smt.executeUpdate(sql);
					spCreate.suspend();				
					spSelect.resume();
					sql = "select sum(num)*10000 as penalties from t";
					rs = smt.executeQuery(sql);
					if(rs.next()){
						penalties = rs.getInt("penalties");
					}
					spSelect.suspend();
					spDrop.resume();
					sql = "drop table t";
					smt.executeUpdate(sql);
					spDrop.suspend();
				}else if(mode==2){
					//�e�u�`�̒S���u�t�̐l���̏�������邩�ǂ����H
					String sql = "select sum(num)*10000 as penalties from (select a.task_id,count(*),b.required_processors_ub,count(*)-CAST(b.required_processors_ub AS SIGNED) as num from timetableSQL a, task_properties b where a.task_id = b.task_id group by a.task_id,b.required_processors_lb having count(*)-CAST(b.required_processors_ub AS SIGNED) > 0) as t";
					rs = smt.executeQuery(sql);
					if(rs.next()){
						penalties = rs.getInt("penalties");
					}
					spSelect.suspend();
				}else{
					//�e�u�`�̒S���u�t�̐l���̏�������邩�ǂ���?
					spCreate.resume();
					String sql = "create table t select * from penalties";
					smt.executeUpdate(sql);
					spCreate.suspend();
					spDrop.resume();
					sql = "drop table t";
					smt.executeUpdate(sql);
					spDrop.suspend();
				}
			}
			sp.stop();
			System.out.println(sp);
			System.out.println("create:"+spCreate);
			System.out.println("select:"+spSelect);
			System.out.println("drop:"+spDrop);
		}catch(SQLException e) {
			System.out.println(e);
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
				con.close();
			}
		}
	}
}
