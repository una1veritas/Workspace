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
			//MySQLサーバ接続
			con = cmanager.getConnection();
			//Statementインターフェースの生成
			smt = con.createStatement();
			sp.start();
			for(int n = 0; n < 10000; n++){
				if(mode==1){
					//各講義の担当講師の人数の上限を上回るかどうか?
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
					//各講義の担当講師の人数の上限を上回るかどうか？
					String sql = "select sum(num)*10000 as penalties from (select a.task_id,count(*),b.required_processors_ub,count(*)-CAST(b.required_processors_ub AS SIGNED) as num from timetableSQL a, task_properties b where a.task_id = b.task_id group by a.task_id,b.required_processors_lb having count(*)-CAST(b.required_processors_ub AS SIGNED) > 0) as t";
					rs = smt.executeQuery(sql);
					if(rs.next()){
						penalties = rs.getInt("penalties");
					}
					spSelect.suspend();
				}else{
					//各講義の担当講師の人数の上限を上回るかどうか?
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
				con.close();
			}
		}
	}
}
