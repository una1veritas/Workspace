package tool;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

import common.ConnectionManager;

/*
 * 作成日: 2007/07/25
 *
 * TODO この生成されたファイルのテンプレートを変更するには次へジャンプ:
 * ウィンドウ - 設定 - Java - コード・スタイル - コード・テンプレート
 */

/**
 * @author 浦島太郎
 *
 * TODO データベースに登録されているidを指定された値から振りなおす:
 *
 */
public class IDNumbering {
	public static void main(String args[]) throws Exception{
		String host = "localhost";
		String database ="temp_schedule";
		String user = "yukiko";
		String pass = "nishimura";
		ConnectionManager cmanager = new ConnectionManager("jdbc:mysql://"+host+"/"+database+"?useUnicode=true&characterEncoding=sjis",user,pass);
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		Statement smt2 = null;
		int id = 14;
		//mode 1:講義 2:講師
		int mode = 2;
		try{
			//MySQLサーバ接続
			con = cmanager.getConnection();
			//Statementインターフェースの生成
			smt = con.createStatement();
			smt2 = con.createStatement();
			if(mode==1){
				String sql = "select task_id from tasks order by task_id";
				rs = smt.executeQuery(sql);
				while(rs.next()){
					int task_id = rs.getInt("task_id");
					String sql2 = "update tasks set task_id = "+id+" where task_id = "+task_id;
					smt2.executeUpdate(sql2);
					sql2 = "update task_opportunities set task_id = "+id+" where task_id = "+task_id;
					smt2.executeUpdate(sql2);
					sql2 = "update task_properties set task_id = "+id+" where task_id = "+task_id;
					smt2.executeUpdate(sql2);
					id++;
				}
			}else{
				String sql = "select processor_id from processors order by processor_id";
				rs = smt.executeQuery(sql);
				while(rs.next()){
					int processor_id = rs.getInt("processor_id");
					String sql2 = "update processors set processor_id = "+id+" where processor_id = "+processor_id;
					smt2.executeUpdate(sql2);
					sql2 = "update processor_schedules set processor_id = "+id+" where processor_id = "+processor_id;
					smt2.executeUpdate(sql2);
					sql2 = "update processor_qualification set processor_id = "+id+" where processor_id = "+processor_id;
					smt2.executeUpdate(sql2);
					sql2 = "update processor_properties set processor_id = "+id+" where processor_id = "+processor_id;
					smt2.executeUpdate(sql2);
					id++;
				}
			}
		}catch(SQLException e) {
			System.out.println(e);
		}finally{
			if(rs!=null){
				rs.close();
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
