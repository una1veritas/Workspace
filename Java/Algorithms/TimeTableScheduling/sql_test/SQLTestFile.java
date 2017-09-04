package sql_test;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.StringTokenizer;

import org.apache.commons.lang.time.StopWatch;

import common.ConnectionManager;

import data.ValuateSQL;


public class SQLTestFile {
	public static void main(String args[]){
		BufferedReader reader = null;
		ArrayList v_sqls = new ArrayList();
		ConnectionManager cmanager = new ConnectionManager("jdbc:mysql://localhost/schedule?useUnicode=true&characterEncoding=sjis","yukiko","nishimura");
		try {
			reader = new BufferedReader(new FileReader("t_valuate.txt"));
			String line;
			while ((line = reader.readLine()) != null) {
				ValuateSQL v_sql = null;
				StringTokenizer strToken = new StringTokenizer(line, ";");
				if(strToken.hasMoreTokens()) {
					v_sql = new ValuateSQL();
					v_sql.setType(Integer.parseInt(strToken.nextToken().toString()));
				}
				if(strToken.hasMoreTokens()) {
					v_sql.setSql(strToken.nextToken().toString());
				}
				if(v_sql!=null)v_sqls.add(v_sql);
			}
		} catch (FileNotFoundException e) {
			System.out.println("ファイルが見つかりません");
		} catch (IOException e) {
		} finally {
			try {
				if (reader != null) {
					reader.close();
				}
			} catch (Exception e) {
			}
		}
		StopWatch sp = new StopWatch();
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		int penalties = 0;
		try{
			//MySQLサーバ接続
			con = cmanager.getConnection();
			//Statementインターフェースの生成
			smt = con.createStatement();
			sp.start();
			for(int n = 0; n < 100; n++){
				Iterator i = v_sqls.iterator();
				while(i.hasNext()){
					ValuateSQL t = (ValuateSQL)i.next();
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
			}
		}catch(SQLException e) {
			e.printStackTrace();
		}catch(Exception e){
			e.printStackTrace();
		}finally{
			try{
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
					con = null;
				}
			}catch(Exception e){
				e.printStackTrace();
			}
		}
		sp.stop();
		System.out.println(sp);
		System.out.println(penalties);
	}
}
