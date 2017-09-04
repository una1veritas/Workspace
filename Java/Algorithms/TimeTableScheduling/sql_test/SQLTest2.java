package sql_test;
import java.sql.CallableStatement;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

import org.apache.commons.lang.time.StopWatch;

import common.ConnectionManager;

/*
 * 作成日: 2007/05/24
 *
 * TODO この生成されたファイルのテンプレートを変更するには次へジャンプ:
 * ウィンドウ - 設定 - Java - コード・スタイル - コード・テンプレート
 */

/**
 * @author 浦島太郎
 *
 * TODO この生成された型コメントのテンプレートを変更するには次へジャンプ:
 * ウィンドウ - 設定 - Java - コード・スタイル - コード・テンプレート
 */
public class SQLTest2 {
	public static void main(String args[]) throws Exception{
		ConnectionManager cmanager = new ConnectionManager("jdbc:mysql://localhost/schedule?useUnicode=true&characterEncoding=sjis","yukiko","nishimura");
		StopWatch sp = new StopWatch();
		Connection con = null;
		CallableStatement smt = null;
		int penalties = 0;
		int mode = 1;
		try{
			//MySQLサーバ接続
			con = cmanager.getConnection();
			//Statementインターフェースの生成
			smt = con.prepareCall("call getPenalties4()");
			sp.start();
			for(int n = 0; n < 100; n++){
				if(mode==1){
					smt.execute();
				}else if(mode==2){
				}else{
				}
			}
			sp.stop();
			System.out.println(sp);
		}catch(SQLException e) {
			System.out.println(e);
		}catch(Exception e){
			e.printStackTrace();
		}finally{
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
