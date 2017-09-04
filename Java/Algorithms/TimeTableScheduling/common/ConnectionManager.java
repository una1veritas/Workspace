package common;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;


/**
 * コネクション管理クラス
 */
public class ConnectionManager {
	
	/**
	 * JDBCドライバーのクラス名
	 */
	private String driver;
	/**
	 * データベースのURL
	 */
	private String url;
	/**
	 * データベースのユーザ
	 */
	private String user;
	/**
	 * データベースのパスワード
	 */
	private String pass;
	
	public ConnectionManager(String url,String user,String pass){
		driver = "com.mysql.jdbc.Driver";
		this.url = url;
		this.user = user;
		this.pass = pass;
	}
	/**
	 * Conncectionを取得します。
	 */	
	public Connection getConnection() throws SQLException{
		try {
			Class.forName(driver).newInstance();
		} catch (Exception e) {
			e.printStackTrace();
		}
		Connection con = DriverManager.getConnection(url,user,pass);
		return con;
	}
}
