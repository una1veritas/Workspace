import java.sql.*;
import java.util.*;

public class DBConnectionPool {
  private String driver; // ドライバクラス
  private String url; // URL
  private String user; // ユーザー名
  private String pass; // パスワード

  private int maxConnection; // 最大接続数
  private int checkedOut; // 貸し出している接続数
  private Vector connectionPool = new Vector();

  /**
   * コンストラクタ
   */
  public DBConnectionPool(String url,String user,String pass) {
    this.driver = "com.mysql.jdbc.Driver";
    this.url = url;
    this.user = "yukiko";
    this.pass = "nishimura";
    this.maxConnection = 10;
  }
  /**
   * コネクションを取得
   */
  public synchronized Connection getConnection() throws Exception {
    Connection con = null;
    if (connectionPool.size() > 0) {
      con = (Connection) connectionPool.firstElement();
      connectionPool.removeElementAt(0);
      try {
        if (con.isClosed()) {
          con = getConnection();
        }
      } catch (SQLException e) {
        con = getConnection();
      }
    } else if (maxConnection == 0 || checkedOut < maxConnection) {
      con = newConnection();
    }
    if (con != null) {
      checkedOut++;
    }
    return con;
  }
  /**
   * 新規にコネクションを作成
   */
  private Connection newConnection() throws Exception {
    Class.forName(driver);
    return DriverManager.getConnection(url, user, pass);
  }
  /**
   * コネクションを返却
   */
  public synchronized void freeConnection(Connection con) {
    connectionPool.addElement(con);
    checkedOut--;
    notifyAll();
  }
  /**
   * すべてのコネクションを開放
   */
  public synchronized void release() {
    Enumeration enumConnections = connectionPool.elements();
    while (enumConnections.hasMoreElements()) {
      Connection con = (Connection)enumConnections.nextElement();
      try {
          con.close();
      } catch (SQLException e) {
      }
    }
    connectionPool.removeAllElements();
  }
}