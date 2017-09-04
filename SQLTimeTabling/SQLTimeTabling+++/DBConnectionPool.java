import java.sql.*;
import java.util.*;

public class DBConnectionPool {
  private String driver; // �h���C�o�N���X
  private String url; // URL
  private String user; // ���[�U�[��
  private String pass; // �p�X���[�h

  private int maxConnection; // �ő�ڑ���
  private int checkedOut; // �݂��o���Ă���ڑ���
  private Vector connectionPool = new Vector();

  /**
   * �R���X�g���N�^
   */
  public DBConnectionPool(String url,String user,String pass) {
    this.driver = "com.mysql.jdbc.Driver";
    this.url = url;
    this.user = "yukiko";
    this.pass = "nishimura";
    this.maxConnection = 10;
  }
  /**
   * �R�l�N�V�������擾
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
   * �V�K�ɃR�l�N�V�������쐬
   */
  private Connection newConnection() throws Exception {
    Class.forName(driver);
    return DriverManager.getConnection(url, user, pass);
  }
  /**
   * �R�l�N�V������ԋp
   */
  public synchronized void freeConnection(Connection con) {
    connectionPool.addElement(con);
    checkedOut--;
    notifyAll();
  }
  /**
   * ���ׂẴR�l�N�V�������J��
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