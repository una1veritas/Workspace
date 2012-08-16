package AttendServerNew;
import java.util.*;
import java.sql.*;

public class JDBCConnection {    
  public Connection connection;
  public Statement statement;

  private String Schema = "ATTEND";
  private String SchemaPassword = "ATTEND";
  private String DBURL = "jdbc:oracle:thin:@131.206.103.230:1521:orc1";
    
  public JDBCConnection( ) throws SQLException {
    DriverManager.registerDriver(new oracle.jdbc.driver.OracleDriver());
    connection = DriverManager.getConnection(DBURL, Schema, SchemaPassword);
    statement = connection.createStatement();  
  }
            
  public synchronized String executeAttendQuery(String query) throws Exception {
    StringBuffer sbuf = new StringBuffer();
    String col = null;
    
    if (statement == null) {
      System.err.println("There is no database to execute the query.");
      return null;
    }
    
    ResultSet resultSet = statement.executeQuery(query);    
    if (resultSet != null) {
      int totalCols = resultSet.getMetaData().getColumnCount();
      while (resultSet.next()) {
	for (int i = 1; i <= totalCols; i++) {
	  col = resultSet.getString(i);
	  if (col == null) {
	    sbuf.append(" |");
	  } else {
	    if (col.trim().length() == 0) {
	      sbuf.append(" |");
	    } else { 
	      sbuf.append(col.trim()).append("|");
	    }
	  }
	}
	sbuf.append("$");
      } 
      resultSet.close();
    }
    if (sbuf == null) {
      return null;
    } else {
      if (sbuf.length() != 0) {
	return sbuf.toString();
      } else {
	return null;
      }
    }
  }
    
  public synchronized int executeAttendUpdate(String updateMessage) {
    if (statement == null) {
      System.err.println("There is no database to execute the query.");
      return -1;
    }
    try {
      return statement.executeUpdate(updateMessage);
    } catch (SQLException ex) {
      //	    ex.printStackTrace();
      return 0;
    }
  }
    
  public synchronized int executeAttendErrorInsert(String param) {
    if (statement == null) {
      System.err.println("There is no database to execute the query.");
      return -1;
    }
    try {
      StringTokenizer stk = new StringTokenizer(param, "|");
      String err = stk.nextToken();
      String id = stk.nextToken();
      String msg = "insert into CARD_READER_ERROR (ERROR_MSG, CARD_READER, ERROR_DATE) values ('"+err+"', '"+id+"', sysdate)";
      return statement.executeUpdate(msg);
    } catch (Exception ex) {
      //	    ex.printStackTrace();
      return 0;
    }
  }
        
  public void close() throws SQLException {
    System.out.println("Closing db connection");
    statement.close();
    connection.close();
  }
  
  protected void finalize() throws Throwable {
    close();
  }
}
