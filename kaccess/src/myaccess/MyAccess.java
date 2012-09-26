package myaccess;

import java.io.InputStream;
import java.sql.*;

public class MyAccess {
	transient Connection dbconn;
	transient Statement statem;
	transient PreparedStatement prepared;
	transient ResultSet results;
	transient int updates;
	transient int issuedQueries;
	
	
	public MyAccess() {
		dbconn = null;
	}
	
	public void connect(String url, String user, String pass) throws Exception {
		if ( dbconn != null && (!dbconn.isClosed()) ) // これ以外は何かのエラー
			return;
		try {
			Class.forName("com.mysql.jdbc.Driver"); 
			dbconn = DriverManager.getConnection(url, user, pass);
		} catch (SQLException e) {
			System.err.println("Couldn't get connection: " + e);
			throw e;
		}
		statem = dbconn.createStatement();
		
		issuedQueries = 0;
		return;
	}

	/**
	 * すべてのコネクションを開放
	 */
	public synchronized void disconnect() throws SQLException {
		if (statem!=null) {
			statem.close();
		}
		if (prepared!=null) {
			prepared.close();
		}
		// The ResultSet instances associated w/ the statements will be closed by the above.
		if (dbconn!=null) {
			dbconn.close();
		}
		System.err.println("Queries: "+issuedQueries);
	}
	
	public ResultSet results() {
		return results;
	}
	
	public void closeResults() throws SQLException {
		if (results != null)
			results.close();
		return;
	}

	public boolean execute(String sql) throws SQLException  {
		boolean qtype;
		try {
			qtype = statem.execute(sql);
		} catch (SQLException e) {
			System.err.println("SQL: "+sql);
			throw e;
		}
		if (qtype) {
			results = statem.getResultSet();
		}
		issuedQueries++;
		return qtype;
	}
	
	
	public ResultSet executeQuery(String sql) throws SQLException  {
		try {
			results = statem.executeQuery(sql);
		} catch (SQLException e) {
			System.err.println("SQL: "+sql);
			throw e;
		}
		issuedQueries++;
		return results;
	}
	
	
	public int executeUpdate(String sql) throws SQLException  {
		try {
			updates = statem.executeUpdate(sql);
		} catch (SQLException e) {
			System.err.println("SQL: "+sql);
			throw e;
		}
		issuedQueries++;
		return updates;
	}
	
	
	public void setPreparedStatement(String sql) throws SQLException  {
		prepared = dbconn.prepareStatement(sql);
	}
	
	public void setStatementValue(int pos, int val) throws SQLException {
		prepared.setInt(pos, val);
	}
	
	public void setStatementValue(int pos, String val) throws SQLException {
		prepared.setString(pos, val);
	}
	
	public void setStatementValue(int pos, Timestamp val) throws SQLException {
		prepared.setTimestamp(pos, val);
	}
	
	public void setStatementValue(int pos, InputStream stream, int size) throws SQLException {
		prepared.setBinaryStream(pos, stream, size ); //-(3)
}	
	
	public void setStatementValue(int pos, byte b[]) throws SQLException {
		prepared.setBytes(pos, b);
	}

	//
	public int executePreparedUpdate() throws Exception  {
		updates = prepared.executeUpdate();
		
		issuedQueries++;
		
		return updates;
	}
	
	public ResultSet executePreparedQuery() throws Exception  {
		results = prepared.executeQuery();
		
		issuedQueries++;
		
		return results;
	}

}
