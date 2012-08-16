package common;
import java.util.*;
import java.sql.*;

public class JDBCConnectionBase {
  public CommonInfoBase  commonInfo;

  public Connection kyomuConnection;
  public Statement kyomuStatement; 

  public Connection attendConnection;
  public Statement attendStatement; 

  public String sysdate;
  public int thisSchoolYear;
  public int thisSemester;

  
  public JDBCConnectionBase(CommonInfoBase commonInfo) throws SQLException {
    this.commonInfo = commonInfo;
    DriverManager.registerDriver(new oracle.jdbc.driver.OracleDriver());
    
    kyomuConnection = DriverManager.getConnection(commonInfo.kyomuDBURL,
						  commonInfo.kyomuDBSchema,
						  commonInfo.kyomuDBPasswd);
    kyomuStatement = kyomuConnection.createStatement();
    
    attendConnection = DriverManager.getConnection(commonInfo.attendDBURL,
						   commonInfo.attendDBSchema,
						   commonInfo.attendDBPasswd);
    attendStatement = attendConnection.createStatement();

    setSysdate();  
  }

  public void setSysdate() {
    try {
      String quest = "select to_char(sysdate, 'YYYY-MM-DD'), to_char(sysdate, 'YYYY'), to_char(sysdate, 'MM') from dual"; 
      ResultSet resultSet = attendStatement.executeQuery(quest);
      resultSet.next();    
      sysdate = resultSet.getString(1);
      String year = resultSet.getString(2);
      String month = resultSet.getString(3);
      resultSet.close();      
      int thisYear = Integer.parseInt(year);
      int thisMonth = Integer.parseInt(month);    
      if (thisMonth <= 3) {
	thisSchoolYear = thisYear - 1;
	thisSemester = 2;
      } else {
	thisSchoolYear = thisYear;
	if (thisMonth >= 10) {
	  thisSemester = 2;
	} else {
	  thisSemester = 1;
	}
      }  
    } catch (Exception e) { }
  }
      
  public synchronized String executeKyomuQuery(String query) throws Exception {
    StringBuilder sbuf = new StringBuilder();
    String col = null;
	
    if (kyomuConnection == null || kyomuStatement == null) {
      System.err.println("There is no database to execute the query.");
      return null;
    }
	
    ResultSet resultSet = kyomuStatement.executeQuery(query); 
    if (resultSet != null) {
      int totalCols = resultSet.getMetaData().getColumnCount();
      while (resultSet.next()) {
	for (int i = 1; i <= totalCols; i++) {
	  if (resultSet.getString(i) == null) {
	    sbuf.append(" |");
	  } else {
	    col = resultSet.getString(i).trim();
	    if (col.length() == 0) {
	      sbuf.append(" |");
	    } else { 
	      sbuf.append(col).append("|");
	    }
	  }
	}
	sbuf.append("$");
      } 
      resultSet.close();
    }
    if (sbuf.length() != 0) {
      return sbuf.toString();
    } else {
      return null;
    }
  } 
      
  public synchronized String executeAttendQuery(String query) throws Exception {
    StringBuilder sbuf = new StringBuilder();
    String col = null;
	
    if (attendConnection == null || attendStatement == null) {
      System.err.println("There is no database to execute the query.");
      return null;
    }
	
    ResultSet resultSet = attendStatement.executeQuery(query);    
    if (resultSet != null) {
      int totalCols = resultSet.getMetaData().getColumnCount();
      while (resultSet.next()) {
	for (int i = 1; i <= totalCols; i++) {
	  if (resultSet.getString(i) == null) {
	    sbuf.append(" |");
	  } else {
	    col = resultSet.getString(i).trim();
	    if (col.length() == 0) {
	      sbuf.append(" |");
	    } else { 
	      sbuf.append(col).append("|");
	    }
	  }
	}
	sbuf.append("$");
      } 
      resultSet.close();
    }
    if (sbuf.length() != 0) {
      return sbuf.toString();
    } else {
      return null;
    }
  }
    
  public synchronized int executeKyomuUpdate(String update) throws Exception  {	
    if (kyomuConnection == null || kyomuStatement == null) {
      System.err.println("There is no database to execute the update.");
      return -1;
    }	
    return kyomuStatement.executeUpdate(update);
  }
    
  public synchronized int executeAttendUpdate(String update) throws Exception {	
    if (attendConnection == null || attendStatement == null) {
      System.err.println("There is no database to execute the update.");
      return -1;
    }	
    return attendStatement.executeUpdate(update);
  }

  
  public synchronized String getCommonQueryResult(String commandCode,
						  String paramValues) throws Exception 
  {
    String dbName = null;
    String paramList = null;
    String rawSQL = null;
    String querySQL = null;
    ResultSet resultSet = null;
    try {
      String quest = "select DB_NAME, PARAM_LIST, QUERY_SQL from STRUCT2.COMMON_QUERY_STRUCT where QUERY_NAME = '" + commandCode + "'";
      resultSet = attendStatement.executeQuery(quest);
      if (resultSet == null) {
	System.out.println("commandCode = " + commandCode + " is not found in STRUCT2.COMMON_QUERY_STRUCT.");
	return null;
      }
      resultSet.next();
      dbName = resultSet.getString(1);
      paramList = resultSet.getString(2);
      rawSQL = resultSet.getString(3);
      resultSet.close();
    } catch (Exception e) {
      e.printStackTrace();
      throw e;
    }
    
    if (rawSQL == null) return null;
    if (paramValues == null) {
      querySQL = rawSQL;
    } else {
      querySQL = matchParams(rawSQL, paramValues);
      if (querySQL == null) {
	Exception e = new NumberFormatException("MatchingError: COMMON_QUERY_STRUCT commandCode = " + commandCode);
	throw e;
      }
    }
 
    StringBuilder sbuf = new StringBuilder();
    if (dbName.equals("ATTEND")) {
      resultSet = attendStatement.executeQuery(querySQL); 
    } else {
      resultSet = kyomuStatement.executeQuery(querySQL); 
    }
    if (resultSet != null) {
      int totalCols = resultSet.getMetaData().getColumnCount();
      while (resultSet.next()) {
	for (int i = 1; i <= totalCols; i++) {
	  if (resultSet.getString(i) == null) {
	    sbuf.append(" |");
	  } else {
	    String col = resultSet.getString(i).trim();
	    if (col.length() == 0) {
	      sbuf.append(" |");
	    } else { 
	      sbuf.append(col).append("|");
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
      if (sbuf.length() == 0) {
	return null;
      } else {
	return sbuf.toString();
      } 
    }
  } 


  public synchronized String getCommonQueryResultByQueryName(String queryName,
							     String paramValues) throws Exception 
  {
    String dbName = null;
    String paramList = null;
    String rawSQL = null;
    String querySQL = null;
    ResultSet resultSet = null;
    try {
      String quest = "select DB_NAME, PARAM_LIST, QUERY_SQL from STRUCT2.COMMON_QUERY_STRUCT where QUERY_NAME = '" + queryName + "'";
      resultSet = attendStatement.executeQuery(quest);
      if (resultSet == null) {
	System.out.println("queryName = " + queryName + " is not found in STRUCT2.COMMON_QUERY_STRUCT.");
	return null;
      }
      resultSet.next();
      dbName = resultSet.getString(1);
      paramList = resultSet.getString(2);
      rawSQL = resultSet.getString(3);
      resultSet.close();
    } catch (Exception e) {
      e.printStackTrace();
      throw e;
    }
    
    if (rawSQL == null) return null;
    if (paramValues == null) {
      querySQL = rawSQL;
    } else {
      querySQL = matchParams(rawSQL, paramValues);
      if (querySQL == null) {
	Exception e = new NumberFormatException("MatchingError: COMMON_QUERY_STRUCT queryName = " + queryName);
	throw e;
      }
    }
 
    StringBuilder sbuf = new StringBuilder();
    if (dbName.equals("ATTEND")) {
      resultSet = attendStatement.executeQuery(querySQL); 
    } else {
      resultSet = kyomuStatement.executeQuery(querySQL); 
    }
    if (resultSet != null) {
      int totalCols = resultSet.getMetaData().getColumnCount();
      while (resultSet.next()) {
	for (int i = 1; i <= totalCols; i++) {
	  if (resultSet.getString(i) == null) {
	    sbuf.append(" |");
	  } else {
	    String col = resultSet.getString(i).trim();
	    if (col.length() == 0) {
	      sbuf.append(" |");
	    } else { 
	      sbuf.append(col).append("|");
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
      if (sbuf.length() == 0) {
	return null;
      } else {
	return sbuf.toString();
      } 
    }
  } 

         
  public synchronized String getGakusekiInfo(String studentCode) throws Exception {
    StringBuffer sbuf = new StringBuffer();
    String col;
    String querySql = "select * from MASTER.GAKUSEKI where STUDENT_CODE = '" + studentCode + "'";
    ResultSet resultSet = kyomuStatement.executeQuery(querySql); 
    if (resultSet != null) {
      ResultSetMetaData meta = resultSet.getMetaData();
      int totalCols = meta.getColumnCount();
      while (resultSet.next()) {
	for (int i = 1; i <= totalCols; i++) {
	  sbuf.append(meta.getColumnName(i)).append("|");
	  if (resultSet.getString(i) == null) {
	    sbuf.append(" |$");
	  } else {
	    col = resultSet.getString(i).trim();
	    if (col.length() == 0) {
	      sbuf.append(" |$");
	    } else { 
	      sbuf.append(col).append("|$");
	    }
	  }
	}
      } 
      resultSet.close();
    }  
    if (sbuf == null) {
      return null;
    } else {
      if (sbuf.length() == 0) {
	return null;
      } else {
	return sbuf.toString();
      } 
    }
  }
 
  public synchronized String getQueryResult(String serviceName,
					    String panelID,
					    String switchFlag,
					    String paramValues,
					    Person me) throws Exception 
  {
    ResultSet resultSet;
    StringBuilder sbuf = new StringBuilder(); 
    String querySQL = null;
    String rawSQL = null;
    String dbName = null;
    String qualificationMethod = null;
    String qualificationMethodParam = null;

    try {
      String quest = "select DB_NAME, QUERY_SQL, QUALIFICATION_METHOD, QUALIFICATION_METHOD_PARAM from STRUCT2.SERVER_QUERY_STRUCT where SERVICE_NAME = '" + serviceName + "' and PANEL_ID = '" + panelID + "' and SWITCH_FLAG = '" + switchFlag + "'";
      resultSet = attendStatement.executeQuery(quest);
      if (resultSet == null) {
	System.out.println("serviceName = " + serviceName + ", panelID = " + panelID + ", switchFlag = " + switchFlag + " is not found in SERVER_QUERY_STRUCT.");
	return null;
      }
      resultSet.next();
      dbName = resultSet.getString(1);
      rawSQL = resultSet.getString(2);
      qualificationMethod = resultSet.getString(3);
      qualificationMethodParam = resultSet.getString(4);
      resultSet.close();
    } catch (Exception e) {
      e.printStackTrace();
      throw e;
    }
	
    if (rawSQL == null) return null;
	
    if (qualificationMethod != null) {
      RetValue ret = new RetValue();
      try {
	getClass().getMethod(qualificationMethod, new Class[] { String.class, String.class, Person.class, RetValue.class }).invoke(this, new Object[] { qualificationMethodParam, paramValues, me, ret } );
      } catch (Exception ex) {
	System.out.println("qualificationMethod = " + qualificationMethod + " cannot be invoked.");
	throw ex;
      }
      if (!ret.getValue()) {
	return null;
      }
    }
	
    if (paramValues != null) {
      querySQL = matchParams(rawSQL, paramValues);
      if (querySQL == null) {
	Exception e = new NumberFormatException("MatchingError: serviceName = " + serviceName + ", panelID = " + panelID + ", switchFlag = " + switchFlag);
	throw e;
      }
    } else {
      querySQL = rawSQL;
    }

    resultSet = null;
    if (dbName.equals("KYOMU")) {
      resultSet = kyomuStatement.executeQuery(querySQL); 
    } else if (dbName.equals("ATTEND")) {
      resultSet = attendStatement.executeQuery(querySQL); 
    }

    if (resultSet != null) {
      int totalCols = resultSet.getMetaData().getColumnCount();
      while (resultSet.next()) {
	for (int i = 1; i <= totalCols; i++) {
	  String col = resultSet.getString(i);
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


  public synchronized String getSingleStructQuery(String serviceName,
						  String panelID,
						  String switchFlag,
						  String paramValues) throws Exception 
  {
    ResultSet resultSet;
    String dbName, paramList, querySQL, rawSQL;
    StringBuffer sbuf = new StringBuffer();

    try {
      String quest = "select DB_NAME, PARAM_LIST, QUERY_SQL from STRUCT2.SERVER_QUERY_STRUCT where SERVICE_NAME = '" + serviceName + "' and PANEL_ID = '" + panelID + "' and SWITCH_FLAG = '" + switchFlag + "'";
      resultSet = attendStatement.executeQuery(quest);
      if (resultSet == null) {
	System.out.println("serviceName = " + serviceName + ", panelID = " + panelID + ", switchFlag = " + switchFlag + " is not found in SERVER_QUERY_STRUCT.");
	return null;
      }
      resultSet.next();
      dbName = resultSet.getString(1);
      paramList = resultSet.getString(2);
      rawSQL = resultSet.getString(3);
      resultSet.close();
    } catch (Exception e) {
      e.printStackTrace();
      throw e;
    }    
    String[] pkeys = paramList.split(":");
    HashSet<String> pkSet = new HashSet<String>();
    for (String pk : pkeys) {
      pkSet.add(pk);
    }
    
    if (rawSQL == null) return null;	
    if (paramValues != null) {
      querySQL = matchParams(rawSQL, paramValues);
      if (querySQL == null) {
	Exception e = new NumberFormatException("MatchingError: serviceName = " + serviceName + ", panelID = " + panelID + ", switchFlag = " + switchFlag);
	throw e;
      }
    } else {
      querySQL = rawSQL;
    }

    resultSet = attendStatement.executeQuery(querySQL);     
    if (resultSet != null) {
      ResultSetMetaData meta = resultSet.getMetaData();
      int totalCols = meta.getColumnCount();
      while (resultSet.next()) {
	for (int i = 1; i <= totalCols; i++) {
	  String colName = meta.getColumnName(i);
	  String colVal = resultSet.getString(i);
	  if (colVal == null) {
	    colVal = " ";
	  } else {
	    colVal = colVal.trim();
	    if (colVal.length() == 0) {
	      colVal = " ";
	    }
	  }	  
	  if (pkSet.contains(colName)) {
	    sbuf.append(colName).append("|").append(colVal).append("|0|$");
	  } else {
	    sbuf.append(colName).append("|").append(colVal).append("|1|$");
	  }
	}
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
    
  public String matchParams(String rawSQL, String paramValues) {
    if (rawSQL == null) return null;

    ArrayList<String> paramList = new ArrayList<String>();
    String[] tokens = paramValues.split("\\|");
    for (String token : tokens) {
      paramList.add(token);
    }

    StringBuilder sbuf = new StringBuilder();
    String tail = rawSQL;
    try {
      while (true) {
	int index = tail.indexOf("?");
	if (index == -1) break;
	sbuf.append(tail.substring(0, index));
	String s = String.valueOf(tail.charAt(index+1));
	if (s.equals("?")) {
	  s = String.valueOf(tail.charAt(index+2));
	  int n = Integer.parseInt(s, 16);
	  String val = paramList.get(n);
	  sbuf.append(val);
	  tail = tail.substring(index+3);	  
	} else {
	  int n = Integer.parseInt(s, 16);
	  String val = paramList.get(n);
	  if (val.equals(" ")) {
	    sbuf.append("null");
	  } else if (val.equals("null")) {
	    sbuf.append("null");
	  } else {
	    sbuf.append("'").append(val).append("'");
	  }
	  tail = tail.substring(index+2);
	}
      }
      sbuf.append(tail);
      return sbuf.toString();
    } catch (Exception e) {      
      return null;
    }
  }   
 
  public synchronized int deleteCommand(String serviceName,
					String deleteCode,
					String paramValues,
					Person me) throws Exception 
  {
    ResultSet resultSet;
    StringBuilder sbuf = new StringBuilder(); 
    String deleteSQL = null;
    String sideEffectSQL = null;
    String rawDeleteSQL = null;
    String rawSideEffect = null;
    String dbName = null;
    String qualificationMethod = null;
    String qualificationMethodParam = null;

    try {
      String quest = "select DB_NAME, DELETE_SQL, SIDE_EFFECT, QUALIFICATION_METHOD, QUALIFICATION_METHOD_PARAM from STRUCT2.SERVER_DELETE_STRUCT where SERVICE_NAME = '" + serviceName + "' and DELETE_CODE = '" + deleteCode + "'";
      resultSet = attendStatement.executeQuery(quest);
      if (resultSet == null) {
	System.out.println("serviceName = " + serviceName + ", deleteCode = " + deleteCode + " is not found in SERVER_DELETE_STRUCT.");
	return 0;
      }
      resultSet.next();
      dbName = resultSet.getString(1);
      rawDeleteSQL = resultSet.getString(2);
      rawSideEffect = resultSet.getString(3);
      qualificationMethod = resultSet.getString(4);
      qualificationMethodParam = resultSet.getString(5);
      resultSet.close();
    } catch (Exception e) {
      e.printStackTrace();
      throw e;
    }
	
    if (rawDeleteSQL == null) return 0;
	
    if (qualificationMethod != null) {
      RetValue ret = new RetValue();
      try {
	getClass().getMethod(qualificationMethod, new Class[] { String.class, String.class, Person.class, RetValue.class }).invoke(this, new Object[] { qualificationMethodParam, paramValues, me, ret } );
      } catch (Exception ex) {
	System.out.println("qualificationMethod = " + qualificationMethod + " cannot be invoked.");
	throw ex;
      }
      if (!ret.getValue()) {
	return 0;
      }
    }
	
    if (paramValues != null) {
      deleteSQL  = matchParams(rawDeleteSQL, paramValues);
      sideEffectSQL = matchParams(rawSideEffect, paramValues);
      if (deleteSQL == null) {
	Exception e = new NumberFormatException("MatchingError: serviceName = " + serviceName + ", deleteCode = " + deleteCode);
	throw e;
      }
    } else {
      deleteSQL = rawDeleteSQL;
      sideEffectSQL = rawSideEffect;
    }

    int ans = 0;
    if (dbName.equals("KYOMU")) {
      ans = executeKyomuUpdate(deleteSQL);
      if (sideEffectSQL != null) {
	executeKyomuUpdate(sideEffectSQL);
      }
    } else if (dbName.equals("ATTEND")) {
      ans = executeAttendUpdate(deleteSQL);
      if (sideEffectSQL != null) {
	executeAttendUpdate(sideEffectSQL);
      }
    }
    return ans;
  }

  public synchronized int updateCommand(String serviceName,
					String updateCode,
					String paramValues,
					Person me) throws Exception 
  {
    ResultSet resultSet;
    StringBuilder sbuf = new StringBuilder(); 
    String updateSQL = null;
    String sideEffectSQL = null;
    String rawUpdateSQL = null;
    String rawSideEffect = null;
    String dbName = null;
    String qualificationMethod = null;
    String qualificationMethodParam = null;

    try {
      String quest = "select DB_NAME, UPDATE_SQL, SIDE_EFFECT, QUALIFICATION_METHOD, QUALIFICATION_METHOD_PARAM from STRUCT2.SERVER_UPDATE_STRUCT where SERVICE_NAME = '" + serviceName + "' and UPDATE_CODE = '" + updateCode + "'";
      resultSet = attendStatement.executeQuery(quest);
      if (resultSet == null) {
	System.out.println("serviceName = " + serviceName + ", updateCode = " + updateCode + " is not found in SERVER_UPDATE_STRUCT.");
	return 0;
      }
      resultSet.next();
      dbName = resultSet.getString(1);
      rawUpdateSQL = resultSet.getString(2);
      rawSideEffect = resultSet.getString(3);
      qualificationMethod = resultSet.getString(4);
      qualificationMethodParam = resultSet.getString(5);
      resultSet.close();
    } catch (Exception e) {
      e.printStackTrace();
      throw e;
    }
	
    if (rawUpdateSQL == null) return 0;
	
    if (qualificationMethod != null) {
      RetValue ret = new RetValue();
      try {
	getClass().getMethod(qualificationMethod, new Class[] { String.class, String.class, Person.class, RetValue.class }).invoke(this, new Object[] { qualificationMethodParam, paramValues, me, ret } );
      } catch (Exception ex) {
	System.out.println("qualificationMethod = " + qualificationMethod + " cannot be invoked.");
	throw ex;
      }
      if (!ret.getValue()) {
	return 0;
      }
    }
	
    if (paramValues != null) {
      updateSQL  = matchParams(rawUpdateSQL, paramValues);
      sideEffectSQL = matchParams(rawSideEffect, paramValues);
      if (updateSQL == null) {
	Exception e = new NumberFormatException("MatchingError: serviceName = " + serviceName + ", updateCode = " + updateCode);
	throw e;
      }
    } else {
      updateSQL = rawUpdateSQL;
      sideEffectSQL = rawSideEffect;
    }

    int ans = 0;
    if (dbName.equals("KYOMU")) {
      ans = executeKyomuUpdate(updateSQL);
      if (sideEffectSQL != null) {
	executeKyomuUpdate(sideEffectSQL);
      }
    } else if (dbName.equals("ATTEND")) {
      ans = executeAttendUpdate(updateSQL);
      if (sideEffectSQL != null) {
	executeAttendUpdate(sideEffectSQL);
      }
    }
    return ans;
  }

  public synchronized int insertCommand(String serviceName,
					String insertCode,
					String paramValues,
					Person me) throws Exception 
  {
    ResultSet resultSet;
    StringBuilder sbuf = new StringBuilder(); 
    String insertSQL = null;
    String sideEffectSQL = null;
    String rawInsertSQL = null;
    String rawSideEffect = null;
    String dbName = null;
    String qualificationMethod = null;
    String qualificationMethodParam = null;

    try {
      String quest = "select DB_NAME, INSERT_SQL, SIDE_EFFECT, QUALIFICATION_METHOD, QUALIFICATION_METHOD_PARAM from STRUCT2.SERVER_INSERT_STRUCT where SERVICE_NAME = '" + serviceName + "' and INSERT_CODE = '" + insertCode + "'";
      resultSet = attendStatement.executeQuery(quest);
      if (resultSet == null) {
	System.out.println("serviceName = " + serviceName + ", insertCode = " + insertCode + " is not found in SERVER_INSERT_STRUCT.");
	return 0;
      }
      resultSet.next();
      dbName = resultSet.getString(1);
      rawInsertSQL = resultSet.getString(2);
      rawSideEffect = resultSet.getString(3);
      qualificationMethod = resultSet.getString(4);
      qualificationMethodParam = resultSet.getString(5);
      resultSet.close();
    } catch (Exception e) {
      e.printStackTrace();
      throw e;
    }
	
    if (rawInsertSQL == null) return 0;
	
    if (qualificationMethod != null) {
      RetValue ret = new RetValue();
      try {
	getClass().getMethod(qualificationMethod, new Class[] { String.class, String.class, Person.class, RetValue.class }).invoke(this, new Object[] { qualificationMethodParam, paramValues, me, ret } );
      } catch (Exception ex) {
	System.out.println("qualificationMethod = " + qualificationMethod + " cannot be invoked.");
	throw ex;
      }
      if (!ret.getValue()) {
	return 0;
      }
    }
	
    if (paramValues != null) {
      insertSQL  = matchParams(rawInsertSQL, paramValues);
      sideEffectSQL = matchParams(rawSideEffect, paramValues);
      if (insertSQL == null) {
	Exception e = new NumberFormatException("MatchingError: serviceName = " + serviceName + ", insertCode = " + insertCode);
	throw e;
      }
    } else {
      insertSQL = rawInsertSQL;
      sideEffectSQL = rawSideEffect;
    }

    int ans = 0;
    if (dbName.equals("KYOMU")) {
      ans = executeKyomuUpdate(insertSQL);
      if (sideEffectSQL != null) {
	executeKyomuUpdate(sideEffectSQL);
      }
    } else if (dbName.equals("ATTEND")) {
      ans = executeAttendUpdate(insertSQL);
      if (sideEffectSQL != null) {
	executeAttendUpdate(sideEffectSQL);
      }
    }
    return ans;
  }

  public synchronized int specialCommand(String serviceName,
					 String specialCode,
					 String paramValues,
					 Person me) throws Exception 
  {
    ResultSet resultSet;
    StringBuilder sbuf = new StringBuilder(); 
    String dbName = null;
    String qualificationMethod = null;
    String qualificationMethodParam = null;

    try {
      String quest = "select DB_NAME, QUALIFICATION_METHOD, QUALIFICATION_METHOD_PARAM from STRUCT2.SERVER_SPECIAL_STRUCT where SERVICE_NAME = '" + serviceName + "' and SPECIAL_CODE = '" + specialCode + "'";
      resultSet = attendStatement.executeQuery(quest);
      if (resultSet == null) {
	System.out.println("serviceName = " + serviceName + ", specialCode = " + specialCode + " is not found in SERVER_SPECIAL_STRUCT.");
	return 0;
      }
      resultSet.next();
      dbName = resultSet.getString(1);
      qualificationMethod = resultSet.getString(2);
      qualificationMethodParam = resultSet.getString(3);
      resultSet.close();
    } catch (Exception e) {
      e.printStackTrace();
      throw e;
    }
	
    if (qualificationMethod != null) {
      RetValue ret = new RetValue();
      try {
	getClass().getMethod(qualificationMethod, new Class[] { String.class, String.class, Person.class, RetValue.class }).invoke(this, new Object[] { qualificationMethodParam, paramValues, me, ret } );
      } catch (Exception ex) {
	System.out.println("qualificationMethod = " + qualificationMethod + " cannot be invoked.");
	throw ex;
      }
      if (!ret.getValue()) {
	return 0;
      }
    }
	
    int ans = 0;
    if (dbName.equals("KYOMU")) { 
      if (serviceName.equals("UserControl")) {
	if (specialCode.equals("400")) {
	  StringTokenizer stk = new StringTokenizer(paramValues, "|");
	  String studentCode = stk.nextToken();  

	  String quest = "select INIT_PASSWORD from STUDENT.STUDENT_PASSWD where STUDENT_CODE = '"+studentCode+"'";
	  resultSet = kyomuStatement.executeQuery(quest);
	  if (resultSet != null) {
	    resultSet.next();    
	    String INIT_PASSWORD = resultSet.getString(1);
	    resultSet.close();
	    String SALT = commonInfo.getSalt();
	    String cpasswd = commonInfo.crypt(INIT_PASSWORD, SALT);  
	    String update = "update STUDENT.STUDENT_PASSWD set PASSWORD = '"+cpasswd+"', PASSWORD_STATUS = '0', REVISED_DATE = sysdate where STUDENT_CODE = '" + studentCode + "'";
	    return executeKyomuUpdate(update); 
	  } else {
	    return 0;
	  }	  
	} else if (specialCode.equals("401")) {
	  StringTokenizer stk = new StringTokenizer(paramValues, "|");
	  String studentCode = stk.nextToken();  
	  String update = "update STUDENT.STUDENT_PASSWD set PASSWORD = '*', PASSWORD_STATUS = '9', REVISED_DATE = sysdate where STUDENT_CODE = '" + studentCode + "'";
	  return executeKyomuUpdate(update); 
	} else if (specialCode.equals("411")) {
	  StringTokenizer stk = new StringTokenizer(paramValues, "|");
	  String staffID = stk.nextToken();  
	  String update = "update TEACHER.STAFF_PASSWD set PASSWORD = '*', PASSWORD_STATUS = '9', REVISED_DATE = sysdate where STAFF_ID = '" + staffID + "'";
	  return executeKyomuUpdate(update); 
	} 
      } else if (serviceName.equals("NameControl")) {
	if (specialCode.equals("400")) {
	  StringTokenizer stk = new StringTokenizer(paramValues, "|");
	  String staffCode = stk.nextToken();  
	  String update = "update MASTER.STAFF set SHORTER_NAME = STAFF_NAME where STAFF_CODE = '" + staffCode + "'";
	  return executeKyomuUpdate(update); 
	} else if (specialCode.equals("401")) {
	  StringTokenizer stk = new StringTokenizer(paramValues, "|");
	  String studentCode = stk.nextToken();  
	  String update = "update MASTER.GAKUSEKI set SHORTER_NAME = STUDENT_NAME where STUDENT_CODE = '" + studentCode + "'";
	  return executeKyomuUpdate(update); 
	} else if (specialCode.equals("402")) {
	  StringTokenizer stk = new StringTokenizer(paramValues, "|");
	  String subjectCode = stk.nextToken();  
	  String update = "update MASTER.SUBJECT set SHORTER_NAME = SUBJECT_NAME where SUBJECT_CODE = '" + subjectCode + "'";
	  return executeKyomuUpdate(update); 
	}
      }

    } else if (dbName.equals("ATTEND")) {   
      if (serviceName.equals("AttendTool")) {

	StringTokenizer stk = new StringTokenizer(paramValues, "|");
	String SCHOOL_YEAR = stk.nextToken();
	String SEMESTER = stk.nextToken();
	String SUBJECT_CODE = stk.nextToken();
	String CLASS_CODE = stk.nextToken();
	String TEACHER_CODE = stk.nextToken();
	String STUDENT_CODE = stk.nextToken();
	String ATTEND_FLAG = stk.nextToken();
	String READ_FLAG = stk.nextToken();
	String CLASS_DATE = stk.nextToken();
	String HOUR = stk.nextToken();
	String ROOM = stk.nextToken();
	String READ_DATE_TIME = stk.nextToken();
	String ATTEND_START_DATE = stk.nextToken();
	String LATE_START_DATE = stk.nextToken();
	String LATE_END_DATE = stk.nextToken();   
 
	String staffName = me.STAFF_NAME;
	String readDateTime = CLASS_DATE+":"+READ_DATE_TIME;

	String remark1 = "出席データの追加 ... " + sysdate + " ("+staffName+")"; 
	String remark2 = "入力時刻の変更   ... " + sysdate + " ("+staffName+")"; 
	String remark4 = "出席データ無効化 ... " + sysdate + " ("+staffName+")"; 

	// READ_FLAG = 0:  (学生による)カードリーダ入力
	// READ_FLAG = 1:  (教官による)追加データ
	// READ_FLAG = 2:  (教官による)修正データ    ... オリジナルのデータはカードリーダ入力
	// READ_FLAG = 3:  (教官による)無効化データ  ... オリジナルのデータはカードリーダ入力

	String attendDate, ins, insList, upd, updList, del, delList;
	int res1 = 0;
	int res2 = 0;

	if (READ_FLAG.equals("1")) {  
	  // この場合、ATTEND_FLAG が 欠 であることはないことになっている。
	  // 教官が入力した出席データの場合、その出席データは削除しても
 	  // かまわない。

	  del = "delete from ATTEND.CARD_READER_DATA where SCHOOL_YEAR = '"+SCHOOL_YEAR+"' and SEMESTER = '"+SEMESTER+"' and STUDENT_CODE = '"+STUDENT_CODE+"' and ATTEND_DATE = to_date('"+CLASS_DATE+"','YYYY:MM:DD') and READ_DATE_TIME = to_date('"+readDateTime+"','YYYY:MM:DD:HH24:MI:SS') and READ_FLAG = '1'";
	  res1 = executeAttendUpdate(del);
	  delList = "delete from ATTEND.ATTEND_LIST where SCHOOL_YEAR = '"+SCHOOL_YEAR+"' and SEMESTER = '"+SEMESTER+"' and SUBJECT_CODE = '"+SUBJECT_CODE+"' and CLASS_CODE = '"+CLASS_CODE+"' and STUDENT_CODE ='"+STUDENT_CODE+"' and CLASS_DATE = to_date('"+CLASS_DATE+"','YYYY:MM:DD') and HOUR = '"+HOUR+"' and READ_FLAG = '1'";
	  executeAttendUpdate(delList);

	  if ((specialCode.equals("405")) || (specialCode.equals("408")) || (specialCode.equals("411"))) {  // →欠
	    return res1;

	  } else if ((specialCode.equals("406")) || (specialCode.equals("409"))) {  // →◯
 
	    attendDate = CLASS_DATE+":"+ATTEND_START_DATE+":01";	  
	    ins = "insert into ATTEND.CARD_READER_DATA (SCHOOL_YEAR, SEMESTER, STUDENT_CODE, CARD_READER, ATTEND_DATE, READ_DATE_TIME, READ_FLAG, REMARK) select '"+SCHOOL_YEAR+"','"+SEMESTER+"','"+STUDENT_CODE+"',min(CR.CARD_READER),to_date('"+CLASS_DATE+"','YYYY:MM:DD'),to_date('"+attendDate+"','YYYY:MM:DD:HH24:MI:SS'),'1','"+remark1+"' from ATTEND.CARD_READER_INFO CR where ROOM = '"+ROOM+"'";
	    res1 = executeAttendUpdate(ins);
	    insList = "insert into ATTEND.ATTEND_LIST (SCHOOL_YEAR, SEMESTER, SUBJECT_CODE, CLASS_CODE, TEACHER_CODE, STUDENT_CODE, CLASS_DATE, HOUR, READ_DATE_TIME, ATTEND_FLAG, READ_FLAG, REMARK ) values ('"+SCHOOL_YEAR+"','"+SEMESTER+"','"+SUBJECT_CODE+"','"+CLASS_CODE+"','"+TEACHER_CODE+"','"+STUDENT_CODE+"',to_date('"+CLASS_DATE+"','YYYY:MM:DD'),'"+HOUR+"',to_date('"+attendDate+"','YYYY:MM:DD:HH24:MI:SS'),'1','1','"+remark1+"')";
	    res2 = executeAttendUpdate(insList);
	    return res1;

	  } else if ((specialCode.equals("403")) || (specialCode.equals("410"))) {  // →△ 
 
	    attendDate = CLASS_DATE+":"+LATE_START_DATE+":01";	  
	    ins = "insert into ATTEND.CARD_READER_DATA (SCHOOL_YEAR, SEMESTER, STUDENT_CODE, CARD_READER, ATTEND_DATE, READ_DATE_TIME, READ_FLAG, REMARK) select '"+SCHOOL_YEAR+"','"+SEMESTER+"','"+STUDENT_CODE+"',min(CR.CARD_READER),to_date('"+CLASS_DATE+"','YYYY:MM:DD'),to_date('"+attendDate+"','YYYY:MM:DD:HH24:MI:SS'),'1','"+remark1+"' from ATTEND.CARD_READER_INFO CR where ROOM = '"+ROOM+"'";
	    res1 = executeAttendUpdate(ins);
	    insList = "insert into ATTEND.ATTEND_LIST (SCHOOL_YEAR, SEMESTER, SUBJECT_CODE, CLASS_CODE, TEACHER_CODE, STUDENT_CODE, CLASS_DATE, HOUR, READ_DATE_TIME, ATTEND_FLAG, READ_FLAG, REMARK ) values ('"+SCHOOL_YEAR+"','"+SEMESTER+"','"+SUBJECT_CODE+"','"+CLASS_CODE+"','"+TEACHER_CODE+"','"+STUDENT_CODE+"',to_date('"+CLASS_DATE+"','YYYY:MM:DD'),'"+HOUR+"',to_date('"+attendDate+"','YYYY:MM:DD:HH24:MI:SS'),'2','1','"+remark1+"')";
	    res2 = executeAttendUpdate(insList);
	    return res1;
	    
	  } else if ((specialCode.equals("404")) || (specialCode.equals("407"))) {  // → ▲
 
	    attendDate = CLASS_DATE+":"+LATE_END_DATE+":01";	  
	    ins = "insert into ATTEND.CARD_READER_DATA (SCHOOL_YEAR, SEMESTER, STUDENT_CODE, CARD_READER, ATTEND_DATE, READ_DATE_TIME, READ_FLAG, REMARK) select '"+SCHOOL_YEAR+"','"+SEMESTER+"','"+STUDENT_CODE+"',min(CR.CARD_READER),to_date('"+CLASS_DATE+"','YYYY:MM:DD'),to_date('"+attendDate+"','YYYY:MM:DD:HH24:MI:SS'),'1','"+remark1+"' from ATTEND.CARD_READER_INFO CR where ROOM = '"+ROOM+"'";
	    res1 = executeAttendUpdate(ins);
	    insList = "insert into ATTEND.ATTEND_LIST (SCHOOL_YEAR, SEMESTER, SUBJECT_CODE, CLASS_CODE, TEACHER_CODE, STUDENT_CODE, CLASS_DATE, HOUR, READ_DATE_TIME, ATTEND_FLAG, READ_FLAG, REMARK ) values ('"+SCHOOL_YEAR+"','"+SEMESTER+"','"+SUBJECT_CODE+"','"+CLASS_CODE+"','"+TEACHER_CODE+"','"+STUDENT_CODE+"',to_date('"+CLASS_DATE+"','YYYY:MM:DD'),'"+HOUR+"',to_date('"+attendDate+"','YYYY:MM:DD:HH24:MI:SS'),'3','1','"+remark1+"')";
	    res2 = executeAttendUpdate(insList);
	    return res1;

	  } 

	} else {

	  // 学生が入力した出席データの場合、その出席データに変更が加えられることはあっても、
 	  // 削除をしてはならない。

	  if (specialCode.equals("400")) {  // 欠→◯
	    if (!ATTEND_FLAG.equals(" ")) return 0;

	    attendDate = CLASS_DATE+":"+ATTEND_START_DATE+":01";	  
	    ins = "insert into ATTEND.CARD_READER_DATA (SCHOOL_YEAR, SEMESTER, STUDENT_CODE, CARD_READER, ATTEND_DATE, READ_DATE_TIME, READ_FLAG, REMARK) select '"+SCHOOL_YEAR+"','"+SEMESTER+"','"+STUDENT_CODE+"',min(CR.CARD_READER),to_date('"+CLASS_DATE+"','YYYY:MM:DD'),to_date('"+attendDate+"','YYYY:MM:DD:HH24:MI:SS'),'1','"+remark1+"' from ATTEND.CARD_READER_INFO CR where ROOM = '"+ROOM+"'";
	    res1 = executeAttendUpdate(ins);
	    insList = "insert into ATTEND.ATTEND_LIST (SCHOOL_YEAR, SEMESTER, SUBJECT_CODE, CLASS_CODE, TEACHER_CODE, STUDENT_CODE, CLASS_DATE, HOUR, READ_DATE_TIME, ATTEND_FLAG, READ_FLAG, REMARK ) values ('"+SCHOOL_YEAR+"','"+SEMESTER+"','"+SUBJECT_CODE+"','"+CLASS_CODE+"','"+TEACHER_CODE+"','"+STUDENT_CODE+"',to_date('"+CLASS_DATE+"','YYYY:MM:DD'),'"+HOUR+"',to_date('"+attendDate+"','YYYY:MM:DD:HH24:MI:SS'),'1','1','"+remark1+"')";
	    res2 = executeAttendUpdate(insList);
	    return res1;

	  } else if (specialCode.equals("401")) {  // 欠→△
	    if (!ATTEND_FLAG.equals(" ")) return 0;

	    attendDate = CLASS_DATE+":"+LATE_START_DATE+":01";	  
	    ins = "insert into ATTEND.CARD_READER_DATA (SCHOOL_YEAR, SEMESTER, STUDENT_CODE, CARD_READER, ATTEND_DATE, READ_DATE_TIME, READ_FLAG, REMARK) select '"+SCHOOL_YEAR+"','"+SEMESTER+"','"+STUDENT_CODE+"',min(CR.CARD_READER),to_date('"+CLASS_DATE+"','YYYY:MM:DD'),to_date('"+attendDate+"','YYYY:MM:DD:HH24:MI:SS'),'1','"+remark1+"' from ATTEND.CARD_READER_INFO CR where ROOM = '"+ROOM+"'";
	    res1 = executeAttendUpdate(ins);
	    insList = "insert into ATTEND.ATTEND_LIST (SCHOOL_YEAR, SEMESTER, SUBJECT_CODE, CLASS_CODE, TEACHER_CODE, STUDENT_CODE, CLASS_DATE, HOUR, READ_DATE_TIME, ATTEND_FLAG, READ_FLAG, REMARK ) values ('"+SCHOOL_YEAR+"','"+SEMESTER+"','"+SUBJECT_CODE+"','"+CLASS_CODE+"','"+TEACHER_CODE+"','"+STUDENT_CODE+"',to_date('"+CLASS_DATE+"','YYYY:MM:DD'),'"+HOUR+"',to_date('"+attendDate+"','YYYY:MM:DD:HH24:MI:SS'),'2','1','"+remark1+"')";
	    res2 = executeAttendUpdate(insList);
	    return res1;
	    
	  } else if (specialCode.equals("402")) {  // 欠→▲
	    if (!ATTEND_FLAG.equals(" ")) return 0;

	    attendDate = CLASS_DATE+":"+LATE_END_DATE+":01";	  
	    ins = "insert into ATTEND.CARD_READER_DATA (SCHOOL_YEAR, SEMESTER, STUDENT_CODE, CARD_READER, ATTEND_DATE, READ_DATE_TIME, READ_FLAG, REMARK) select '"+SCHOOL_YEAR+"','"+SEMESTER+"','"+STUDENT_CODE+"',min(CR.CARD_READER),to_date('"+CLASS_DATE+"','YYYY:MM:DD'),to_date('"+attendDate+"','YYYY:MM:DD:HH24:MI:SS'),'1','"+remark1+"' from ATTEND.CARD_READER_INFO CR where ROOM = '"+ROOM+"'";
	    res1 = executeAttendUpdate(ins);
	    insList = "insert into ATTEND.ATTEND_LIST (SCHOOL_YEAR, SEMESTER, SUBJECT_CODE, CLASS_CODE, TEACHER_CODE, STUDENT_CODE, CLASS_DATE, HOUR, READ_DATE_TIME, ATTEND_FLAG, READ_FLAG, REMARK ) values ('"+SCHOOL_YEAR+"','"+SEMESTER+"','"+SUBJECT_CODE+"','"+CLASS_CODE+"','"+TEACHER_CODE+"','"+STUDENT_CODE+"',to_date('"+CLASS_DATE+"','YYYY:MM:DD'),'"+HOUR+"',to_date('"+attendDate+"','YYYY:MM:DD:HH24:MI:SS'),'3','1','"+remark1+"')";
	    res2 = executeAttendUpdate(insList);
	    return res1;

	  } else if (specialCode.equals("403")) {  // ◯→△
	    if (!ATTEND_FLAG.equals("1")) return 0;

	    attendDate = CLASS_DATE+":"+LATE_START_DATE+":01";		
	  upd = "update ATTEND.CARD_READER_DATA set READ_DATE_TIME = to_date('"+attendDate+"','YYYY:MM:DD:HH24:MI:SS'), READ_FLAG = '2', REMARK = '"+remark2+"' where SCHOOL_YEAR = '"+SCHOOL_YEAR+"' and SEMESTER = '"+SEMESTER+"' and STUDENT_CODE = '"+STUDENT_CODE+"' and READ_DATE_TIME = to_date('"+readDateTime+"','YYYY:MM:DD:HH24:MI:SS')";
	    res1 = executeAttendUpdate(upd);
	    updList = "update ATTEND.ATTEND_LIST set ATTEND_FLAG = '2', READ_FLAG = '2', REMARK = '"+remark2+"', READ_DATE_TIME = to_date('"+attendDate+"','YYYY:MM:DD:HH24:MI:SS') where SCHOOL_YEAR = '"+SCHOOL_YEAR+"' and SEMESTER = '"+SEMESTER+"' and SUBJECT_CODE = '"+SUBJECT_CODE+"' and CLASS_CODE = '"+CLASS_CODE+"' and STUDENT_CODE ='"+STUDENT_CODE+"' and CLASS_DATE = to_date('"+CLASS_DATE+"','YYYY:MM:DD') and HOUR = '"+HOUR+"'";	
	    res2 = executeAttendUpdate(updList);
	    return res1; 

	  } else if (specialCode.equals("404")) {  // ◯→▲
	    if (!ATTEND_FLAG.equals("1")) return 0;

	    attendDate = CLASS_DATE+":"+LATE_END_DATE+":01";		
	    upd = "update ATTEND.CARD_READER_DATA set READ_DATE_TIME = to_date('"+attendDate+"','YYYY:MM:DD:HH24:MI:SS'), READ_FLAG = '2', REMARK = '"+remark2+"' where SCHOOL_YEAR = '"+SCHOOL_YEAR+"' and SEMESTER = '"+SEMESTER+"' and STUDENT_CODE = '"+STUDENT_CODE+"' and READ_DATE_TIME = to_date('"+readDateTime+"','YYYY:MM:DD:HH24:MI:SS')";
	    res1 = executeAttendUpdate(upd);
	    updList = "update ATTEND.ATTEND_LIST set ATTEND_FLAG = '3', READ_FLAG = '2', REMARK = '"+remark2+"', READ_DATE_TIME = to_date('"+attendDate+"','YYYY:MM:DD:HH24:MI:SS') where SCHOOL_YEAR = '"+SCHOOL_YEAR+"' and SEMESTER = '"+SEMESTER+"' and SUBJECT_CODE = '"+SUBJECT_CODE+"' and CLASS_CODE = '"+CLASS_CODE+"' and STUDENT_CODE ='"+STUDENT_CODE+"' and CLASS_DATE = to_date('"+CLASS_DATE+"','YYYY:MM:DD') and HOUR = '"+HOUR+"'";	
	    res2 = executeAttendUpdate(updList);
	    return res1; 

	  } else if (specialCode.equals("405")) {  // ◯→欠
	    if (!ATTEND_FLAG.equals("1")) return 0;

	    upd = "update ATTEND.CARD_READER_DATA set READ_FLAG = '3', REMARK = '"+remark4+"' where SCHOOL_YEAR = '"+SCHOOL_YEAR+"' and SEMESTER = '"+SEMESTER+"' and STUDENT_CODE = '"+STUDENT_CODE+"' and READ_DATE_TIME = to_date('"+readDateTime+"','YYYY:MM:DD:HH24:MI:SS')";
	    res1 = executeAttendUpdate(upd);
	    updList = "delete from ATTEND.ATTEND_LIST where SCHOOL_YEAR = '"+SCHOOL_YEAR+"' and SEMESTER = '"+SEMESTER+"' and SUBJECT_CODE = '"+SUBJECT_CODE+"' and CLASS_CODE = '"+CLASS_CODE+"' and STUDENT_CODE ='"+STUDENT_CODE+"' and CLASS_DATE = to_date('"+CLASS_DATE+"','YYYY:MM:DD') and HOUR = '"+HOUR+"'";
	    res2 = executeAttendUpdate(updList);
	    return res1;

	  } else if (specialCode.equals("406")) {  // △→◯
	    if (!ATTEND_FLAG.equals("2")) return 0;

	    attendDate = CLASS_DATE+":"+ATTEND_START_DATE+":01";	
	    upd = "update ATTEND.CARD_READER_DATA set READ_DATE_TIME = to_date('"+attendDate+"','YYYY:MM:DD:HH24:MI:SS'), READ_FLAG = '2', REMARK = '"+remark2+"' where SCHOOL_YEAR = '"+SCHOOL_YEAR+"' and SEMESTER = '"+SEMESTER+"' and STUDENT_CODE = '"+STUDENT_CODE+"' and READ_DATE_TIME = to_date('"+readDateTime+"','YYYY:MM:DD:HH24:MI:SS')";
	    res1 = executeAttendUpdate(upd);
	    updList = "update ATTEND.ATTEND_LIST set ATTEND_FLAG = '1', READ_FLAG = '2', REMARK = '"+remark2+"', READ_DATE_TIME = to_date('"+attendDate+"','YYYY:MM:DD:HH24:MI:SS') where SCHOOL_YEAR = '"+SCHOOL_YEAR+"' and SEMESTER = '"+SEMESTER+"' and SUBJECT_CODE = '"+SUBJECT_CODE+"' and CLASS_CODE = '"+CLASS_CODE+"' and STUDENT_CODE ='"+STUDENT_CODE+"' and CLASS_DATE = to_date('"+CLASS_DATE+"','YYYY:MM:DD') and HOUR = '"+HOUR+"'";	
	    res2 = executeAttendUpdate(updList);
	    return res1;	  
	    
	  } else if (specialCode.equals("407")) {  // △→▲
	    if (!ATTEND_FLAG.equals("2")) return 0;

	    attendDate = CLASS_DATE+":"+LATE_END_DATE+":01";	
	    upd = "update ATTEND.CARD_READER_DATA set READ_DATE_TIME = to_date('"+attendDate+"','YYYY:MM:DD:HH24:MI:SS'), READ_FLAG = '2', REMARK = '"+remark2+"' where SCHOOL_YEAR = '"+SCHOOL_YEAR+"' and SEMESTER = '"+SEMESTER+"' and STUDENT_CODE = '"+STUDENT_CODE+"' and READ_DATE_TIME = to_date('"+readDateTime+"','YYYY:MM:DD:HH24:MI:SS')";
	    res1 = executeAttendUpdate(upd);
	    updList = "update ATTEND.ATTEND_LIST set ATTEND_FLAG = '3', READ_FLAG = '2', REMARK = '"+remark2+"', READ_DATE_TIME = to_date('"+attendDate+"','YYYY:MM:DD:HH24:MI:SS') where SCHOOL_YEAR = '"+SCHOOL_YEAR+"' and SEMESTER = '"+SEMESTER+"' and SUBJECT_CODE = '"+SUBJECT_CODE+"' and CLASS_CODE = '"+CLASS_CODE+"' and STUDENT_CODE ='"+STUDENT_CODE+"' and CLASS_DATE = to_date('"+CLASS_DATE+"','YYYY:MM:DD') and HOUR = '"+HOUR+"'";	
	    res2 = executeAttendUpdate(updList);
	    return res1;	  

	  } else if (specialCode.equals("408")) {  // △→欠
	    if (!ATTEND_FLAG.equals("2")) return 0;

	    upd = "update ATTEND.CARD_READER_DATA set READ_FLAG = '3', REMARK = '"+remark4+"' where SCHOOL_YEAR = '"+SCHOOL_YEAR+"' and SEMESTER = '"+SEMESTER+"' and STUDENT_CODE = '"+STUDENT_CODE+"' and READ_DATE_TIME = to_date('"+readDateTime+"','YYYY:MM:DD:HH24:MI:SS')";
	    res1 = executeAttendUpdate(upd);
	    updList = "delete from ATTEND.ATTEND_LIST where SCHOOL_YEAR = '"+SCHOOL_YEAR+"' and SEMESTER = '"+SEMESTER+"' and SUBJECT_CODE = '"+SUBJECT_CODE+"' and CLASS_CODE = '"+CLASS_CODE+"' and STUDENT_CODE ='"+STUDENT_CODE+"' and CLASS_DATE = to_date('"+CLASS_DATE+"','YYYY:MM:DD') and HOUR = '"+HOUR+"'";
	    res2 = executeAttendUpdate(updList);
	    return res1;
	  
	  } else if (specialCode.equals("409")) {  // ▲→◯
	    if (!ATTEND_FLAG.equals("3")) return 0;

	    attendDate = CLASS_DATE+":"+ATTEND_START_DATE+":01";	
	    upd = "update ATTEND.CARD_READER_DATA set READ_DATE_TIME = to_date('"+attendDate+"','YYYY:MM:DD:HH24:MI:SS'), READ_FLAG = '2', REMARK = '"+remark2+"' where SCHOOL_YEAR = '"+SCHOOL_YEAR+"' and SEMESTER = '"+SEMESTER+"' and STUDENT_CODE = '"+STUDENT_CODE+"' and READ_DATE_TIME = to_date('"+readDateTime+"','YYYY:MM:DD:HH24:MI:SS')";
	    res1 = executeAttendUpdate(upd);
	    updList = "update ATTEND.ATTEND_LIST set ATTEND_FLAG = '1', READ_FLAG = '2', REMARK = '"+remark2+"', READ_DATE_TIME = to_date('"+attendDate+"','YYYY:MM:DD:HH24:MI:SS') where SCHOOL_YEAR = '"+SCHOOL_YEAR+"' and SEMESTER = '"+SEMESTER+"' and SUBJECT_CODE = '"+SUBJECT_CODE+"' and CLASS_CODE = '"+CLASS_CODE+"' and STUDENT_CODE ='"+STUDENT_CODE+"' and CLASS_DATE = to_date('"+CLASS_DATE+"','YYYY:MM:DD') and HOUR = '"+HOUR+"'";	
	    res2 = executeAttendUpdate(updList);
	    return res1;	  	  

	  } else if (specialCode.equals("410")) {  // ▲→△
	    if (!ATTEND_FLAG.equals("3")) return 0;

	    attendDate = CLASS_DATE+":"+LATE_START_DATE+":01";		
	    upd = "update ATTEND.CARD_READER_DATA set READ_DATE_TIME = to_date('"+attendDate+"','YYYY:MM:DD:HH24:MI:SS'), READ_FLAG = '2', REMARK = '"+remark2+"' where SCHOOL_YEAR = '"+SCHOOL_YEAR+"' and SEMESTER = '"+SEMESTER+"' and STUDENT_CODE = '"+STUDENT_CODE+"' and READ_DATE_TIME = to_date('"+readDateTime+"','YYYY:MM:DD:HH24:MI:SS')";
	    res1 = executeAttendUpdate(upd);
	    updList = "update ATTEND.ATTEND_LIST set ATTEND_FLAG = '2', READ_FLAG = '2', REMARK = '"+remark2+"', READ_DATE_TIME = to_date('"+attendDate+"','YYYY:MM:DD:HH24:MI:SS') where SCHOOL_YEAR = '"+SCHOOL_YEAR+"' and SEMESTER = '"+SEMESTER+"' and SUBJECT_CODE = '"+SUBJECT_CODE+"' and CLASS_CODE = '"+CLASS_CODE+"' and STUDENT_CODE ='"+STUDENT_CODE+"' and CLASS_DATE = to_date('"+CLASS_DATE+"','YYYY:MM:DD') and HOUR = '"+HOUR+"'";	
	    res2 = executeAttendUpdate(updList);
	    return res1;
	  	  
	  } else if (specialCode.equals("411")) {  // ▲→欠
	    if (!ATTEND_FLAG.equals("3")) return 0;
	  
	    upd = "update ATTEND.CARD_READER_DATA set READ_FLAG = '3', REMARK = '"+remark4+"' where SCHOOL_YEAR = '"+SCHOOL_YEAR+"' and SEMESTER = '"+SEMESTER+"' and STUDENT_CODE = '"+STUDENT_CODE+"' and READ_DATE_TIME = to_date('"+readDateTime+"','YYYY:MM:DD:HH24:MI:SS')";
	    res1 = executeAttendUpdate(upd);
	    updList = "delete from ATTEND.ATTEND_LIST where SCHOOL_YEAR = '"+SCHOOL_YEAR+"' and SEMESTER = '"+SEMESTER+"' and SUBJECT_CODE = '"+SUBJECT_CODE+"' and CLASS_CODE = '"+CLASS_CODE+"' and STUDENT_CODE ='"+STUDENT_CODE+"' and CLASS_DATE = to_date('"+CLASS_DATE+"','YYYY:MM:DD') and HOUR = '"+HOUR+"'";
	    res2 = executeAttendUpdate(updList);
	    return res1; 
	  }
	}
      }
    }
    return ans;
  }


  public void checkQualification(String qualificationMethodParam, 
				 String paramList, Person me, RetValue ret) {
    // paramList の先頭要素が STUDENT_CODE である場合について
    // その学生の教務情報へのアクセス権限に関する審査を行う。
    // 
    // QUALIFICATION >= 4    ==> accept
    // QUALIFICATION <= 1    ==> reject
    // 2 <= QUALIFICATION <= 3  
    //   教員 me が当該学生の指導教官、または、その教員が学生の所属学科の学務委員補佐 ==> accept
    //   それ以外の場合    ==> reject

    StringTokenizer stk = new StringTokenizer(paramList, "|");
    String STUDENT_CODE = stk.nextToken();

    if (STUDENT_CODE.equals(me.STUDENT_CODE)) {
      ret.accept();
      return;
    }

    int qual;
    String SUPERVISOR, FACULTY, DEPARTMENT, COURSE;
    try {
      qual = Integer.parseInt(me.QUALIFICATION);
    } catch (Exception e) {
      qual = 0;
    }
    if ((qual >= 4) && (qual != 7)) {
      ret.accept();
      return;
    }
    if ((qual <= 1) || (qual == 7)) {
      ret.reject(); 
      return;
    }
    if (paramList == null) {
      ret.reject(); 
      return;
    } else {
      String STAFF_CODE   = me.STAFF_CODE;
      String STAFF_ATTRIB = me.STAFF_ATTRIB;
      try {
	String query = "select SUPERVISOR, FACULTY, DEPARTMENT, COURSE from MASTER.STUDENT where STUDENT_CODE = '"+STUDENT_CODE+"'";
	ResultSet resultSet = attendStatement.executeQuery(query);  
	if (resultSet == null)  {
	  ret.reject(); 
	  return;
	}
	resultSet.next();
	SUPERVISOR = resultSet.getString(1);
	FACULTY    = resultSet.getString(2).trim();
	DEPARTMENT = resultSet.getString(3).trim();
	COURSE     = resultSet.getString(4).trim();
	resultSet.close();
	
	if (SUPERVISOR == null) {
	  SUPERVISOR = " ";
	} else if (SUPERVISOR.trim().equals(STAFF_CODE)) {
	  ret.accept(); 
	  return;
	}
	if (isGakumuHosa(qual, FACULTY, DEPARTMENT, COURSE, STAFF_ATTRIB)) {
	  ret.accept(); 
	  return;
	}	

	// 次のメソッドは「学科別に自学科の学生の成績情報の閲覧を許すという設定
	// を実現するためのメソッドである。

	if (isQualifiedDepartment(qual, FACULTY, DEPARTMENT, COURSE, STAFF_ATTRIB)) {
	  ret.accept();
	  return;
	}
      } catch (Exception e) {
	e.printStackTrace();
      }
    }
    ret.reject(); 
  }

  public void checkQualification2(String qualificationMethodParam, 
				  String paramList, Person me, RetValue ret) {
    // paramList の先頭要素が  FACULTY:DEPARTMENT である場合について、
    // (通常はその学科で開講義された科目の成績やその集計データへの)
    // アクセス権限に関する審査を行う。
    // 
    // QUALIFICATION >= 4    ==> accept
    // QUALIFICATION <= 1    ==> reject
    // 2 <= QUALIFICATION <= 3  
    //   FACULTY が情報工学部である場合には
    //     ユーザ me その学科に所属している場合  ==> accept
    //     それ以外の場合                        ==> reject
    //   FACULTY が大学院情報工学研究科である場合には 
    //   (教官の大学院での所属が不明であるので)
    //                                           ==> accept

    if (me.STUDENT_CODE.length() == 8) {
      ret.accept();
      return;
    }

    int qual;
    String FACULTY, DEPARTMENT;
    try {
      qual = Integer.parseInt(me.QUALIFICATION);
    } catch (Exception e) {
      qual = 0;
    }
    if ((qual >= 4) && (qual != 7)) {
      ret.accept();
      return;
    }
    if ((qual <= 1) || (qual == 7)) {
      ret.reject(); 
      return;
    }
    if (paramList == null) {
      ret.reject(); 
      return;
    } else {
      StringTokenizer stk = new StringTokenizer(paramList, "|");
      FACULTY    = stk.nextToken();
      DEPARTMENT = stk.nextToken();
      String MY_DEPT = getMyDepartment(me);
      
      if (FACULTY.equals("11")) {
	if (DEPARTMENT.equals(MY_DEPT)) {
	  ret.accept();
	  return;
	} else {
	  ret.reject(); 
	  return;	
	}
      } else {
	ret.accept();
	return;
      }
    }
  }
  
  public void checkQualification3(String qualificationMethodParam, 
				  String paramList, Person me, RetValue ret) {
    // paramList の先頭要素が STUDENT_CODE である場合について
    // その学生の教務情報へのアクセス権限に関する審査を行う。
    // 
    // QUALIFICATION >= 4    ==> accept
    // QUALIFICATION <= 1    ==> reject
    // 2 <= QUALIFICATION <= 3  
    //   教員 me が当該学生の指導教官であるか、または、その学生
    //   が教員の所属学科の学生である場合    ==> accept
    //   それ以外の場合                      ==> reject

    StringTokenizer stk = new StringTokenizer(paramList, "|");
    String STUDENT_CODE = stk.nextToken();
    if (STUDENT_CODE.equals(me.STUDENT_CODE)) {
      ret.accept();
      return;
    }

    int qual;
    try {
      qual = Integer.parseInt(me.QUALIFICATION);
    } catch (Exception e) {
      qual = 0;
    }
    if ((qual >= 4) && (qual != 7)) {
      ret.accept();
      return;
    }
    if ((qual <= 1) || (qual == 7)) {
      ret.reject(); 
      return;
    }
    if (paramList == null) {
      ret.reject(); 
      return;
    } else {
      String MY_DEPT = getMyDepartment(me);
      String STAFF_CODE = me.STAFF_CODE;

      String SUPERVISOR, FACULTY, DEPARTMENT;
      try {
	String query = "select SUPERVISOR, FACULTY, DEPARTMENT from MASTER.STUDENT where STUDENT_CODE = '"+STUDENT_CODE+"'";
	ResultSet resultSet = attendStatement.executeQuery(query);  
	if (resultSet == null)  {
	  ret.reject(); 
	  return;
	}
	resultSet.next();
	SUPERVISOR = resultSet.getString(1);
	FACULTY    = resultSet.getString(2).trim();
	DEPARTMENT = resultSet.getString(3).trim();
	resultSet.close();
	
	if (SUPERVISOR == null) {
	  SUPERVISOR = " ";
	} else if (SUPERVISOR.trim().equals(STAFF_CODE)) {
	  ret.accept(); 
	  return;
	}
	if (DEPARTMENT.equals(MY_DEPT)) {
	  ret.accept(); 
	  return;
	}	
      } catch (Exception e) {
	e.printStackTrace();
      }
    }
    ret.reject(); 
  }

  
  public void checkQualification4(String qualificationMethodParam, 
				  String paramList, Person me, RetValue ret) {
    // paramList の先頭要素が TEACHER_CODE である場合について
    // その教務情報へのアクセス権限に関する審査を行う。
    // 
    // QUALIFICATION >= 4    ==> accept
    // QUALIFICATION <= 1    ==> reject
    // 2 <= QUALIFICATION <= 3  
    //   教員 me が先頭要素の TEACHER_CODE 本人である場合    ==> accept
    //   それ以外の場合                      ==> reject

    int qual;
    try {
      qual = Integer.parseInt(me.QUALIFICATION);
    } catch (Exception e) {
      qual = 0;
    }
    if ((qual >= 4) && (qual != 7)) {
      ret.accept();
      return;
    }
    if ((qual <= 1) || (qual == 7)) {
      ret.reject(); 
      return;
    }
    if (paramList == null) {
      ret.reject(); 
      return;
    } else {
      StringTokenizer stk = new StringTokenizer(paramList, "|");
      String TEACHER_CODE = stk.nextToken();

      if (TEACHER_CODE.equals(me.STAFF_CODE)) {	
	ret.accept(); 
	return;
      }
    }
    ret.reject(); 
  }
  

  public String getMyDepartment(Person me) {
    int attrib = Integer.parseInt(me.STAFF_ATTRIB);    
    switch (attrib) {
    case 55: 
    case 205: 
      return "31";
    case 57: 
    case 210: 
      return "32";
    case 59:
    case 215: 
      return "33";
    case 61: 
    case 220: 
      return "34";
    case 63: 
    case 225: 
      return "35";
    case 65:  
    case 230: 
      return "30";
    }
    if (me.LOCAL_ATTRIB.equals("知能情報")) {
      return "31";
    } else if (me.LOCAL_ATTRIB.equals("電子情報")) {
      return "32";
    } else if (me.LOCAL_ATTRIB.equals("創成情報")) {
      return "33";
    } else if (me.LOCAL_ATTRIB.equals("機械情報")) {
      return "34";
    } else if (me.LOCAL_ATTRIB.equals("生命情報")) {
      return "35";
    } else if (me.LOCAL_ATTRIB.equals("人間科学")) {
      return "30";
    } else {
      return "" + attrib;
    }
  }

  public boolean isGakumuHosa(int qual, 
			      String FACULTY, String DEPARTMENT, String COURSE,
			      String STAFF_ATTRIB) {
    if (qual != 3) return false;
    int staffAttrib = Integer.parseInt(STAFF_ATTRIB);

    if (COURSE == null) {
      switch (staffAttrib) {
      case 55: 
      case 205: 
	if ((FACULTY.equals("11")) && (DEPARTMENT.equals("31")))  {
	  return true;
	} else {
	  return false;
	}
      case 57:
      case 210: 
	if ((FACULTY.equals("11")) && (DEPARTMENT.equals("32"))) {
	  return true;
	} else {
	  return false;
	}
      case 59:
      case 215: 
	if ((FACULTY.equals("11")) && (DEPARTMENT.equals("33"))) {
	  return true;
	} else {
	  return false;
	}
      case 61:
      case 220: 
	if ((FACULTY.equals("11")) && (DEPARTMENT.equals("34"))) {
	  return true;
	} else {
	  return false;
	}
      case 63:
      case 225: 
	if ((FACULTY.equals("11")) && (DEPARTMENT.equals("35"))) {
	  return true;
	} else {
	  return false;
	}
      case 65:
      case 230: 
	return false;
      case 81:
      case 235: 
	if (((FACULTY.equals("32")) && (DEPARTMENT.equals("75"))) ||
	    ((FACULTY.equals("51")) && (DEPARTMENT.equals("93")))) {
	  return true;
	} else {
	  return false;
	}
      default:
	return false;
      } 
    } else {
      switch (staffAttrib) {
      case 55: 
      case 205: 
	if (((FACULTY.equals("11")) && (DEPARTMENT.equals("31"))) ||
	    ((FACULTY.equals("32")) && (DEPARTMENT.equals("73")) && (COURSE.equals("85"))) ||
	    ((FACULTY.equals("51")) && (DEPARTMENT.equals("91")) && (COURSE.equals("85")))) {
	  return true;
	} else {
	  return false;
	}
      case 57:
      case 210: 
	if (((FACULTY.equals("11")) && (DEPARTMENT.equals("32"))) ||
	    ((FACULTY.equals("32")) && (DEPARTMENT.equals("74")) && (COURSE.equals("88"))) ||
	    ((FACULTY.equals("51")) && (DEPARTMENT.equals("92")) && (COURSE.equals("88")))) {
	  return true;
	} else {
	  return false;
	}
      case 59:
      case 215: 
	if (((FACULTY.equals("11")) && (DEPARTMENT.equals("33"))) ||
	    ((FACULTY.equals("32")) && (DEPARTMENT.equals("73")) && (COURSE.equals("86"))) ||
	    ((FACULTY.equals("51")) && (DEPARTMENT.equals("91")) && (COURSE.equals("86")))) {
	  return true;
	} else {
	  return false;
	}
      case 61:
      case 220: 
	if (((FACULTY.equals("11")) && (DEPARTMENT.equals("34"))) ||
	    ((FACULTY.equals("32")) && (DEPARTMENT.equals("74")) && (COURSE.equals("89"))) ||
	    ((FACULTY.equals("51")) && (DEPARTMENT.equals("92")) && (COURSE.equals("89")))) {
	  return true;
	} else {
	  return false;
	}
      case 63:
      case 225: 
	if (((FACULTY.equals("11")) && (DEPARTMENT.equals("35"))) ||
	    ((FACULTY.equals("32")) && (DEPARTMENT.equals("73")) && (COURSE.equals("87"))) ||
	    ((FACULTY.equals("51")) && (DEPARTMENT.equals("91")) && (COURSE.equals("87")))) {
	  return true;
	} else {
	  return false;
	}
      case 65:
      case 230: 
	return false;
      case 81:
      case 235: 
	if (((FACULTY.equals("32")) && (DEPARTMENT.equals("75"))) ||
	    ((FACULTY.equals("51")) && (DEPARTMENT.equals("93"))) ||
	    ((FACULTY.equals("32")) && (DEPARTMENT.equals("75")) && (COURSE.equals("90"))) ||
	    ((FACULTY.equals("51")) && (DEPARTMENT.equals("93")) && (COURSE.equals("90")))) {
	  return true;
	} else {
	  return false;
	}
      default:
	return false;
      }
    }
  }

  // 次のメソッドは、学科別の「自学科の学生の成績閲覧は許可する」という
  // 設定を実現するための手段であり、「審査基準」は学科別に設定されて
  // いるので、メンテナンスにあたっては注意する必要がある。
  public boolean isQualifiedDepartment(int qual, 
				       String FACULTY, String DEPARTMENT,
				       String COURSE,
				       String STAFF_ATTRIB) {
    if (qual != 2) return false;
    int staffAttrib = Integer.parseInt(STAFF_ATTRIB);

    switch (staffAttrib) {
    case 55: 
    case 205: 
      if ((FACULTY.equals("11")) && (DEPARTMENT.equals("31")))  {
	return true;
      } else {
	return false;
      }
    default:
      return false;
    }
  }
  
  public void close() throws SQLException {
    System.out.println("Closing db connection");
    kyomuStatement.close();
    kyomuConnection.close();
    attendStatement.close();
    attendConnection.close();
  }
    
  protected void finalize() throws Throwable {
    close();
  }
}
