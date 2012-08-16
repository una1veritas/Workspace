package Struct2ServerNew;
import common.*;
import java.util.*;
import java.sql.*;
import java.io.*;

public class JDBCConnection extends JDBCConnectionBase {
  
  public JDBCConnection(CommonInfo commonInfo) throws SQLException {
    super(commonInfo);
  }
  
  private String changeQuotationCode(String text) {
    StringBuffer sbuf = new StringBuffer();
    String tail = text;
    try {
      while (true) {
	int index = tail.indexOf("'");
	if (index == -1) break;
	sbuf.append(tail.substring(0, index+1)).append("'");
	tail = tail.substring(index+1);
      }
      sbuf.append(tail);
      return sbuf.toString();
    } catch (Exception e) {      
      return null;
    }
  }
  
  private String changeConcatCode2(String text) {
    StringBuffer sbuf = new StringBuffer();
    String tail = text;
    try {
      while (true) {
	int index = tail.indexOf("|");
	if (index == -1) break;
	sbuf.append(tail.substring(0, index)).append("&&");
	tail = tail.substring(index+2);
      }
      sbuf.append(tail);
      return sbuf.toString();
    } catch (Exception e) {      
      return null;
    }	
  }

  private String changeConcatCode(String text) {
    StringBuffer sbuf = new StringBuffer();
    String tail = text;
    try {
      while (true) {
	int index = tail.indexOf("&");
	if (index == -1) break;
	sbuf.append(tail.substring(0, index)).append("||");
	tail = tail.substring(index+2);
      }
      sbuf.append(tail);
      return sbuf.toString();
    } catch (Exception e) {      
      return null;
    }	
  }
  
  private boolean compareList(List<String> list1, List<String> list2) {
    if (list1 == null) return false;
    if (list2 == null) return false;
    int len1 = list1.size();
    int len2 = list2.size();
    if (len1 != len2) return false;
    for (int i = 0; i < len1; i++) {
      String str1 = list1.get(i);
      String str2 = list2.get(i);
      if (!(str1.equals(str2))) return false;
    }
    return true;
  }			    
  
  public void backupTabbedPaneStruct(BufferedReader cin,
				     PrintWriter cout) throws Exception { 
    String[] col = { "SERVICE_NAME", "NODE_PATH", "CHILD_NODE_NUMBER", 
		     "CHILD_NODE_ID", 
		     "CHILD_TAB_TITLE", 
		     "CHILD_TAB_FG_COLOR", "CHILD_TAB_BG_COLOR",
		     "CHILD_COMPONENT_TYPE", "CHILD_PANEL_ID",
		     "CHILD_METHOD_WHEN_SELECTED",
		     "CHILD_QUALIFICATION_METHOD", 
		     "CHILD_QUALIFICATION_PARAM" };
    int colSize = col.length;  
    String query = "select SERVICE_NAME, NODE_PATH, CHILD_NODE_NUMBER, CHILD_NODE_ID, CHILD_TAB_TITLE, CHILD_TAB_FG_COLOR, CHILD_TAB_BG_COLOR, CHILD_COMPONENT_TYPE, CHILD_PANEL_ID, CHILD_METHOD_WHEN_SELECTED, CHILD_QUALIFICATION_METHOD, CHILD_QUALIFICATION_PARAM from STRUCT2.TABBED_PANE_STRUCT order by SERVICE_NAME, NODE_PATH, CHILD_NODE_NUMBER";       
    int lines = 0;
    ResultSet resultSet = attendStatement.executeQuery(query);    
    if (resultSet != null) {
      while (resultSet.next()) {	
	StringBuffer sbuf = new StringBuffer();
	for (int i = 0; i < colSize; i++) {
	  String colVal = resultSet.getString(i+1);		    
	  if ((colVal == null) || (colVal.equals(""))) {
	    sbuf.append(" |");
	  } else {
	    sbuf.append(colVal.trim()).append("|");
	  } 
	}
	cout.println(sbuf.toString());
	lines++;
      }
      resultSet.close();
    } 
  }
  
  public void backupDataPanelStruct(BufferedReader cin,
				    PrintWriter cout) throws Exception { 
    String[] col = { "SERVICE_NAME", "PANEL_ID", 
		     "HEAD_TITLE", "HEAD_FG_COLOR", "HEAD_BG_COLOR",
		     "DATA_VIEW_TYPE",  
		     "BOTTOM_TITLE", 
		     "BOTTOM_FG_COLOR", "BOTTOM_BG_COLOR" };
    int colSize = col.length;
    String query = "select SERVICE_NAME, PANEL_ID, HEAD_TITLE, HEAD_FG_COLOR, HEAD_BG_COLOR, DATA_VIEW_TYPE, BOTTOM_TITLE, BOTTOM_FG_COLOR, BOTTOM_BG_COLOR from STRUCT2.DATA_PANEL_STRUCT order by SERVICE_NAME, PANEL_ID";          
    int lines = 0;
    ResultSet resultSet = attendStatement.executeQuery(query);    
    if (resultSet != null) {
      while (resultSet.next()) {	
	StringBuffer sbuf = new StringBuffer();
	for (int i = 0; i < colSize; i++) {
	  String colVal = resultSet.getString(i+1);		    
	  if ((colVal == null) || (colVal.equals(""))) {
	    sbuf.append(" |");
	  } else {
	    sbuf.append(colVal.trim()).append("|");
	  } 
	}
	cout.println(sbuf.toString());
	lines++;
      }
      resultSet.close();
    } 
  }
  
  public void backupTableViewStruct(BufferedReader cin,
				    PrintWriter cout) throws Exception { 
    String[] col = { "SERVICE_NAME", "PANEL_ID", 
		     "METHOD_WHEN_OPENED", 
		     "FONT_SIZE", "ROW_HEIGHT", 
		     "TABLE_FG_COLOR", "TABLE_BG_COLOR",
		     "SORTER_TYPE", "SELECTION_MODE",
		     "SWITCH_METHOD", "SWITCH_METHOD_PARAM",
		     "ROW_SEL_METHOD", "ROW_SEL_METHOD_PARAM" };
    int colSize = col.length;
    String query = "select SERVICE_NAME, PANEL_ID, METHOD_WHEN_OPENED, FONT_SIZE, ROW_HEIGHT, TABLE_FG_COLOR, TABLE_BG_COLOR, SORTER_TYPE, SELECTION_MODE, SWITCH_METHOD, SWITCH_METHOD_PARAM, ROW_SEL_METHOD, ROW_SEL_METHOD_PARAM from TABLE_VIEW_STRUCT order by SERVICE_NAME, PANEL_ID"; 
    int lines = 0;
    ResultSet resultSet = attendStatement.executeQuery(query);    
    if (resultSet != null) {
      while (resultSet.next()) {	
	StringBuffer sbuf = new StringBuffer();
	for (int i = 0; i < colSize; i++) {
	  String colVal = resultSet.getString(i+1);		    
	  if ((colVal == null) || (colVal.equals(""))) {
	    sbuf.append(" |");
	  } else {
	    sbuf.append(colVal.trim()).append("|");
	  } 
	}
	cout.println(sbuf.toString());
	lines++;
      }
      resultSet.close();
    } 
  }
  
  public void backupVarTableViewStruct(BufferedReader cin,
				       PrintWriter cout) throws Exception { 
    String[] col = { "SERVICE_NAME", "PANEL_ID", 
		     "KEY_COLUMN_CODE",       
		     "ADD_COLUMN_DISPLAY", 
		     "ADD_COLUMN_WIDTH", 
		     "ADD_COLUMN_RENDERER", 
		     "ADD_DATA_SWITCH_METHOD", 
		     "ADD_DATA_SWITCH_PARAM" };
    int colSize = col.length;
    String query = "select SERVICE_NAME, PANEL_ID, KEY_COLUMN_CODE, ADD_COLUMN_DISPLAY, ADD_COLUMN_WIDTH, ADD_COLUMN_RENDERER, ADD_DATA_SWITCH_METHOD, ADD_DATA_SWITCH_PARAM from VAR_TABLE_VIEW_STRUCT order by SERVICE_NAME, PANEL_ID";
    int lines = 0;
    ResultSet resultSet = attendStatement.executeQuery(query);    
    if (resultSet != null) {
      while (resultSet.next()) {	
	StringBuffer sbuf = new StringBuffer();
	for (int i = 0; i < colSize; i++) {
	  String colVal = resultSet.getString(i+1);		    
	  if ((colVal == null) || (colVal.equals(""))) {
	    sbuf.append(" |");
	  } else {
	    sbuf.append(colVal.trim()).append("|");
	  } 
	}
	cout.println(sbuf.toString());
	lines++;
      }
      resultSet.close();
    } 
  }

  public void backupTableColumnStruct(BufferedReader cin,
				      PrintWriter cout) throws Exception { 
    String[] col = { "SERVICE_NAME", "PANEL_ID", "COLUMN_NUMBER", 
		     "COLUMN_TITLE", 
		     "COLUMN_CODE", 
		     "COLUMN_DISPLAY", 
		     "COLUMN_WIDTH", 
		     "COLUMN_FG_COLOR", 
		     "COLUMN_BG_COLOR", 
		     "COLUMN_RENDERER" };	    
    int colSize = col.length;
    String query = "select SERVICE_NAME, PANEL_ID, COLUMN_NUMBER, COLUMN_TITLE, COLUMN_CODE, COLUMN_DISPLAY, COLUMN_WIDTH, COLUMN_FG_COLOR, COLUMN_BG_COLOR, COLUMN_RENDERER from STRUCT2.TABLE_COLUMN_STRUCT order by SERVICE_NAME, PANEL_ID, COLUMN_NUMBER";
    int lines = 0;
    ResultSet resultSet = attendStatement.executeQuery(query);    
    if (resultSet != null) {
      while (resultSet.next()) {	
	StringBuffer sbuf = new StringBuffer();
	for (int i = 0; i < colSize; i++) {
	  String colVal = resultSet.getString(i+1);		    
	  if ((colVal == null) || (colVal.equals(""))) {
	    sbuf.append(" |");
	  } else {
	    sbuf.append(colVal.trim()).append("|");
	  } 
	}
	cout.println(sbuf.toString());
	lines++;
      }
      resultSet.close();
    } 
  }

  public void backupJikanwariViewStruct(BufferedReader cin,
					PrintWriter cout) throws Exception { 
    String[] col = { "SERVICE_NAME", "PANEL_ID",
		     "METHOD_WHEN_OPENED", 
		     "ROW_DISPLAY", 
		     "ROW_HEIGHT_UNIT" };	    
    int colSize = col.length;	
    String query = "select SERVICE_NAME, PANEL_ID, METHOD_WHEN_OPENED, ROW_DISPLAY, ROW_HEIGHT_UNIT from STRUCT2.JIKANWARI_VIEW_STRUCT order by SERVICE_NAME, PANEL_ID";
    int lines = 0;
    ResultSet resultSet = attendStatement.executeQuery(query);    
    if (resultSet != null) {
      while (resultSet.next()) {	
	StringBuffer sbuf = new StringBuffer();
	for (int i = 0; i < colSize; i++) {
	  String colVal = resultSet.getString(i+1);		    
	  if ((colVal == null) || (colVal.equals(""))) {
	    sbuf.append(" |");
	  } else {
	    sbuf.append(colVal.trim()).append("|");
	  } 
	}
	cout.println(sbuf.toString());
	lines++;
      }
      resultSet.close();
    } 
  }

  public void backupJikanwariColumnStruct(BufferedReader cin,
					  PrintWriter cout) throws Exception {
    String[] col = { "SERVICE_NAME", "PANEL_ID", "COLUMN_NUMBER", 
		     "COLUMN_TITLE", 
		     "COLUMN_KEY", 
		     "COLUMN_WIDTH", 
		     "COLUMN_RENDERER" };
    int colSize = col.length;  
    String query = "select SERVICE_NAME, PANEL_ID, COLUMN_NUMBER, COLUMN_TITLE, COLUMN_KEY, COLUMN_WIDTH, COLUMN_RENDERER from STRUCT2.JIKANWARI_COLUMN_STRUCT order by SERVICE_NAME, PANEL_ID, COLUMN_NUMBER";
    int lines = 0;
    ResultSet resultSet = attendStatement.executeQuery(query);    
    if (resultSet != null) {
      while (resultSet.next()) {	
	StringBuffer sbuf = new StringBuffer();
	for (int i = 0; i < colSize; i++) {
	  String colVal = resultSet.getString(i+1);		    
	  if ((colVal == null) || (colVal.equals(""))) {
	    sbuf.append(" |");
	  } else {
	    sbuf.append(colVal.trim()).append("|");
	  } 
	}
	cout.println(sbuf.toString());
	lines++;
      }
      resultSet.close();
    } 
  }

  public void backupHtmlViewStruct(BufferedReader cin,
				   PrintWriter cout) throws Exception { 
    String[] col = { "SERVICE_NAME", "PANEL_ID", "URL" };
    int colSize = col.length;
    String query = "select SERVICE_NAME, PANEL_ID, URL from STRUCT2.HTML_VIEW_STRUCT order by SERVICE_NAME, PANEL_ID";
    int lines = 0;
    ResultSet resultSet = attendStatement.executeQuery(query);    
    if (resultSet != null) {
      while (resultSet.next()) {	
	StringBuffer sbuf = new StringBuffer();
	for (int i = 0; i < colSize; i++) {
	  String colVal = resultSet.getString(i+1);		    
	  if ((colVal == null) || (colVal.equals(""))) {
	    sbuf.append(" |");
	  } else {
	    sbuf.append(colVal.trim()).append("|");
	  } 
	}
	cout.println(sbuf.toString());
	lines++;
      }
      resultSet.close();
    } 
  }

  public void backupSimpleButtonStruct(BufferedReader cin,
				       PrintWriter cout) throws Exception { 
    String[] col = { "SERVICE_NAME", "PANEL_ID", "BUTTON_TITLE", 
		     "BUTTON_ROW", 
		     "BUTTON_COL", 
		     "BUTTON_FG_COLOR", 
		     "BUTTON_BG_COLOR", 
		     "QUALIFICATION_METHOD", 
		     "QUALIFICATION_METHOD_PARAM",  
		     "METHOD_TO_INVOKE", 
		     "METHOD_TO_INVOKE_PARAM" };	    
    int colSize = col.length; 
    String query = "select SERVICE_NAME, PANEL_ID, BUTTON_TITLE, BUTTON_ROW, BUTTON_COL, BUTTON_FG_COLOR, BUTTON_BG_COLOR, QUALIFICATION_METHOD, QUALIFICATION_METHOD_PARAM, METHOD_TO_INVOKE, METHOD_TO_INVOKE_PARAM from STRUCT2.SIMPLE_BUTTON_STRUCT order by SERVICE_NAME, PANEL_ID, BUTTON_ROW, BUTTON_COL";
    int lines = 0;
    ResultSet resultSet = attendStatement.executeQuery(query);    
    if (resultSet != null) {
      while (resultSet.next()) {	
	StringBuffer sbuf = new StringBuffer();
	for (int i = 0; i < colSize; i++) {
	  String colVal = resultSet.getString(i+1);		    
	  if ((colVal == null) || (colVal.equals(""))) {
	    sbuf.append(" |");
	  } else {
	    sbuf.append(colVal.trim()).append("|");
	  } 
	}
	cout.println(sbuf.toString());
	lines++;
      }
      resultSet.close();
    } 
  }

  public void backupUpdateButtonStruct(BufferedReader cin,
				       PrintWriter cout) throws Exception {
    String[] col = { "SERVICE_NAME", "PANEL_ID", "BUTTON_TITLE", 
		     "BUTTON_ROW", 
		     "BUTTON_COL", 
		     "BUTTON_FG_COLOR", 
		     "BUTTON_BG_COLOR", 
		     "QUALIFICATION_METHOD", 
		     "QUALIFICATION_METHOD_PARAM",  
		     "COMMAND_TYPE", 
		     "COMMAND_CODE" };	    
    int colSize = col.length;
    String query = "select SERVICE_NAME, PANEL_ID, BUTTON_TITLE, BUTTON_ROW, BUTTON_COL, BUTTON_FG_COLOR, BUTTON_BG_COLOR, QUALIFICATION_METHOD, QUALIFICATION_METHOD_PARAM, COMMAND_TYPE, COMMAND_CODE from STRUCT2.UPDATE_BUTTON_STRUCT order by SERVICE_NAME, PANEL_ID, BUTTON_ROW, BUTTON_COL";
    int lines = 0; 
    ResultSet resultSet = attendStatement.executeQuery(query);    
    if (resultSet != null) {
      while (resultSet.next()) {	
	StringBuffer sbuf = new StringBuffer();
	for (int i = 0; i < colSize; i++) {
	  String colVal = resultSet.getString(i+1);		    
	  if ((colVal == null) || (colVal.equals(""))) {
	    sbuf.append(" |");
	  } else {
	    sbuf.append(colVal.trim()).append("|");
	  } 
	}
	cout.println(sbuf.toString());
	lines++;
      }
      resultSet.close();
    } 
  }
    
  public void backupServerQueryStruct(BufferedReader cin,
				      PrintWriter cout) throws Exception {
    String[] col = { "SERVICE_NAME", 
		     "PANEL_ID", 
		     "SWITCH_FLAG", 
		     "DB_NAME", 
		     "PARAM_LIST", 
		     "QUERY_SQL", 
		     "QUALIFICATION_METHOD", 
		     "QUALIFICATION_METHOD_PARAM" };
    int colSize = col.length;
    String query = "select SERVICE_NAME, PANEL_ID, SWITCH_FLAG, DB_NAME, PARAM_LIST, QUERY_SQL, QUALIFICATION_METHOD, QUALIFICATION_METHOD_PARAM from SERVER_QUERY_STRUCT order by SERVICE_NAME, PANEL_ID";  
    int lines = 0;
    ResultSet resultSet = attendStatement.executeQuery(query);    
    if (resultSet != null) {
      while (resultSet.next()) {	
	StringBuffer sbuf = new StringBuffer();
	for (int i = 0; i < colSize; i++) {
	  String colVal = resultSet.getString(i+1);		    
	  if ((colVal == null) || (colVal.equals(""))) {
	    sbuf.append(" |");
	  } else {
	    if (i == 5) {
	      colVal = changeQuotationCode(colVal); 
	      colVal = changeConcatCode2(colVal);
	    }
	    sbuf.append(colVal.trim()).append("|");
	  } 
	}
	cout.println(sbuf.toString());
	lines++;
      }
      resultSet.close();
    } 
  }
    
  public void backupServerDeleteStruct(BufferedReader cin,
				       PrintWriter cout) throws Exception {
    String[] col = { "SERVICE_NAME", "DELETE_CODE", 
		     "DB_NAME", 
		     "WHERE_PARAM_LIST", 
		     "DELETE_SQL", 
		     "SIDE_EFFECT", 
		     "QUALIFICATION_METHOD", 
		     "QUALIFICATION_METHOD_PARAM" };
    int colSize = col.length;
    String query = "select SERVICE_NAME, DELETE_CODE, DB_NAME, WHERE_PARAM_LIST, DELETE_SQL, SIDE_EFFECT, QUALIFICATION_METHOD, QUALIFICATION_METHOD_PARAM from SERVER_DELETE_STRUCT order by SERVICE_NAME, DELETE_CODE";
    int lines = 0;
    ResultSet resultSet = attendStatement.executeQuery(query);    
    if (resultSet != null) {
      while (resultSet.next()) {	
	StringBuffer sbuf = new StringBuffer();
	for (int i = 0; i < colSize; i++) {
	  String colVal = resultSet.getString(i+1);		    
	  if ((colVal == null) || (colVal.equals(""))) {
	    sbuf.append(" |");
	  } else {
	    if ((i == 4) || (i == 5)) {
	      colVal = changeQuotationCode(colVal); 
	      colVal = changeConcatCode2(colVal);
	    }
	    sbuf.append(colVal.trim()).append("|");
	  } 
	}
	cout.println(sbuf.toString());
	lines++;
      }
      resultSet.close();
    }
  }

  public void backupServerUpdateStruct(BufferedReader cin,
				       PrintWriter cout) throws Exception {
    String[] col = { "SERVICE_NAME", "UPDATE_CODE", 
		     "DB_NAME", 
		     "SET_PARAM_LIST", 
		     "SET_EDITOR_LIST", 
		     "WHERE_PARAM_LIST", 
		     "UPDATE_SQL", 
		     "SIDE_EFFECT", 
		     "QUALIFICATION_METHOD", 
		     "QUALIFICATION_METHOD_PARAM" };
    int colSize = col.length;
    String query = "select SERVICE_NAME, UPDATE_CODE, DB_NAME, SET_PARAM_LIST, SET_EDITOR_LIST, WHERE_PARAM_LIST, UPDATE_SQL, SIDE_EFFECT, QUALIFICATION_METHOD, QUALIFICATION_METHOD_PARAM from SERVER_UPDATE_STRUCT order by SERVICE_NAME, UPDATE_CODE";  
    int lines = 0;
    ResultSet resultSet = attendStatement.executeQuery(query);    
    if (resultSet != null) {
      while (resultSet.next()) {	
	StringBuffer sbuf = new StringBuffer();
	for (int i = 0; i < colSize; i++) {
	  String colVal = resultSet.getString(i+1);		    
	  if ((colVal == null) || (colVal.equals(""))) {
	    sbuf.append(" |");
	  } else {
	    if ((i == 6) || (i == 7)) {
	      colVal = changeQuotationCode(colVal); 
	      colVal = changeConcatCode2(colVal);
	    }
	    sbuf.append(colVal.trim()).append("|");
	  } 
	}
	cout.println(sbuf.toString());
	lines++;
      }
      resultSet.close();
    } 
  }

  public void backupServerInsertStruct(BufferedReader cin,
				       PrintWriter cout) throws Exception {
    String[] col = { "SERVICE_NAME", "INSERT_CODE", 
		     "DB_NAME", 
		     "INSERT_PARAM_LIST", 
		     "INSERT_EDITOR_LIST", 
		     "WHERE_PARAM_LIST", 
		     "INSERT_SQL", 
		     "SIDE_EFFECT", 
		     "QUALIFICATION_METHOD", 
		     "QUALIFICATION_METHOD_PARAM" };
    int colSize = col.length;
    String query = "select SERVICE_NAME, INSERT_CODE, DB_NAME, INSERT_PARAM_LIST, INSERT_EDITOR_LIST, WHERE_PARAM_LIST, INSERT_SQL, SIDE_EFFECT, QUALIFICATION_METHOD, QUALIFICATION_METHOD_PARAM from SERVER_INSERT_STRUCT order by SERVICE_NAME, INSERT_CODE";  
    int lines = 0;
    ResultSet resultSet = attendStatement.executeQuery(query);    
    if (resultSet != null) {
      while (resultSet.next()) {	
	StringBuffer sbuf = new StringBuffer();
	for (int i = 0; i < colSize; i++) {
	  String colVal = resultSet.getString(i+1);		    
	  if ((colVal == null) || (colVal.equals(""))) {
	    sbuf.append(" |");
	  } else {
	    if ((i == 6) || (i == 7)) {
	      colVal = changeQuotationCode(colVal); 
	      colVal = changeConcatCode2(colVal);
	    }
	    sbuf.append(colVal.trim()).append("|");
	  } 
	}
	cout.println(sbuf.toString());
	lines++;
      }
      resultSet.close();
    } 
  }
    
  public void backupServerSpecialStruct(BufferedReader cin,
					PrintWriter cout) throws Exception {
    String[] col = { "SERVICE_NAME", "SPECIAL_CODE", 
		     "DB_NAME", 
		     "SPECIAL_PARAM_LIST", 
		     "QUALIFICATION_METHOD", 
		     "QUALIFICATION_METHOD_PARAM" };
    int colSize = col.length;
    String query = "select SERVICE_NAME, SPECIAL_CODE, DB_NAME, SPECIAL_PARAM_LIST, QUALIFICATION_METHOD, QUALIFICATION_METHOD_PARAM from SERVER_SPECIAL_STRUCT order by SERVICE_NAME, SPECIAL_CODE";  
    int lines = 0;
    ResultSet resultSet = attendStatement.executeQuery(query);    
    if (resultSet != null) {
      while (resultSet.next()) {	
	StringBuffer sbuf = new StringBuffer();
	for (int i = 0; i < colSize; i++) {
	  String colVal = resultSet.getString(i+1);		    
	  if ((colVal == null) || (colVal.equals(""))) {
	    sbuf.append(" |");
	  } else {
	    sbuf.append(colVal.trim()).append("|");
	  } 
	}
	cout.println(sbuf.toString());
	lines++;
      }
      resultSet.close();
    } 
  }
    
  public void backupCommonQueryStruct(BufferedReader cin,
				      PrintWriter cout) throws Exception {
    String[] col = { "QUERY_NAME", "DB_NAME", 
		     "PARAM_LIST", "QUERY_SQL"  };
    int colSize = col.length;
    String query = "select QUERY_NAME, DB_NAME, PARAM_LIST, QUERY_SQL from COMMON_QUERY_STRUCT order by QUERY_NAME"; 
    int lines = 0;
    ResultSet resultSet = attendStatement.executeQuery(query);    
    if (resultSet != null) {
      while (resultSet.next()) {	
	StringBuffer sbuf = new StringBuffer();
	for (int i = 0; i < colSize; i++) {
	  String colVal = resultSet.getString(i+1);		    
	  if ((colVal == null) || (colVal.equals(""))) {
	    sbuf.append(" |");
	  } else {
	    if (i == 3) {
	      colVal = changeQuotationCode(colVal); 
	      colVal = changeConcatCode2(colVal);
	    }
	    sbuf.append(colVal.trim()).append("|");
	  } 
	}
	cout.println(sbuf.toString());
	lines++;
      }
      resultSet.close();
    } 
  }
     
  public void backupColorNameDef(BufferedReader cin,
				 PrintWriter cout) throws Exception {
    String[] col = { "COLOR_NAME", 
		     "RED", 
		     "GREEN", 
		     "BLUE" };
    int colSize = col.length;
    String query = "select COLOR_NAME, RED, GREEN, BLUE from COLOR_NAME_DEF order by COLOR_NAME";  
    int lines = 0;
    ResultSet resultSet = attendStatement.executeQuery(query);    
    if (resultSet != null) {
      while (resultSet.next()) {	
	StringBuffer sbuf = new StringBuffer();
	for (int i = 0; i < colSize; i++) {
	  String colVal = resultSet.getString(i+1);		    
	  if ((colVal == null) || (colVal.equals(""))) {
	    sbuf.append(" |");
	  } else {
	    sbuf.append(colVal.trim()).append("|");
	  } 
	}
	cout.println(sbuf.toString());
	lines++;
      }
      resultSet.close();
    } 
  }
      
  public void backupGakumuCodeDef(BufferedReader cin,
				  PrintWriter cout) throws Exception {
    String[] col = { "CODE_CATEGORY", "GAKUMU_CODE", 
		     "GAKUMU_NAME", 
		     "SHORTER_NAME", 
		     "REMARK" };
    int colSize = col.length;
    String query = "select CODE_CATEGORY, GAKUMU_CODE, GAKUMU_NAME, SHORTER_NAME, REMARK  from GAKUMU_CODE_DEF order by CODE_CATEGORY, GAKUMU_CODE";  
    int lines = 0;
    ResultSet resultSet = attendStatement.executeQuery(query);    
    if (resultSet != null) {
      while (resultSet.next()) {	
	StringBuffer sbuf = new StringBuffer();
	for (int i = 0; i < colSize; i++) {
	  String colVal = resultSet.getString(i+1);		    
	  if ((colVal == null) || (colVal.equals(""))) {
	    sbuf.append(" |");
	  } else {
	    sbuf.append(colVal.trim()).append("|");
	  } 
	}
	cout.println(sbuf.toString());
	lines++;
      }
      resultSet.close();
    } 
  }

  public void readTabbedPaneStruct(BufferedReader cin,
				   PrintWriter cout) throws Exception { 	
    HashMap<String, ArrayList<String>> mapOrigin = new HashMap<String, ArrayList<String>>();
    HashMap<String, ArrayList<String>> mapChanged = new HashMap<String, ArrayList<String>>();
    int ins_count = 0;
    int upd_count = 0;
    int dlt_count = 0;  

    String[] col = { "SERVICE_NAME", "NODE_PATH", "CHILD_NODE_NUMBER", 
		     "CHILD_NODE_ID", 
		     "CHILD_TAB_TITLE", 
		     "CHILD_TAB_FG_COLOR", "CHILD_TAB_BG_COLOR",
		     "CHILD_COMPONENT_TYPE", "CHILD_PANEL_ID",
		     "CHILD_METHOD_WHEN_SELECTED",
		     "CHILD_QUALIFICATION_METHOD", 
		     "CHILD_QUALIFICATION_PARAM" };
    int colSize = col.length;  
	      
    String query = "select SERVICE_NAME, NODE_PATH, CHILD_NODE_NUMBER, CHILD_NODE_ID, CHILD_TAB_TITLE, CHILD_TAB_FG_COLOR, CHILD_TAB_BG_COLOR, CHILD_COMPONENT_TYPE, CHILD_PANEL_ID, CHILD_METHOD_WHEN_SELECTED, CHILD_QUALIFICATION_METHOD, CHILD_QUALIFICATION_PARAM from STRUCT2.TABBED_PANE_STRUCT order by SERVICE_NAME, NODE_PATH, CHILD_NODE_NUMBER";  
    ResultSet resultSet = attendStatement.executeQuery(query);    
    while (resultSet.next()) {	
      ArrayList<String> list = new ArrayList<String>();
      String[] tokens = new String[colSize];		
      for (int i = 0; i < colSize; i++) {
	String val = resultSet.getString(i+1);
	if ((val == null) || (val.equals(""))) {
	  tokens[i] = " ";
	} else {
	  tokens[i] = val.trim();
	}
	list.add(tokens[i]);
      }	  
      String key = tokens[0]+"|"+tokens[1]+"|"+tokens[2];
      mapOrigin.put(key, list);
    }
    resultSet.close();

    String s;
    while ((s = cin.readLine()) != null) {
      if (s.equals(".")) break;
      if (!s.startsWith("//")) {
	String[] tokens = s.split("\\|");
	if (tokens.length == colSize) {
	  ArrayList<String> list = new ArrayList<String>();
	  for (String token : tokens) {
	    list.add(token);
	  }	  
	  String key = tokens[0]+"|"+tokens[1]+"|"+tokens[2];
	  mapChanged.put(key, list);
	} else {
	  cout.println("Error:" + s);
	}
      }
    }

    Set<String> keySet = mapOrigin.keySet();
    ArrayList<String> listToRemove = new ArrayList<String>();
    for (String key : keySet) {
      if (!mapChanged.containsKey(key)) {
	ArrayList<String> list1 = mapOrigin.get(key);
	StringBuilder del = new StringBuilder();
	del.append("delete from TABBED_PANE_STRUCT where ");
	for (int i = 0; i < 3; i++) {
	  String colName = col[i];
	  String val = list1.get(i);
	  if (i < 2) {			    
	    if (colName.equals("CHILD_NODE_NUMBER")) {
	      del.append(colName).append(" = ").append(val).append(" and ");
	    } else {
	      del.append(colName).append(" = '").append(val).append("' and ");
	    }
	  } else {			    
	    if (colName.equals("CHILD_NODE_NUMBER")) {
	      del.append(colName).append(" = ").append(val);
	    } else {
	      del.append(colName).append(" = '").append(val).append("'");
	    }
	  }
	}
	String delString = del.toString();
	if (executeAttendUpdate(delString) != 1) {
	  cout.println(delString + "  ... failed");
	} else {
	  dlt_count++;
	  //	cout.println(delString);
	  listToRemove.add(key);
	}
      }
    }
    if (dlt_count != 0) {
      cout.println("TABBED_PANE_STRUCT: deleted " + dlt_count + " lines");
      for (String key : listToRemove) {
	mapOrigin.remove(key);
      }
    }

    keySet = mapChanged.keySet();
    for (String key : keySet) {
      if (!mapOrigin.containsKey(key)) {
	ArrayList<String> list2 = mapChanged.get(key);
	StringBuilder ins = new StringBuilder();
	ins.append("insert into TABBED_PANE_STRUCT (SERVICE_NAME, NODE_PATH, CHILD_NODE_NUMBER, CHILD_NODE_ID, CHILD_TAB_TITLE, CHILD_TAB_FG_COLOR, CHILD_TAB_BG_COLOR, CHILD_COMPONENT_TYPE, CHILD_PANEL_ID, CHILD_METHOD_WHEN_SELECTED, CHILD_QUALIFICATION_METHOD, CHILD_QUALIFICATION_PARAM) values (");
	for (int i = 0; i < colSize; i++) {
	  String colName = col[i];
	  String val = list2.get(i);
	  if (i < colSize-1) {
	    if (val.equals(" ")) {
	      ins.append("NULL, ");
	    } else if (colName.equals("CHILD_NODE_NUMBER")) {
	      ins.append(val).append(", ");
	    } else {
	      ins.append("'").append(val).append("', ");
	    }
	  } else {
	    if (val.equals(" ")) {
	      ins.append("NULL)");
	    } else {
	      ins.append("'").append(val).append("')");
	    }
	  }
	}     
	String insStr = ins.toString();
	if (executeAttendUpdate(insStr) != 1) {
	  cout.println(insStr + "  ... failed");
	} else {
	  ins_count++;
	  //	cout.println(insStr);
	  mapOrigin.put(key, list2);
	}
      }
    }
    if (ins_count != 0) {
      cout.println("TABBED_PANE_STRUCT: inserted " + ins_count + " lines");
    }
    
    keySet = mapOrigin.keySet();
    for (String key : keySet) {	    
      ArrayList<String> list1 = mapOrigin.get(key);
      ArrayList<String> list2 = mapChanged.get(key);
      if (!compareList(list1, list2)) {
	
	StringBuilder upd = new StringBuilder("update TABBED_PANE_STRUCT set ");
	for (int i = 3; i < colSize; i++) {
	  String val = list2.get(i);
	  String colName = col[i];
	  if (i < colSize-1) {
	    if (val.equals(" ")) {
	      upd.append(colName).append(" = null, ");
	    } else {
	      upd.append(colName).append(" = '").append(val).append("', ");
	    }
	  } else {
	    if (val.equals(" ")) {
	      upd.append(colName).append(" = null ");
	    } else {
	      upd.append(colName).append(" = '").append(val).append("' ");
	    }
	  }
	}
	upd.append(" where SERVICE_NAME = '").append(list2.get(0));
	upd.append("' and NODE_PATH = '").append(list2.get(1));
	upd.append("' and CHILD_NODE_NUMBER = ").append(list2.get(2));
	
	String updStr = upd.toString();	
	if (executeAttendUpdate(updStr) != 1) {
	  cout.println(updStr + "  ... failed");
	} else {
	  upd_count++;
	  //  cout.println(updStr); //
	}
      }		
    }
    if (upd_count != 0) {
      cout.println("TABBED_PANE_STRUCT: updated " + upd_count + " lines");
    }
  }
  
  public void readDataPanelStruct(BufferedReader cin,
				   PrintWriter cout) throws Exception { 
    HashMap<String, ArrayList<String>> mapOrigin = new HashMap<String, ArrayList<String>>();
    HashMap<String, ArrayList<String>> mapChanged = new HashMap<String, ArrayList<String>>();
    int ins_count = 0;
    int upd_count = 0;
    int dlt_count = 0;    
	
    String[] col = { "SERVICE_NAME", "PANEL_ID", 
		     "HEAD_TITLE", "HEAD_FG_COLOR", "HEAD_BG_COLOR",
		     "DATA_VIEW_TYPE",  
		     "BOTTOM_TITLE", "BOTTOM_FG_COLOR", "BOTTOM_BG_COLOR" };
    int colSize = col.length;
	
    String query = "select SERVICE_NAME, PANEL_ID, HEAD_TITLE, HEAD_FG_COLOR, HEAD_BG_COLOR, DATA_VIEW_TYPE, BOTTOM_TITLE, BOTTOM_FG_COLOR, BOTTOM_BG_COLOR from STRUCT2.DATA_PANEL_STRUCT order by SERVICE_NAME, PANEL_ID";
    ResultSet resultSet = attendStatement.executeQuery(query); 
    while (resultSet.next()) {	
      ArrayList<String> list = new ArrayList<String>();
      String[] tokens = new String[colSize];		
      for (int i = 0; i < colSize; i++) {
	String val = resultSet.getString(i+1);
	if ((val == null) || (val.equals(""))) {
	  tokens[i] = " ";
	} else {
	  tokens[i] = val.trim();
	}
	list.add(tokens[i]);
      }	  
      String key = tokens[0]+"|"+tokens[1];
      mapOrigin.put(key, list);
    }
    resultSet.close();
	    
    String s;
    while ((s = cin.readLine()) != null) {
      if (s.equals(".")) break;
      if (!s.startsWith("//")) {
	String[] tokens = s.split("\\|");
	if (tokens.length == colSize) {
	  ArrayList<String> list = new ArrayList<String>();
	  for (String token : tokens) {
	    list.add(token);
	  }	  
	  String key = tokens[0]+"|"+tokens[1];
	  mapChanged.put(key, list);
	} else {
	  cout.println("Error:" + s);
	}
      }
    }

    Set<String> keySet = mapOrigin.keySet();
    ArrayList<String> listToRemove = new ArrayList<String>();
    for (String key : keySet) {
      if (!mapChanged.containsKey(key)) {
	ArrayList<String> list1 = mapOrigin.get(key);
	StringBuilder del = new StringBuilder();
	del.append("delete from DATA_PANEL_STRUCT where ");
	for (int i = 0; i < 2; i++) {
	  String colName = col[i];
	  String val = list1.get(i);
	  if (i < 1) {
	    del.append(colName).append(" = '").append(val).append("' and ");
	  } else {
	    del.append(colName).append(" = '").append(val).append("'");
	  }
	}
	String delString = del.toString();
	if (executeAttendUpdate(delString) != 1) {
	  cout.println(delString + "  ... failed");
	} else {
	  dlt_count++;
	  //	cout.println(delString);
	  listToRemove.add(key);
	}
      }
    }
    if (dlt_count != 0) {
      cout.println("DATA_PANEL_STRUCT: deleted " + dlt_count + " lines");
      for (String key : listToRemove) {
	mapOrigin.remove(key);
      }
    }

    keySet = mapChanged.keySet();
    for (String key : keySet) {
      if (!mapOrigin.containsKey(key)) {
	ArrayList<String> list2 = mapChanged.get(key);
	StringBuilder ins = new StringBuilder();
	ins.append("insert into DATA_PANEL_STRUCT (SERVICE_NAME, PANEL_ID, HEAD_TITLE, HEAD_FG_COLOR, HEAD_BG_COLOR, DATA_VIEW_TYPE, BOTTOM_TITLE, BOTTOM_FG_COLOR, BOTTOM_BG_COLOR) values (");
	for (int i = 0; i < colSize; i++) {
	  String colName = col[i];
	  String val = list2.get(i);
	  if (i < colSize-1) {
	    if (val.equals(" ")) {
	      ins.append("NULL, ");
	    } else {
	      ins.append("'").append(val).append("', ");
	    }
	  } else {
	    if (val.equals(" ")) {
	      ins.append("NULL)");
	    } else {
	      ins.append("'").append(val).append("')");
	    }
	  }
	}     
	String insStr = ins.toString();
	if (executeAttendUpdate(insStr) != 1) {
	  cout.println(insStr + "  ... failed");
	} else {
	  ins_count++;
	  //	cout.println(insStr);
	  mapOrigin.put(key, list2);
	}
      }
    }
    if (ins_count != 0) {
      cout.println("DATA_PANEL_STRUCT: inserted " + ins_count + " lines");
    }
		
    keySet = mapOrigin.keySet();
    for (String key : keySet) {	    
      ArrayList<String> list1 = mapOrigin.get(key);
      ArrayList<String> list2 = mapChanged.get(key);
      if (!compareList(list1, list2)) {
	StringBuilder upd = new StringBuilder("update DATA_PANEL_STRUCT set ");
	for (int i = 2; i < colSize; i++) {
	  String val = list2.get(i);
	  String colName = col[i];
	  if (i < colSize-1) {
	    if (val.equals(" ")) {
	      upd.append(colName).append(" = null, ");
	    } else {
	      upd.append(colName).append(" = '").append(val).append("', ");
	    }
	  } else {
	    if (val.equals(" ")) {
	      upd.append(colName).append(" = null ");
	    } else {
	      upd.append(colName).append(" = '").append(val).append("' ");
	    }
	  }
	}
	upd.append(" where SERVICE_NAME = '").append(list2.get(0));
	upd.append("' and PANEL_ID = '").append(list2.get(1)).append("'");		    
	String updStr = upd.toString();
	
	if (executeAttendUpdate(updStr) != 1) {
	  cout.println(updStr + "  ... failed");
	} else {
	  upd_count++;
	  // cout.println(updStr); 
	}
      }		
    }
    if (upd_count != 0) {
      cout.println("DATA_PANEL_STRUCT: updated " + upd_count + " lines");
    }
  }

  public void readTableViewStruct(BufferedReader cin,
				  PrintWriter cout) throws Exception {	
    HashMap<String, ArrayList<String>> mapOrigin = new HashMap<String, ArrayList<String>>();
    HashMap<String, ArrayList<String>> mapChanged = new HashMap<String, ArrayList<String>>();
    int ins_count = 0;
    int upd_count = 0;
    int dlt_count = 0;  

    String[] col = { "SERVICE_NAME", 
		     "PANEL_ID", 
		     "METHOD_WHEN_OPENED", 
		     "FONT_SIZE", 
		     "ROW_HEIGHT", 
		     "TABLE_FG_COLOR", 
		     "TABLE_BG_COLOR",
		     "SORTER_TYPE", 
		     "SELECTION_MODE",			 
		     "SWITCH_METHOD", 
		     "SWITCH_METHOD_PARAM",
		     "ROW_SEL_METHOD",
		     "ROW_SEL_METHOD_PARAM" };    
    int colSize = col.length;

    String query = "select SERVICE_NAME, PANEL_ID, METHOD_WHEN_OPENED, FONT_SIZE, ROW_HEIGHT, TABLE_FG_COLOR, TABLE_BG_COLOR, SORTER_TYPE, SELECTION_MODE, SWITCH_METHOD, SWITCH_METHOD_PARAM, ROW_SEL_METHOD, ROW_SEL_METHOD_PARAM from TABLE_VIEW_STRUCT order by SERVICE_NAME, PANEL_ID";  

    ResultSet resultSet = attendStatement.executeQuery(query); 
    while (resultSet.next()) {	
      ArrayList<String> list = new ArrayList<String>();
      String[] tokens = new String[colSize];		
      for (int i = 0; i < colSize; i++) {
	String val = resultSet.getString(i+1);
	if ((val == null) || (val.equals(""))) {
	  tokens[i] = " ";
	} else {
	  tokens[i] = val.trim();
	}
	list.add(tokens[i]);
      }	  
      String key = tokens[0]+"|"+tokens[1];
      mapOrigin.put(key, list);
    }
    resultSet.close();
	    
    String s;
    while ((s = cin.readLine()) != null) {
      if (s.equals(".")) break;
      if (!s.startsWith("//")) {
	String[] tokens = s.split("\\|");	    
	if (tokens.length == colSize) {
	  ArrayList<String> list = new ArrayList<String>();
	  for (String token : tokens) {
	    list.add(token);
	  }	  
	  String key = tokens[0]+"|"+tokens[1];
	  mapChanged.put(key, list);
	} else {
	  cout.println(s);
	}
      }
    }

    Set<String> keySet = mapOrigin.keySet();
    ArrayList<String> listToRemove = new ArrayList<String>();
    for (String key : keySet) {
      if (!mapChanged.containsKey(key)) {
	ArrayList<String> list1 = mapOrigin.get(key);
	StringBuilder del = new StringBuilder();
	del.append("delete from TABLE_VIEW_STRUCT where ");
	for (int i = 0; i < 2; i++) {
	  String colName = col[i];
	  String val = list1.get(i);
	  if (i < 1) {
	    del.append(colName).append(" = '").append(val).append("' and ");
	  } else {
	    del.append(colName).append(" = '").append(val).append("'");
	  }
	}
	String delString = del.toString();
	if (executeAttendUpdate(delString) != 1) {
	  cout.println(delString + "  ... failed");
	} else {
	  dlt_count++;
	  //	cout.println(delString);
	  listToRemove.add(key);
	}
      }
    }
    if (dlt_count != 0) {
      cout.println("TABLE_VIEW_STRUCT: deleted " + dlt_count + " lines");
      for (String key : listToRemove) {
	mapOrigin.remove(key);
      }
    }

    keySet = mapChanged.keySet();
    for (String key : keySet) {
      if (!mapOrigin.containsKey(key)) {
	ArrayList<String> list2 = mapChanged.get(key);
	StringBuilder ins = new StringBuilder();
	ins.append("insert into TABLE_VIEW_STRUCT (SERVICE_NAME, PANEL_ID, METHOD_WHEN_OPENED, FONT_SIZE, ROW_HEIGHT, TABLE_FG_COLOR, TABLE_BG_COLOR, SORTER_TYPE, SELECTION_MODE, SWITCH_METHOD, SWITCH_METHOD_PARAM, ROW_SEL_METHOD, ROW_SEL_METHOD_PARAM) values (");
	for (int i = 0; i < colSize; i++) {
	  String colName = col[i];
	  String val = list2.get(i);
	  if (i < colSize-1) {
	    if (val.equals(" ")) {
	      ins.append("NULL, ");
	    } else if (colName.equals("FONT_SIZE")) {
	      ins.append(val).append(", ");
	    } else if (colName.equals("ROW_HEIGHT")) {
	      ins.append(val).append(", ");
	    } else {
	      ins.append("'").append(val).append("', ");
	    }
	  } else {
	    if (val.equals(" ")) {
	      ins.append("NULL)");
	    } else {
	      ins.append("'").append(val).append("')");
	    }
	  }
	}     
	String insStr = ins.toString();
	if (executeAttendUpdate(insStr) != 1) {
	  cout.println(insStr + "  ... failed");
	} else {
	  ins_count++;
	  //	cout.println(insStr);
	  mapOrigin.put(key, list2);
	}
      }
    }
    if (ins_count != 0) {
      cout.println("TABLE_VIEW_STRUCT: inserted " + ins_count + " lines");
    }
		
    keySet = mapOrigin.keySet();
    for (String key : keySet) {	    
      ArrayList<String> list1 = mapOrigin.get(key);
      ArrayList<String> list2 = mapChanged.get(key);
      if (!compareList(list1, list2)) {
	
	StringBuilder upd = new StringBuilder("update TABLE_VIEW_STRUCT set ");
	for (int i = 2; i < colSize; i++) {
	  String val = list2.get(i);
	  String colName = col[i];
	  if (i < colSize-1) {
	    if (val.equals(" ")) {
	      upd.append(colName).append(" = null, ");
	    } else if (colName.equals("FONT_SIZE")) {
	      upd.append(colName).append(" = ").append(val).append(", ");
	    } else if (colName.equals("ROW_HEIGHT")) {
	      upd.append(colName).append(" = ").append(val).append(", ");
	    } else {
	      upd.append(colName).append(" = '").append(val).append("', ");
	    }
	  } else {
	    if (val.equals(" ")) {
	      upd.append(colName).append(" = null ");
	    } else {
	      upd.append(colName).append(" = '").append(val).append("' ");
	    }
	  }
	}
	upd.append(" where SERVICE_NAME = '").append(list2.get(0));
	upd.append("' and PANEL_ID = '").append(list2.get(1)).append("'");		    
	String updStr = upd.toString();
	
	if (executeAttendUpdate(updStr) != 1) {
	  cout.println(updStr + "  ... failed");
	} else {
	  upd_count++;
	  //  cout.println(updStr); 
	}
      }		
    }
    if (upd_count != 0) {
      cout.println("TABLE_VIEW_STRUCT: updated " + upd_count + " lines");
    }
  }

  public void readVarTableViewStruct (BufferedReader cin,
				   PrintWriter cout) throws Exception { 	
    HashMap<String, ArrayList<String>> mapOrigin = new HashMap<String, ArrayList<String>>();
    HashMap<String, ArrayList<String>> mapChanged = new HashMap<String, ArrayList<String>>();

    int ins_count = 0;
    int upd_count = 0;
    int dlt_count = 0;  
	
    String[] col = { "SERVICE_NAME", 
		     "PANEL_ID", 
		     "KEY_COLUMN_CODE",       
		     "ADD_COLUMN_DISPLAY", 
		     "ADD_COLUMN_WIDTH", 
		     "ADD_COLUMN_RENDERER", 
		     "ADD_DATA_SWITCH_METHOD", 
		     "ADD_DATA_SWITCH_PARAM" };    
    int colSize = col.length;

    String query = "select SERVICE_NAME, PANEL_ID, KEY_COLUMN_CODE, ADD_COLUMN_DISPLAY, ADD_COLUMN_WIDTH, ADD_COLUMN_RENDERER, ADD_DATA_SWITCH_METHOD, ADD_DATA_SWITCH_PARAM from VAR_TABLE_VIEW_STRUCT order by SERVICE_NAME, PANEL_ID";
    ResultSet resultSet = attendStatement.executeQuery(query); 
    while (resultSet.next()) {	
      ArrayList<String> list = new ArrayList<String>();
      String[] tokens = new String[colSize];		
      for (int i = 0; i < colSize; i++) {
	String val = resultSet.getString(i+1);
	if ((val == null) || (val.equals(""))) {
	  tokens[i] = " ";
	} else {
	  tokens[i] = val.trim();
	}
	list.add(tokens[i]);
      }	  
      String key = tokens[0]+"|"+tokens[1];
      mapOrigin.put(key, list);
    }
    resultSet.close();
	    
    String s;
    while ((s = cin.readLine()) != null) {
      if (s.equals(".")) break;
      if (!s.startsWith("//")) {
	String[] tokens = s.split("\\|");
	if (tokens.length == colSize) {
	  ArrayList<String> list = new ArrayList<String>();
	  for (String token : tokens) {
	    list.add(token);
	  }	  
	  String key = tokens[0]+"|"+tokens[1];
	  mapChanged.put(key, list);
	} else {
	  cout.println(s);
	}
      }
    }

    Set<String> keySet = mapOrigin.keySet();
    ArrayList<String> listToRemove = new ArrayList<String>();
    for (String key : keySet) {
      if (!mapChanged.containsKey(key)) {
	ArrayList<String> list1 = mapOrigin.get(key);
	StringBuilder del = new StringBuilder();
	del.append("delete from VAR_TABLE_VIEW_STRUCT where ");
	for (int i = 0; i < 2; i++) {
	  String colName = col[i];
	  String val = list1.get(i);
	  if (i < 1) {
	    del.append(colName).append(" = '").append(val).append("' and ");
	  } else {
	    del.append(colName).append(" = '").append(val).append("'");
	  }
	}
	String delString = del.toString();
	if (executeAttendUpdate(delString) != 1) {
	  cout.println(delString + "  ... failed");
	} else {
	  dlt_count++;
	  //	cout.println(delString);
	  listToRemove.add(key);
	}
      }
    }
    if (dlt_count != 0) {
      cout.println("VAR_TABLE_VIEW_STRUCT: deleted " + dlt_count + " lines");
      for (String key : listToRemove) {
	mapOrigin.remove(key);
      }
    }

    keySet = mapChanged.keySet();
    for (String key : keySet) {
      if (!mapOrigin.containsKey(key)) {
	ArrayList<String> list2 = mapChanged.get(key);
	StringBuilder ins = new StringBuilder();
	ins.append("insert into VAR_TABLE_VIEW_STRUCT (SERVICE_NAME, PANEL_ID, KEY_COLUMN_CODE, ADD_COLUMN_DISPLAY, ADD_COLUMN_WIDTH, ADD_COLUMN_RENDERER, ADD_DATA_SWITCH_METHOD, ADD_DATA_SWITCH_PARAM) values (");
	for (int i = 0; i < colSize; i++) {
	  String colName = col[i];
	  String val = list2.get(i);
	  if (i < colSize-1) {
	    if (val.equals(" ")) {
	      ins.append("NULL, ");
	    } else if (colName.equals("ADD_COLUMN_WIDTH")) {
	      ins.append(val).append(", ");
	    } else {
	      ins.append("'").append(val).append("', ");
	    }
	  } else {
	    if (val.equals(" ")) {
	      ins.append("NULL)");
	    } else {
	      ins.append("'").append(val).append("')");
	    }
	  }
	}     
	String insStr = ins.toString();
	if (executeAttendUpdate(insStr) != 1) {
	  cout.println(insStr + "  ... failed");
	} else {
	  ins_count++;
	  //	cout.println(insStr);
	  mapOrigin.put(key, list2);
	}
      }
    }
    if (ins_count != 0) {
      cout.println("VAR_TABLE_VIEW_STRUCT: inserted " + ins_count + " lines");
    }
    
    keySet = mapOrigin.keySet();
    for (String key : keySet) {	    
      ArrayList<String> list1 = mapOrigin.get(key);
      ArrayList<String> list2 = mapChanged.get(key);
      if (!compareList(list1, list2)) {
	
	StringBuilder upd = new StringBuilder("update VAR_TABLE_VIEW_STRUCT set ");
	for (int i = 2; i < colSize; i++) {
	  String val = list2.get(i);
	  String colName = col[i];
	  if (i < colSize-1) {
	    if (val.equals(" ")) {
	      upd.append(colName).append(" = null, ");
	    } else if (colName.equals("ADD_COLUMN_WIDTH")) {
	      upd.append(colName).append(" = ").append(val).append(", ");
	    } else {
	      upd.append(colName).append(" = '").append(val).append("', ");
	    }
	  } else {
	    if (val.equals(" ")) {
	      upd.append(colName).append(" = null ");
	    } else {
	      upd.append(colName).append(" = '").append(val).append("' ");
	    }
	  }
	}
	upd.append(" where SERVICE_NAME = '").append(list2.get(0));
	upd.append("' and PANEL_ID = '").append(list2.get(1)).append("'");		    
	String updStr = upd.toString();
	
	if (executeAttendUpdate(updStr) != 1) {
	  cout.println(updStr + "  ... failed");
	} else {
	  upd_count++;
	  //  cout.println(updStr);
	}
      }		
    }
    if (upd_count != 0) {
      cout.println("VAR_TABLE_VIEW_STRUCT: updated " + upd_count + " lines");
    }    
  }


  public void readTableColumnStruct(BufferedReader cin,
				    PrintWriter cout) throws Exception { 	
    HashMap<String, ArrayList<String>> mapOrigin = new HashMap<String, ArrayList<String>>();
    HashMap<String, ArrayList<String>> mapChanged = new HashMap<String, ArrayList<String>>();
    int ins_count = 0;
    int upd_count = 0;
    int dlt_count = 0;  
	
    String[] col = { "SERVICE_NAME", "PANEL_ID", "COLUMN_NUMBER", 
		     "COLUMN_TITLE", 
		     "COLUMN_CODE", 
		     "COLUMN_DISPLAY", 
		     "COLUMN_WIDTH", 
		     "COLUMN_FG_COLOR", 
		     "COLUMN_BG_COLOR", 
		     "COLUMN_RENDERER" };
    int colSize = col.length;

    String query = "select SERVICE_NAME, PANEL_ID, COLUMN_NUMBER, COLUMN_TITLE, COLUMN_CODE, COLUMN_DISPLAY, COLUMN_WIDTH, COLUMN_FG_COLOR, COLUMN_BG_COLOR, COLUMN_RENDERER from STRUCT2.TABLE_COLUMN_STRUCT order by SERVICE_NAME, PANEL_ID, COLUMN_NUMBER";
    ResultSet resultSet = attendStatement.executeQuery(query); 
    while (resultSet.next()) {	
      ArrayList<String> list = new ArrayList<String>();
      String[] tokens = new String[colSize];		
      for (int i = 0; i < colSize; i++) {
	String val = resultSet.getString(i+1);
	if ((val == null) || (val.equals(""))) {
	  tokens[i] = " ";
	} else {
	  tokens[i] = val.trim();
	}
	list.add(tokens[i]);
      }	  
      String key = tokens[0]+"|"+tokens[1]+"|"+tokens[2];
      mapOrigin.put(key, list);
    }
    resultSet.close();
	    
    String s;
    while ((s = cin.readLine()) != null) {
      if (s.equals(".")) break;
      if (!s.startsWith("//")) {
	String[] tokens = s.split("\\|");
	if (tokens.length == colSize) {
	  ArrayList<String> list = new ArrayList<String>();
	  for (String token : tokens) {
	    list.add(token);
	  }	  
	  String key = tokens[0]+"|"+tokens[1]+"|"+tokens[2];
	  mapChanged.put(key, list);
	} else {
	  cout.println(s);
	}
      }
    }

    Set<String> keySet = mapOrigin.keySet();
    ArrayList<String> listToRemove = new ArrayList<String>();
    for (String key : keySet) {
      if (!mapChanged.containsKey(key)) {
	ArrayList<String> list1 = mapOrigin.get(key);
	StringBuilder del = new StringBuilder();
	del.append("delete from TABLE_COLUMN_STRUCT where ");
	for (int i = 0; i < 3; i++) {
	  String colName = col[i];
	  String val = list1.get(i);
	  if (i < 2) {			    
	    if (colName.equals("COLUMN_NUMBER")) {
	      del.append(colName).append(" = ").append(val).append(" and ");
	    } else {
	      del.append(colName).append(" = '").append(val).append("' and ");
	    }
	  } else {			    
	    if (colName.equals("COLUMN_NUMBER")) {
	      del.append(colName).append(" = ").append(val);
	    } else {
	      del.append(colName).append(" = '").append(val).append("'");
	    }
	  }
	}
	String delString = del.toString();
	if (executeAttendUpdate(delString) != 1) {
	  cout.println(delString + "  ... failed");
	} else {
	  dlt_count++;
	  //  cout.println(delString);
	  listToRemove.add(key);
	}
      }
    }
    if (dlt_count != 0) {
      cout.println("TABLE_COLUMN_STRUCT: deleted " + dlt_count + " lines");
      for (String key : listToRemove) {
	mapOrigin.remove(key);
      }
    }
    
    keySet = mapChanged.keySet();
    for (String key : keySet) {
      if (!mapOrigin.containsKey(key)) {
	ArrayList<String> list2 = mapChanged.get(key);
	StringBuilder ins = new StringBuilder();
	ins.append("insert into TABLE_COLUMN_STRUCT (SERVICE_NAME, PANEL_ID, COLUMN_NUMBER, COLUMN_TITLE, COLUMN_CODE, COLUMN_DISPLAY, COLUMN_WIDTH, COLUMN_FG_COLOR, COLUMN_BG_COLOR, COLUMN_RENDERER) values (");
	for (int i = 0; i < colSize; i++) {
	  String colName = col[i];
	  String val = list2.get(i);
	  if (i < colSize-1) {
	    if (val.equals(" ")) {
	      ins.append("NULL, ");
	    } else if (colName.equals("COLUMN_NUMBER")) {
	      ins.append(val).append(", ");
	    } else if (colName.equals("COLUMN_WIDTH")) {
	      ins.append(val).append(", ");
	    } else {
	      ins.append("'").append(val).append("', ");
	    }
	  } else {
	    if (val.equals(" ")) {
	      ins.append("NULL)");
	    } else {
	      ins.append("'").append(val).append("')");
	    }
	  }
	}     
	String insStr = ins.toString();
	if (executeAttendUpdate(insStr) != 1) {
	  cout.println(insStr + "  ... failed");
	} else {
	  ins_count++;
	  //	cout.println(insStr);
	  mapOrigin.put(key, list2);
	}
      }
    }
    if (ins_count != 0) {
      cout.println("TABLE_COLUMN_STRUCT: inserted " + ins_count + " lines");
    }
    
    keySet = mapOrigin.keySet();
    for (String key : keySet) {	    
      ArrayList<String> list1 = mapOrigin.get(key);
      ArrayList<String> list2 = mapChanged.get(key);
      if (!compareList(list1, list2)) {
	
	StringBuilder upd = new StringBuilder("update TABLE_COLUMN_STRUCT set ");
	for (int i = 3; i < colSize; i++) {
	  String val = list2.get(i);
	  String colName = col[i];
	  if (i < colSize-1) {
	    if (val.equals(" ")) {
	      upd.append(colName).append(" = null, ");
	    } else if (colName.equals("COLUMN_WIDTH")) {
	      upd.append(colName).append(" = ").append(val).append(", ");
	    }  else {
	      upd.append(colName).append(" = '").append(val).append("', ");
	    }
	  } else {
	    if (val.equals(" ")) {
	      upd.append(colName).append(" = null ");
	    } else {
	      upd.append(colName).append(" = '").append(val).append("' ");
	    }
	  }
	}
	upd.append(" where SERVICE_NAME = '").append(list2.get(0));
	upd.append("' and PANEL_ID = '").append(list2.get(1));
	upd.append("' and COLUMN_NUMBER = ").append(list2.get(2));
	
	String updStr = upd.toString();
	
	if (executeAttendUpdate(updStr) != 1) {
	  cout.println(updStr + "  ... failed");
	} else {
	  upd_count++;
	  // cout.println(updStr); 
	}
      }		
    }
    if (upd_count != 0) {
      cout.println("TABLE_COLUMN_STRUCT: updated " + upd_count + " lines");
    }
  }

  public void readJikanwariViewStruct(BufferedReader cin,
				      PrintWriter cout) throws Exception { 	
    HashMap<String, ArrayList<String>> mapOrigin = new HashMap<String, ArrayList<String>>();
    HashMap<String, ArrayList<String>> mapChanged = new HashMap<String, ArrayList<String>>();
    int ins_count = 0;
    int upd_count = 0;
    int dlt_count = 0;  
	
    String[] col = { "SERVICE_NAME", "PANEL_ID",
		     "METHOD_WHEN_OPENED", 
		     "ROW_DISPLAY", 
		     "ROW_HEIGHT_UNIT" };    
    int colSize = col.length;	
	
    String query = "select SERVICE_NAME, PANEL_ID, METHOD_WHEN_OPENED, ROW_DISPLAY, ROW_HEIGHT_UNIT from STRUCT2.JIKANWARI_VIEW_STRUCT order by SERVICE_NAME, PANEL_ID";
    ResultSet resultSet = attendStatement.executeQuery(query); 
    while (resultSet.next()) {	
      ArrayList<String> list = new ArrayList<String>();
      String[] tokens = new String[colSize];		
      for (int i = 0; i < colSize; i++) {
	String val = resultSet.getString(i+1);
	if ((val == null) || (val.equals(""))) {
	  tokens[i] = " ";
	} else {
	  tokens[i] = val.trim();
	}
	list.add(tokens[i]);
      }	  
      String key = tokens[0]+"|"+tokens[1];
      mapOrigin.put(key, list);
    }
    resultSet.close();
	    
    String s;
    while ((s = cin.readLine()) != null) {
      if (s.equals(".")) break;
      if (!s.startsWith("//")) {
	String[] tokens = s.split("\\|");
	if (tokens.length == colSize) {
	  ArrayList<String> list = new ArrayList<String>();
	  for (String token : tokens) {
	    list.add(token);
	  }	  
	  String key = tokens[0]+"|"+tokens[1];
	  mapChanged.put(key, list);
	} else {
	  cout.println(s);
	}
      }
    }

    Set<String> keySet = mapOrigin.keySet();
    ArrayList<String> listToRemove = new ArrayList<String>();
    for (String key : keySet) {
      if (!mapChanged.containsKey(key)) {
	ArrayList<String> list1 = mapOrigin.get(key);
	StringBuilder del = new StringBuilder();
	del.append("delete from JIKANWARI_VIEW_STRUCT where ");
	for (int i = 0; i < 2; i++) {
	  String colName = col[i];
	  String val = list1.get(i);
	  if (i < 1) {
	    del.append(colName).append(" = '").append(val).append("' and ");
	  } else {
	    del.append(colName).append(" = '").append(val).append("'");
	  }
	}
	String delString = del.toString();
	if (executeAttendUpdate(delString) != 1) {
	  cout.println(delString + "  ... failed");
	} else {
	  dlt_count++;
	  //  cout.println(delString);
	  listToRemove.add(key);
	}
      }
    }
    if (dlt_count != 0) {
      cout.println("JIKANWARI_VIEW_STRUCT: deleted " + dlt_count + " lines");
      for (String key : listToRemove) {
	mapOrigin.remove(key);
      }
    }
    
    keySet = mapChanged.keySet();
    for (String key : keySet) {
      if (!mapOrigin.containsKey(key)) {
	ArrayList<String> list2 = mapChanged.get(key);
	StringBuilder ins = new StringBuilder();
	ins.append("insert into JIKANWARI_VIEW_STRUCT (SERVICE_NAME, PANEL_ID, METHOD_WHEN_OPENED, ROW_DISPLAY, ROW_HEIGHT_UNIT) values (");
	for (int i = 0; i < colSize; i++) {
	  String colName = col[i];
	  String val = list2.get(i);
	  if (i < colSize-1) {
	    if (val.equals(" ")) {
	      ins.append("NULL, ");
	    } else {
	      ins.append("'").append(val).append("', ");
	    }
	  } else {
	    if (val.equals(" ")) {
	      ins.append("NULL)");
	    } else {
	      ins.append("").append(val).append(")");
	    }
	  }
	}     
	String insStr = ins.toString();
	if (executeAttendUpdate(insStr) != 1) {
	  cout.println(insStr + "  ... failed");
	} else {
	  ins_count++;
	  //  cout.println(insStr);
	  mapOrigin.put(key, list2);
	}
      }
    }
    if (ins_count != 0) {
      cout.println("JIKANWARI_VIEW_STRUCT: inserted " + ins_count + " lines");
    }
    
    keySet = mapOrigin.keySet();
    for (String key : keySet) {	    
      ArrayList<String> list1 = mapOrigin.get(key);
      ArrayList<String> list2 = mapChanged.get(key);
      if (!compareList(list1, list2)) {	
	StringBuilder upd = new StringBuilder("update JIKANWARI_VIEW_STRUCT set ");
	for (int i = 2; i < colSize; i++) {
	  String val = list2.get(i);
	  String colName = col[i];
	  if (i < colSize-1) {
	    if (val.equals(" ")) {
	      upd.append(colName).append(" = null, ");
	    } else {
	      upd.append(colName).append(" = '").append(val).append("', ");
	    }
	  } else {
	    if (val.equals(" ")) {
	      upd.append(colName).append(" = null ");
	    } else {
	      upd.append(colName).append(" = ").append(val).append(" ");
	    }
	  }
	}
	upd.append(" where SERVICE_NAME = '").append(list2.get(0));
	upd.append("' and PANEL_ID = '").append(list2.get(1)).append("'");		    
	String updStr = upd.toString();	
	if (executeAttendUpdate(updStr) != 1) {
	  cout.println(updStr + "  ... failed");
	} else {
	  upd_count++;
	  // cout.println(updStr); 
	}
      }		
    }
    if (upd_count != 0) {
      cout.println("JIKANWARI_VIEW_STRUCT: updated " + upd_count + " lines");
    }
  }

  public void readJikanwariColumnStruct(BufferedReader cin,
					PrintWriter cout) throws Exception { 	
    HashMap<String, ArrayList<String>> mapOrigin = new HashMap<String, ArrayList<String>>();
    HashMap<String, ArrayList<String>> mapChanged = new HashMap<String, ArrayList<String>>();
    int ins_count = 0;
    int upd_count = 0;
    int dlt_count = 0;   
	
    String[] col = { "SERVICE_NAME", "PANEL_ID", "COLUMN_NUMBER", 
		     "COLUMN_TITLE", 
		     "COLUMN_KEY", 
		     "COLUMN_WIDTH", 
		     "COLUMN_RENDERER" };	    
    int colSize = col.length;	
	
    String query = "select SERVICE_NAME, PANEL_ID, COLUMN_NUMBER, COLUMN_TITLE, COLUMN_KEY, COLUMN_WIDTH, COLUMN_RENDERER from STRUCT2.JIKANWARI_COLUMN_STRUCT order by SERVICE_NAME, PANEL_ID, COLUMN_NUMBER";
    ResultSet resultSet = attendStatement.executeQuery(query); 
    while (resultSet.next()) {	
      ArrayList<String> list = new ArrayList<String>();
      String[] tokens = new String[colSize];		
      for (int i = 0; i < colSize; i++) {
	String val = resultSet.getString(i+1);
	if ((val == null) || (val.equals(""))) {
	  tokens[i] = " ";
	} else {
	  tokens[i] = val.trim();
	}
	list.add(tokens[i]);
      }	  
      String key = tokens[0]+"|"+tokens[1]+"|"+tokens[2];
      mapOrigin.put(key, list);
    }
    resultSet.close();
	    
    String s;
    while ((s = cin.readLine()) != null) {
      if (s.equals(".")) break;
      if (!s.startsWith("//")) {
	String[] tokens = s.split("\\|");
	if (tokens.length == colSize) {
	  ArrayList<String> list = new ArrayList<String>();
	  for (String token : tokens) {
	    list.add(token);
	  }	  
	  String key = tokens[0]+"|"+tokens[1]+"|"+tokens[2];
	  mapChanged.put(key, list);
	} else {
	  cout.println(s);
	}
      }
    }

    Set<String> keySet = mapOrigin.keySet();
    ArrayList<String> listToRemove = new ArrayList<String>();
    for (String key : keySet) {
      if (!mapChanged.containsKey(key)) {
	ArrayList<String> list1 = mapOrigin.get(key);
	StringBuilder del = new StringBuilder();
	del.append("delete from JIKANWARI_COLUMN_STRUCT where ");
	for (int i = 0; i < 3; i++) {
	  String colName = col[i];
	  String val = list1.get(i);
	  if (i < 2) {			    
	    if (colName.equals("COLUMN_NUMBER")) {
	      del.append(colName).append(" = ").append(val).append(" and ");
	    } else {
	      del.append(colName).append(" = '").append(val).append("' and ");
	    }
	  } else {			    
	    if (colName.equals("COLUMN_NUMBER")) {
	      del.append(colName).append(" = ").append(val);
	    } else {
	      del.append(colName).append(" = '").append(val).append("'");
	    }
	  }
	}
	String delString = del.toString();
	if (executeAttendUpdate(delString) != 1) {
	  cout.println(delString + "  ... failed");
	} else {
	  dlt_count++;
	  //  cout.println(delString);
	  listToRemove.add(key);
	}
      }
    }
    if (dlt_count != 0) {
      cout.println("JIKANWARI_COLUMN_STRUCT: deleted " + dlt_count + " lines");
      for (String key : listToRemove) {
	mapOrigin.remove(key);
      }
    }
    
    keySet = mapChanged.keySet();
    for (String key : keySet) {
      if (!mapOrigin.containsKey(key)) {
	ArrayList<String> list2 = mapChanged.get(key);
	StringBuilder ins = new StringBuilder();
	ins.append("insert into JIKANWARI_COLUMN_STRUCT (SERVICE_NAME, PANEL_ID, COLUMN_NUMBER, COLUMN_TITLE, COLUMN_KEY, COLUMN_WIDTH, COLUMN_RENDERER) values (");
	for (int i = 0; i < colSize; i++) {
	  String colName = col[i];
	  String val = list2.get(i);
	  if (i < colSize-1) {
	    if (val.equals(" ")) {
	      ins.append("NULL, ");
	    } else if (colName.equals("COLUMN_NUMBER")) {
	      ins.append(val).append(", ");
	    } else if (colName.equals("COLUMN_WIDTH")) {
	      ins.append(val).append(", ");
	    } else {
	      ins.append("'").append(val).append("', ");
	    }
	  } else {
	    if (val.equals(" ")) {
	      ins.append("NULL)");
	    } else {
	      ins.append("'").append(val).append("')");
	    }
	  }
	}     
	String insStr = ins.toString();
	if (executeAttendUpdate(insStr) != 1) {
	  cout.println(insStr + "  ... failed");
	} else {
	  ins_count++;
	  //  cout.println(insStr);
	  mapOrigin.put(key, list2);
	}
      }
    }
    if (ins_count != 0) {
      cout.println("JIKANWARI_COLUMN_STRUCT: inserted " + ins_count + " lines");
    }
    
    keySet = mapOrigin.keySet();
    for (String key : keySet) {	    
      ArrayList<String> list1 = mapOrigin.get(key);
      ArrayList<String> list2 = mapChanged.get(key);
      if (!compareList(list1, list2)) {
	StringBuilder upd = new StringBuilder("update JIKANWARI_COLUMN_STRUCT set ");
	for (int i = 3; i < colSize; i++) {
	  String val = list2.get(i);
	  String colName = col[i];
	  if (i < colSize-1) {
	    if (val.equals(" ")) {
	      upd.append(colName).append(" = null, ");
	    } else if (colName.equals("COLUMN_WIDTH")) {
	      upd.append(colName).append(" = ").append(val).append(", ");
	    }  else {
	      upd.append(colName).append(" = '").append(val).append("', ");
	    }
	  } else {
	    if (val.equals(" ")) {
	      upd.append(colName).append(" = null ");
	    } else {
	      upd.append(colName).append(" = '").append(val).append("' ");
	    }
	  }
	}
	upd.append(" where SERVICE_NAME = '").append(list2.get(0));
	upd.append("' and PANEL_ID = '").append(list2.get(1));
	upd.append("' and COLUMN_NUMBER = ").append(list2.get(2));
	
	String updStr = upd.toString();
	if (executeAttendUpdate(updStr) != 1) {
	  cout.println(updStr + "  ... failed");
	} else {
	  upd_count++;
	  //  cout.println(updStr); 
	}
      }		
    }
    if (upd_count != 0) {
      cout.println("JIKANWARI_COLUMN_STRUCT: updated " + upd_count + " lines");
    }
  }

  public void readHtmlViewStruct(BufferedReader cin,
				 PrintWriter cout) throws Exception { 	
    HashMap<String, ArrayList<String>> mapOrigin = new HashMap<String, ArrayList<String>>();
    HashMap<String, ArrayList<String>> mapChanged = new HashMap<String, ArrayList<String>>();

    int ins_count = 0;
    int upd_count = 0;
    int dlt_count = 0; 
	
    String[] col = { "SERVICE_NAME", "PANEL_ID",  "URL" };
    int colSize = col.length;

    String query = "select SERVICE_NAME, PANEL_ID, URL from STRUCT2.HTML_VIEW_STRUCT order by SERVICE_NAME, PANEL_ID";
    ResultSet resultSet = attendStatement.executeQuery(query); 
    while (resultSet.next()) {	
      ArrayList<String> list = new ArrayList<String>();
      String[] tokens = new String[colSize];		
      for (int i = 0; i < colSize; i++) {
	String val = resultSet.getString(i+1);
	if ((val == null) || (val.equals(""))) {
	  tokens[i] = " ";
	} else {
	  tokens[i] = val.trim();
	}
	list.add(tokens[i]);
      }	  
      String key = tokens[0]+"|"+tokens[1];
      mapOrigin.put(key, list);
    }
    resultSet.close();
	    
    String s;
    while ((s = cin.readLine()) != null) {
      if (s.equals(".")) break;
      if (!s.startsWith("//")) {
	String[] tokens = s.split("\\|");
	if (tokens.length == colSize) {
	  ArrayList<String> list = new ArrayList<String>();
	  for (String token : tokens) {
	    list.add(token);
	  }	  
	  String key = tokens[0]+"|"+tokens[1];
	  mapChanged.put(key, list);
	} else {
	  //  cout.println(s);
	}
      }
    }

    Set<String> keySet = mapOrigin.keySet();
    ArrayList<String> listToRemove = new ArrayList<String>();
    for (String key : keySet) {
      if (!mapChanged.containsKey(key)) {
	ArrayList<String> list1 = mapOrigin.get(key);
	StringBuilder del = new StringBuilder();
	del.append("delete from HTML_VIEW_STRUCT where ");
	for (int i = 0; i < 2; i++) {
	  String colName = col[i];
	  String val = list1.get(i);
	  if (i < 1) {
	    del.append(colName).append(" = '").append(val).append("' and ");
	  } else {
	    del.append(colName).append(" = '").append(val).append("'");
	  }
	}
	String delString = del.toString();
	if (executeAttendUpdate(delString) != 1) {
	  cout.println(delString + "  ... failed");
	} else {
	  dlt_count++;
	  //  cout.println(delString);
	  listToRemove.add(key);
	}
      }
    }
    if (dlt_count != 0) {
      cout.println("HTML_VIEW_STRUCT: deleted " + dlt_count + " lines");
      for (String key : listToRemove) {
	mapOrigin.remove(key);
      }
    }

    keySet = mapChanged.keySet();
    for (String key : keySet) {
      if (!mapOrigin.containsKey(key)) {
	ArrayList<String> list2 = mapChanged.get(key);
	StringBuilder ins = new StringBuilder();
	ins.append("insert into HTML_VIEW_STRUCT (SERVICE_NAME, PANEL_ID, URL) values (");
	for (int i = 0; i < colSize; i++) {
	  String colName = col[i];
	  String val = list2.get(i);
	  if (i < colSize-1) {
	    if (val.equals(" ")) {
	      ins.append("NULL, ");
	    } else {
	      ins.append("'").append(val).append("', ");
	    }
	  } else {
	    if (val.equals(" ")) {
	      ins.append("NULL)");
	    } else {
	      ins.append("'").append(val).append("')");
	    }
	  }
	}     
	String insStr = ins.toString();
	if (executeAttendUpdate(insStr) != 1) {
	  cout.println(insStr + "  ... failed");
	} else {
	  ins_count++;
	  //  cout.println(insStr);
	  mapOrigin.put(key, list2);
	}
      }
    }
    if (ins_count != 0) {
      cout.println("HTML_VIEW_STRUCT: inserted " + ins_count + " lines");
    }
    
    keySet = mapOrigin.keySet();
    for (String key : keySet) {	    
      ArrayList<String> list1 = mapOrigin.get(key);
      ArrayList<String> list2 = mapChanged.get(key);
      if (!compareList(list1, list2)) {	
	StringBuilder upd = new StringBuilder("update HTML_VIEW_STRUCT set ");
	for (int i = 2; i < colSize; i++) {
	  String val = list2.get(i);
	  String colName = col[i];
	  if (i < colSize-1) {
	    if (val.equals(" ")) {
	      upd.append(colName).append(" = null, ");
	    }  else {
	      upd.append(colName).append(" = '").append(val).append("', ");
	    }
	  } else {
	    if (val.equals(" ")) {
	      upd.append(colName).append(" = null ");
	    } else {
	      upd.append(colName).append(" = '").append(val).append("' ");
	    }
	  }
	}
	upd.append(" where SERVICE_NAME = '").append(list2.get(0));
	upd.append("' and PANEL_ID = '").append(list2.get(1)).append("'");
	
	String updStr = upd.toString();
	if (executeAttendUpdate(updStr) != 1) {
	  cout.println(updStr + "  ... failed");
	} else {
	  upd_count++;
	  // cout.println(updStr); 
	}
      }		
    }
    if (upd_count != 0) {
      cout.println("HTML_VIEW_STRUCT: updated " + upd_count + " lines");
    }
  }

  public void readSimpleButtonStruct (BufferedReader cin,
				      PrintWriter cout) throws Exception { 	
    HashMap<String, ArrayList<String>> mapOrigin = new HashMap<String, ArrayList<String>>();
    HashMap<String, ArrayList<String>> mapChanged = new HashMap<String, ArrayList<String>>();
    int ins_count = 0;
    int upd_count = 0;
    int dlt_count = 0;  
	
    String[] col = { "SERVICE_NAME", "PANEL_ID", "BUTTON_TITLE", 
		     "BUTTON_ROW", 
		     "BUTTON_COL", 
		     "BUTTON_FG_COLOR", 
		     "BUTTON_BG_COLOR", 
		     "QUALIFICATION_METHOD", 
		     "QUALIFICATION_METHOD_PARAM",  
		     "METHOD_TO_INVOKE", 
		     "METHOD_TO_INVOKE_PARAM" };    
    int colSize = col.length;

    String query = "select SERVICE_NAME, PANEL_ID, BUTTON_TITLE, BUTTON_ROW, BUTTON_COL, BUTTON_FG_COLOR, BUTTON_BG_COLOR, QUALIFICATION_METHOD, QUALIFICATION_METHOD_PARAM, METHOD_TO_INVOKE, METHOD_TO_INVOKE_PARAM from STRUCT2.SIMPLE_BUTTON_STRUCT order by SERVICE_NAME, PANEL_ID, BUTTON_ROW, BUTTON_COL";
    ResultSet resultSet = attendStatement.executeQuery(query); 
    while (resultSet.next()) {	
      ArrayList<String> list = new ArrayList<String>();
      String[] tokens = new String[colSize];		
      for (int i = 0; i < colSize; i++) {
	String val = resultSet.getString(i+1);
	if ((val == null) || (val.equals(""))) {
	  tokens[i] = " ";
	} else {
	  tokens[i] = val.trim();
	}
	list.add(tokens[i]);
      }	  
      String key = tokens[0]+"|"+tokens[1]+"|"+tokens[2];
      mapOrigin.put(key, list);
    }
    resultSet.close();
	    
    String s;
    while ((s = cin.readLine()) != null) {
      if (s.equals(".")) break;
      if (!s.startsWith("//")) {
	String[] tokens = s.split("\\|");
	if (tokens.length == colSize) {
	  ArrayList<String> list = new ArrayList<String>();
	  for (String token : tokens) {
	    list.add(token);
	  }	  
	  String key = tokens[0]+"|"+tokens[1]+"|"+tokens[2];
	  mapChanged.put(key, list);
	} else {
	  cout.println(s);
	}
      }
    }

    Set<String> keySet = mapOrigin.keySet();
    ArrayList<String> listToRemove = new ArrayList<String>();
    for (String key : keySet) {
      if (!mapChanged.containsKey(key)) {
	ArrayList<String> list1 = mapOrigin.get(key);
	StringBuilder del = new StringBuilder();
	del.append("delete from SIMPLE_BUTTON_STRUCT where ");
	for (int i = 0; i < 3; i++) {
	  String colName = col[i];
	  String val = list1.get(i);
	  if (i < 2) {
	    del.append(colName).append(" = '").append(val).append("' and ");
	  } else {
	    del.append(colName).append(" = '").append(val).append("'");
	  }
	}
	String delString = del.toString();
	if (executeAttendUpdate(delString) != 1) {
	  cout.println(delString + "  ... failed");
	} else {
	  dlt_count++;
	  //  cout.println(delString);
	  listToRemove.add(key);
	}
      }
    }
    if (dlt_count != 0) {
      cout.println("SIMPLE_BUTTON_STRUCT: deleted " + dlt_count + " lines");
      for (String key : listToRemove) {
	mapOrigin.remove(key);
      }
    }
    
    keySet = mapChanged.keySet();
    for (String key : keySet) {
      if (!mapOrigin.containsKey(key)) {
	ArrayList<String> list2 = mapChanged.get(key);
	StringBuilder ins = new StringBuilder();
	ins.append("insert into SIMPLE_BUTTON_STRUCT (SERVICE_NAME, PANEL_ID, BUTTON_TITLE, BUTTON_ROW, BUTTON_COL, BUTTON_FG_COLOR, BUTTON_BG_COLOR, QUALIFICATION_METHOD, QUALIFICATION_METHOD_PARAM, METHOD_TO_INVOKE, METHOD_TO_INVOKE_PARAM) values (");
	for (int i = 0; i < colSize; i++) {
	  String colName = col[i];
	  String val = list2.get(i);
	  if (i < colSize-1) {
	    if (val.equals(" ")) {
	      ins.append("NULL, ");
	    } else if (colName.equals("BUTTON_ROW")) {
	      ins.append(val).append(", ");
	    } else if (colName.equals("BUTTON_COL")) {
	      ins.append(val).append(", ");
	    } else {
	      ins.append("'").append(val).append("', ");
	    }
	  } else {
	    if (val.equals(" ")) {
	      ins.append("NULL)");
	    } else {
	      ins.append("'").append(val).append("')");
	    }
	  }
	}     
	String insStr = ins.toString();
	if (executeAttendUpdate(insStr) != 1) {
	  cout.println(insStr + "  ... failed");
	} else {
	  ins_count++;
	  // cout.println(insStr);
	  mapOrigin.put(key, list2);
	}
      }
    }
    if (ins_count != 0) {
      cout.println("SIMPLE_BUTTON_STRUCT: inserted " + ins_count + " lines");
    }
    
    keySet = mapOrigin.keySet();
    for (String key : keySet) {	    
      ArrayList<String> list1 = mapOrigin.get(key);
      ArrayList<String> list2 = mapChanged.get(key);
      if (!compareList(list1, list2)) {
	
	StringBuilder upd = new StringBuilder("update SIMPLE_BUTTON_STRUCT set ");
	for (int i = 3; i < colSize; i++) {
	  String val = list2.get(i);
	  String colName = col[i];
	  if (i < colSize-1) {
	    if (val.equals(" ")) {
	      upd.append(colName).append(" = null, ");
	    } else if (colName.equals("BUTTON_ROW")) {
	      upd.append(colName).append(" = ").append(val).append(", ");
	    } else if (colName.equals("BUTTON_COL")) {
	      upd.append(colName).append(" = ").append(val).append(", ");
	    } else {
	      upd.append(colName).append(" = '").append(val).append("', ");
	    }
	  } else {
	    if (val.equals(" ")) {
	      upd.append(colName).append(" = null ");
	    } else {
	      upd.append(colName).append(" = '").append(val).append("' ");
	    }
	  }
	}
	upd.append(" where SERVICE_NAME = '").append(list2.get(0));
	upd.append("' and PANEL_ID = '").append(list2.get(1));
	upd.append("' and BUTTON_TITLE = '").append(list2.get(2)).append("'");
	
	String updStr = upd.toString();	
	if (executeAttendUpdate(updStr) != 1) {
	  cout.println(updStr + "  ... failed");
	} else {
	  upd_count++;
	  // cout.println(updStr); 
	}
      }		
    }
    if (upd_count != 0) {
      cout.println("SIMPLE_BUTTON_STRUCT: updated " + upd_count + " lines");
    }
  }

  public void readUpdateButtonStruct (BufferedReader cin,
				      PrintWriter cout) throws Exception { 	
    HashMap<String, ArrayList<String>> mapOrigin = new HashMap<String, ArrayList<String>>();
    HashMap<String, ArrayList<String>> mapChanged = new HashMap<String, ArrayList<String>>();
    int ins_count = 0;
    int upd_count = 0;
    int dlt_count = 0;   
	
    String[] col = { "SERVICE_NAME", "PANEL_ID", "BUTTON_TITLE", 
		     "BUTTON_ROW", 
		     "BUTTON_COL", 
		     "BUTTON_FG_COLOR", 
		     "BUTTON_BG_COLOR", 
		     "QUALIFICATION_METHOD", 
		     "QUALIFICATION_METHOD_PARAM",  
		     "COMMAND_TYPE", 
		     "COMMAND_CODE" };    
    int colSize = col.length;
	
    String query = "select SERVICE_NAME, PANEL_ID, BUTTON_TITLE, BUTTON_ROW, BUTTON_COL, BUTTON_FG_COLOR, BUTTON_BG_COLOR, QUALIFICATION_METHOD, QUALIFICATION_METHOD_PARAM, COMMAND_TYPE, COMMAND_CODE from STRUCT2.UPDATE_BUTTON_STRUCT order by SERVICE_NAME, PANEL_ID, BUTTON_ROW, BUTTON_COL";
    ResultSet resultSet = attendStatement.executeQuery(query); 
    while (resultSet.next()) {	
      ArrayList<String> list = new ArrayList<String>();
      String[] tokens = new String[colSize];		
      for (int i = 0; i < colSize; i++) {
	String val = resultSet.getString(i+1);
	if ((val == null) || (val.equals(""))) {
	  tokens[i] = " ";
	} else {
	  tokens[i] = val.trim();
	}
	list.add(tokens[i]);
      }	  
      String key = tokens[0]+"|"+tokens[1]+"|"+tokens[2];
      mapOrigin.put(key, list);
    }
    resultSet.close();
	    
    String s;
    while ((s = cin.readLine()) != null) {
      if (s.equals(".")) break;
      if (!s.startsWith("//")) {
	String[] tokens = s.split("\\|");
	if (tokens.length == colSize) {
	  ArrayList<String> list = new ArrayList<String>();
	  for (String token : tokens) {
	    list.add(token);
	  }	  
	  String key = tokens[0]+"|"+tokens[1]+"|"+tokens[2];
	  mapChanged.put(key, list);
	} else {
	  cout.println(s);
	}
      }
    }

    Set<String> keySet = mapOrigin.keySet();
    ArrayList<String> listToRemove = new ArrayList<String>();
    for (String key : keySet) {
      if (!mapChanged.containsKey(key)) {
	ArrayList<String> list1 = mapOrigin.get(key);
	StringBuilder del = new StringBuilder();
	del.append("delete from UPDATE_BUTTON_STRUCT where ");
	for (int i = 0; i < 3; i++) {
	  String colName = col[i];
	  String val = list1.get(i);
	  if (i < 2) {
	    del.append(colName).append(" = '").append(val).append("' and ");
	  } else {
	    del.append(colName).append(" = '").append(val).append("'");
	  }
	}
	String delString = del.toString();
	if (executeAttendUpdate(delString) != 1) {
	  cout.println(delString + "  ... failed");
	} else {
	  dlt_count++;
	  // cout.println(delString);
	  listToRemove.add(key);
	}
      }
    }
    if (dlt_count != 0) {
      cout.println("UPDATE_BUTTON_STRUCT: deleted " + dlt_count + " lines");
      for (String key : listToRemove) {
	mapOrigin.remove(key);
      }
    }
    
    keySet = mapChanged.keySet();
    for (String key : keySet) {
      if (!mapOrigin.containsKey(key)) {
	ArrayList<String> list2 = mapChanged.get(key);
	StringBuilder ins = new StringBuilder();
	ins.append("insert into UPDATE_BUTTON_STRUCT (SERVICE_NAME, PANEL_ID, BUTTON_TITLE, BUTTON_ROW, BUTTON_COL, BUTTON_FG_COLOR, BUTTON_BG_COLOR, QUALIFICATION_METHOD, QUALIFICATION_METHOD_PARAM, COMMAND_TYPE, COMMAND_CODE) values (");
	for (int i = 0; i < colSize; i++) {
	  String colName = col[i];
	  String val = list2.get(i);
	  if (i < colSize-1) {
	    if (val.equals(" ")) {
	      ins.append("NULL, ");
	    } else if (colName.equals("BUTTON_ROW")) {
	      ins.append(val).append(", ");
	    } else if (colName.equals("BUTTON_COL")) {
	      ins.append(val).append(", ");
	    } else {
	      ins.append("'").append(val).append("', ");
	    }
	  } else {
	    if (val.equals(" ")) {
	      ins.append("NULL)");
	    } else {
	      ins.append("'").append(val).append("')");
	    }
	  }
	}     
	String insStr = ins.toString();
	if (executeAttendUpdate(insStr) != 1) {
	  cout.println(insStr + "  ... failed");
	} else {
	  ins_count++;
	  // cout.println(insStr);
	  mapOrigin.put(key, list2);
	}
      }
    }
    if (ins_count != 0) {
      cout.println("UPDATE_BUTTON_STRUCT: inserted " + ins_count + " lines");
    }
    
    keySet = mapOrigin.keySet();
    for (String key : keySet) {	    
      ArrayList<String> list1 = mapOrigin.get(key);
      ArrayList<String> list2 = mapChanged.get(key);
      if (!compareList(list1, list2)) {
	
	StringBuilder upd = new StringBuilder("update UPDATE_BUTTON_STRUCT set ");
	for (int i = 3; i < colSize; i++) {
	  String val = list2.get(i);
	  String colName = col[i];
	  if (i < colSize-1) {
	    if (val.equals(" ")) {
	      upd.append(colName).append(" = null, ");
	    } else if (colName.equals("BUTTON_ROW")) {
	      upd.append(colName).append(" = ").append(val).append(", ");
	    } else if (colName.equals("BUTTON_COL")) {
	      upd.append(colName).append(" = ").append(val).append(", ");
	    } else {
	      upd.append(colName).append(" = '").append(val).append("', ");
	    }
	  } else {
	    if (val.equals(" ")) {
	      upd.append(colName).append(" = null ");
	    } else {
	      upd.append(colName).append(" = '").append(val).append("' ");
	    }
	  }
	}
	upd.append(" where SERVICE_NAME = '").append(list2.get(0));
	upd.append("' and PANEL_ID = '").append(list2.get(1));
	upd.append("' and BUTTON_TITLE = '").append(list2.get(2)).append("'");
	
	String updStr = upd.toString();
	if (executeAttendUpdate(updStr) != 1) {
	  cout.println(updStr + "  ... failed");
	} else {
	  upd_count++;
	  // cout.println(updStr); 
	}
      }		
    }
    if (upd_count != 0) {
      cout.println("UPDATE_BUTTON_STRUCT: updated " + upd_count + " lines");
    }    
  }

  public void readServerDeleteStruct (BufferedReader cin,
				   PrintWriter cout) throws Exception { 	
    HashMap<String, ArrayList<String>> mapOrigin = new HashMap<String, ArrayList<String>>();
    HashMap<String, ArrayList<String>> mapChanged = new HashMap<String, ArrayList<String>>();
    int ins_count = 0;
    int upd_count = 0;
    int dlt_count = 0;  
	
    String[] col = { "SERVICE_NAME", "DELETE_CODE", 
		     "DB_NAME", 
		     "WHERE_PARAM_LIST", 
		     "DELETE_SQL", 
		     "SIDE_EFFECT", 
		     "QUALIFICATION_METHOD", 
		     "QUALIFICATION_METHOD_PARAM" };
    
    int colSize = col.length;

    String query = "select SERVICE_NAME, DELETE_CODE, DB_NAME, WHERE_PARAM_LIST, DELETE_SQL, SIDE_EFFECT, QUALIFICATION_METHOD, QUALIFICATION_METHOD_PARAM from SERVER_DELETE_STRUCT order by SERVICE_NAME, DELETE_CODE";      
    ResultSet resultSet = attendStatement.executeQuery(query); 
    while (resultSet.next()) {	
      ArrayList<String> list = new ArrayList<String>();
      String[] tokens = new String[colSize];		
      for (int i = 0; i < colSize; i++) {
	String val = resultSet.getString(i+1);
	if ((val == null) || (val.equals(""))) {
	  tokens[i] = " ";
	} else {
	  if ((i == 4) || (i == 5)) {
	    val = changeQuotationCode(val);
	  }
	  tokens[i] = val.trim();
	}
	list.add(tokens[i]);
      }	  
      String key = tokens[0]+"|"+tokens[1];
      mapOrigin.put(key, list);
    }
    resultSet.close();
	    
    String s;
    while ((s = cin.readLine()) != null) {
      if (s.equals(".")) break;
      if (!s.startsWith("//")) {
	String[] tokens = s.split("\\|");
	if (tokens.length == colSize) {
	  ArrayList<String> list = new ArrayList<String>();
	  for (String token : tokens) {
	    String token2 = changeConcatCode(token);
	    list.add(token2);
	  }	  
	  String key = tokens[0]+"|"+tokens[1];
	  mapChanged.put(key, list);
	} else {
	  cout.println(s);
	}
      }
    }

    Set<String> keySet = mapOrigin.keySet();
    ArrayList<String> listToRemove = new ArrayList<String>();
    for (String key : keySet) {
      if (!mapChanged.containsKey(key)) {
	ArrayList<String> list1 = mapOrigin.get(key);
	StringBuilder del = new StringBuilder();
	del.append("delete from SERVER_DELETE_STRUCT where ");
	for (int i = 0; i < 2; i++) {
	  String colName = col[i];
	  String val = list1.get(i);
	  if (i < 1) {
	    del.append(colName).append(" = '").append(val).append("' and ");
	  } else {
	    del.append(colName).append(" = '").append(val).append("'");
	  }
	}
	String delString = del.toString();
	if (executeAttendUpdate(delString) != 1) {
	  cout.println(delString + "  ... failed");
	} else {
	  dlt_count++;
	  // cout.println(delString);
	  listToRemove.add(key);
	}
      }
    }
    if (dlt_count != 0) {
      cout.println("SERVER_DELETE_STRUCT: deleted " + dlt_count + " lines");
      for (String key : listToRemove) {
	mapOrigin.remove(key);
      }
    }
    
    keySet = mapChanged.keySet();
    for (String key : keySet) {
      if (!mapOrigin.containsKey(key)) {
	ArrayList<String> list2 = mapChanged.get(key);
	StringBuilder ins = new StringBuilder();
	ins.append("insert into SERVER_DELETE_STRUCT (SERVICE_NAME, DELETE_CODE, DB_NAME, WHERE_PARAM_LIST, DELETE_SQL, SIDE_EFFECT, QUALIFICATION_METHOD, QUALIFICATION_METHOD_PARAM) values (");
	for (int i = 0; i < colSize; i++) {
	  String colName = col[i];
	  String val = list2.get(i);
	  if (i < colSize-1) {
	    if (val.equals(" ")) {
	      ins.append("NULL, ");
	    } else {
	      ins.append("'").append(val).append("', ");
	    }
	  } else {
	    if (val.equals(" ")) {
	      ins.append("NULL)");
	    } else {
	      ins.append("'").append(val).append("')");
	    }
	  }
	}     
	String insStr = ins.toString();
	if (executeAttendUpdate(insStr) != 1) {
	  cout.println(insStr + "  ... failed");
	} else {
	  ins_count++;
	  // cout.println(insStr);
	  mapOrigin.put(key, list2);
	}
      }
    }
    if (ins_count != 0) {
      cout.println("SERVER_DELETE_STRUCT: inserted " + ins_count + " lines");
    }
    
    keySet = mapOrigin.keySet();
    for (String key : keySet) {	    
      ArrayList<String> list1 = mapOrigin.get(key);
      ArrayList<String> list2 = mapChanged.get(key);
      if (!compareList(list1, list2)) {
	
	StringBuilder upd = new StringBuilder("update SERVER_DELETE_STRUCT set ");
	for (int i = 2; i < colSize; i++) {
	  String val = list2.get(i);
	  String colName = col[i];
	  if (i < colSize-1) {
	    if (val.equals(" ")) {
	      upd.append(colName).append(" = null, ");
	    } else {
	      upd.append(colName).append(" = '").append(val).append("', ");
	    }
	  } else {
	    if (val.equals(" ")) {
	      upd.append(colName).append(" = null ");
	    } else {
	      upd.append(colName).append(" = '").append(val).append("' ");
	    }
	  }
	}
	upd.append(" where SERVICE_NAME = '").append(list2.get(0));
	upd.append("' and DELETE_CODE = '").append(list2.get(1)).append("'");  
	String updStr = upd.toString();
	
	if (executeAttendUpdate(updStr) != 1) {
	  cout.println(updStr + "  ... failed");
	} else {
	  upd_count++;
	  // cout.println(updStr); 
	}
      }		
    }
    if (upd_count != 0) {
      cout.println("SERVER_DELETE_STRUCT: updated " + upd_count + " lines");
    }
  }

  public void readServerUpdateStruct (BufferedReader cin,
				   PrintWriter cout) throws Exception { 	
    HashMap<String, ArrayList<String>> mapOrigin = new HashMap<String, ArrayList<String>>();
    HashMap<String, ArrayList<String>> mapChanged = new HashMap<String, ArrayList<String>>();
    int ins_count = 0;
    int upd_count = 0;
    int dlt_count = 0; 
	
    String[] col = { "SERVICE_NAME", "UPDATE_CODE", 
		     "DB_NAME", 
		     "SET_PARAM_LIST", 
		     "SET_EDITOR_LIST", 
		     "WHERE_PARAM_LIST", 
		     "UPDATE_SQL", 
		     "SIDE_EFFECT", 
		     "QUALIFICATION_METHOD", 
		     "QUALIFICATION_METHOD_PARAM" };    
    int colSize = col.length;

    String query = "select SERVICE_NAME, UPDATE_CODE, DB_NAME, SET_PARAM_LIST, SET_EDITOR_LIST, WHERE_PARAM_LIST, UPDATE_SQL, SIDE_EFFECT, QUALIFICATION_METHOD, QUALIFICATION_METHOD_PARAM from SERVER_UPDATE_STRUCT order by SERVICE_NAME, UPDATE_CODE";  
    ResultSet resultSet = attendStatement.executeQuery(query); 
    while (resultSet.next()) {	
      ArrayList<String> list = new ArrayList<String>();
      String[] tokens = new String[colSize];		
      for (int i = 0; i < colSize; i++) {
	String val = resultSet.getString(i+1);
	if ((val == null) || (val.equals(""))) {
	  tokens[i] = " ";
	} else {
	  if ((i == 6) || (i == 7)) {
	    val = changeQuotationCode(val);
	  }
	  tokens[i] = val.trim();
	}
	list.add(tokens[i]);
      }	  
      String key = tokens[0]+"|"+tokens[1];
      mapOrigin.put(key, list);
    }
    resultSet.close();
	    
    String s;
    while ((s = cin.readLine()) != null) {
      if (s.equals(".")) break;
      if (!s.startsWith("//")) {
	String[] tokens = s.split("\\|");
	if (tokens.length == colSize) {
	  ArrayList<String> list = new ArrayList<String>();
	  for (String token : tokens) {
	    String token2 = changeConcatCode(token);
	    list.add(token2);
	  }	  
	  String key = tokens[0]+"|"+tokens[1];
	  mapChanged.put(key, list);
	} else {
	  cout.println(s);
	}
      }
    }

    Set<String> keySet = mapOrigin.keySet();
    ArrayList<String> listToRemove = new ArrayList<String>();
    for (String key : keySet) {
      if (!mapChanged.containsKey(key)) {
	ArrayList<String> list1 = mapOrigin.get(key);
	StringBuilder del = new StringBuilder();
	del.append("delete from SERVER_UPDATE_STRUCT where ");
	for (int i = 0; i < 2; i++) {
	  String colName = col[i];
	  String val = list1.get(i);
	  if (i < 1) {
	    del.append(colName).append(" = '").append(val).append("' and ");
	  } else {
	    del.append(colName).append(" = '").append(val).append("'");
	  }
	}
	String delString = del.toString();
	if (executeAttendUpdate(delString) != 1) {
	  cout.println(delString + "  ... failed");
	} else {
	  dlt_count++;
	  // cout.println(delString);
	  listToRemove.add(key);
	}
      }
    }
    if (dlt_count != 0) {
      cout.println("SERVER_UPDATE_STRUCT: deleted " + dlt_count + " lines");
      for (String key : listToRemove) {
	mapOrigin.remove(key);
      }
    }
    
    keySet = mapChanged.keySet();
    for (String key : keySet) {
      if (!mapOrigin.containsKey(key)) {
	ArrayList<String> list2 = mapChanged.get(key);
	StringBuilder ins = new StringBuilder();
	ins.append("insert into SERVER_UPDATE_STRUCT (SERVICE_NAME, UPDATE_CODE, DB_NAME, SET_PARAM_LIST, SET_EDITOR_LIST, WHERE_PARAM_LIST, UPDATE_SQL, SIDE_EFFECT, QUALIFICATION_METHOD, QUALIFICATION_METHOD_PARAM) values (");
	for (int i = 0; i < colSize; i++) {
	  String colName = col[i];
	  String val = list2.get(i);
	  if (i < colSize-1) {
	    if (val.equals(" ")) {
	      ins.append("NULL, ");
	    } else {
	      ins.append("'").append(val).append("', ");
	    }
	  } else {
	    if (val.equals(" ")) {
	      ins.append("NULL)");
	    } else {
	      ins.append("'").append(val).append("')");
	    }
	  }
	}     
	String insStr = ins.toString();
	if (executeAttendUpdate(insStr) != 1) {
	  cout.println(insStr + "  ... failed");
	} else {
	  ins_count++;
	  // cout.println(insStr);
	  mapOrigin.put(key, list2);
	}
      }
    }
    if (ins_count != 0) {
      cout.println("SERVER_UPDATE_STRUCT: inserted " + ins_count + " lines");
    }
    
    keySet = mapOrigin.keySet();
    for (String key : keySet) {	    
      ArrayList<String> list1 = mapOrigin.get(key);
      ArrayList<String> list2 = mapChanged.get(key);
      if (!compareList(list1, list2)) {
	
	StringBuilder upd = new StringBuilder("update SERVER_UPDATE_STRUCT set ");
	for (int i = 2; i < colSize; i++) {
	  String val = list2.get(i);
	  String colName = col[i];
	  if (i < colSize-1) {
	    if (val.equals(" ")) {
	      upd.append(colName).append(" = null, ");
	    } else {
	      upd.append(colName).append(" = '").append(val).append("', ");
	    }
	  } else {
	    if (val.equals(" ")) {
	      upd.append(colName).append(" = null ");
	    } else {
	      upd.append(colName).append(" = '").append(val).append("' ");
	    }
	  }
	}
	upd.append(" where SERVICE_NAME = '").append(list2.get(0));
	upd.append("' and UPDATE_CODE = '").append(list2.get(1)).append("'");  
	String updStr = upd.toString();	
	if (executeAttendUpdate(updStr) != 1) {
	  cout.println(updStr + "  ... failed");
	} else {
	  upd_count++;
	  // cout.println(updStr); 
	}
      }		
    }
    if (upd_count != 0) {
      cout.println("SERVER_UPDATE_STRUCT: updated " + upd_count + " lines");
    }
  }

  public void readServerInsertStruct (BufferedReader cin,
				   PrintWriter cout) throws Exception { 	
    HashMap<String, ArrayList<String>> mapOrigin = new HashMap<String, ArrayList<String>>();
    HashMap<String, ArrayList<String>> mapChanged = new HashMap<String, ArrayList<String>>();
    int ins_count = 0;
    int upd_count = 0;
    int dlt_count = 0;   
	
    String[] col = { "SERVICE_NAME", "INSERT_CODE", 
		     "DB_NAME", 
		     "INSERT_PARAM_LIST", 
		     "INSERT_EDITOR_LIST", 
		     "WHERE_PARAM_LIST", 
		     "INSERT_SQL", 
		     "SIDE_EFFECT", 
		     "QUALIFICATION_METHOD", 
		     "QUALIFICATION_METHOD_PARAM" };
    int colSize = col.length;

    String query = "select SERVICE_NAME, INSERT_CODE, DB_NAME, INSERT_PARAM_LIST, INSERT_EDITOR_LIST, WHERE_PARAM_LIST, INSERT_SQL, SIDE_EFFECT, QUALIFICATION_METHOD, QUALIFICATION_METHOD_PARAM from SERVER_INSERT_STRUCT order by SERVICE_NAME, INSERT_CODE";  
    ResultSet resultSet = attendStatement.executeQuery(query); 
    while (resultSet.next()) {	
      ArrayList<String> list = new ArrayList<String>();
      String[] tokens = new String[colSize];		
      for (int i = 0; i < colSize; i++) {
	String val = resultSet.getString(i+1);
	if ((val == null) || (val.equals(""))) {
	  tokens[i] = " ";
	} else {
	  if ((i == 6) || (i == 7)) {
	    val = changeQuotationCode(val);
	  }
	  tokens[i] = val.trim();
	}
	list.add(tokens[i]);
      }	  
      String key = tokens[0]+"|"+tokens[1];
      mapOrigin.put(key, list);
    }
    resultSet.close();
	    
    String s;
    while ((s = cin.readLine()) != null) {
      if (s.equals(".")) break;
      if (!s.startsWith("//")) {
	String[] tokens = s.split("\\|");
	if (tokens.length == colSize) {
	  ArrayList<String> list = new ArrayList<String>();
	  for (String token : tokens) {
	    String token2 = changeConcatCode(token);
	    list.add(token2);
	  }	  
	  String key = tokens[0]+"|"+tokens[1];
	  mapChanged.put(key, list);
	} else {
	  cout.println(s);
	}
      }
    }

    Set<String> keySet = mapOrigin.keySet();
    ArrayList<String> listToRemove = new ArrayList<String>();
    for (String key : keySet) {
      if (!mapChanged.containsKey(key)) {
	ArrayList<String> list1 = mapOrigin.get(key);
	StringBuilder del = new StringBuilder();
	del.append("delete from SERVER_INSERT_STRUCT where ");
	for (int i = 0; i < 2; i++) {
	  String colName = col[i];
	  String val = list1.get(i);
	  if (i < 1) {
	    del.append(colName).append(" = '").append(val).append("' and ");
	  } else {
	    del.append(colName).append(" = '").append(val).append("'");
	  }
	}
	String delString = del.toString();
	if (executeAttendUpdate(delString) != 1) {
	  cout.println(delString + "  ... failed");
	} else {
	  dlt_count++;
	  // cout.println(delString);
	  listToRemove.add(key);
	}
      }
    }
    if (dlt_count != 0) {
      cout.println("SERVER_INSERT_STRUCT: deleted " + dlt_count + " lines");
      for (String key : listToRemove) {
	mapOrigin.remove(key);
      }
    }
    
    keySet = mapChanged.keySet();
    for (String key : keySet) {
      if (!mapOrigin.containsKey(key)) {
	ArrayList<String> list2 = mapChanged.get(key);
	StringBuilder ins = new StringBuilder();
	ins.append("insert into SERVER_INSERT_STRUCT (SERVICE_NAME, INSERT_CODE, DB_NAME,INSERT_PARAM_LIST, INSERT_EDITOR_LIST, WHERE_PARAM_LIST, INSERT_SQL, SIDE_EFFECT, QUALIFICATION_METHOD, QUALIFICATION_METHOD_PARAM) values (");
	for (int i = 0; i < colSize; i++) {
	  String colName = col[i];
	  String val = list2.get(i);
	  if (i < colSize-1) {
	    if (val.equals(" ")) {
	      ins.append("NULL, ");
	    } else {
	      ins.append("'").append(val).append("', ");
	    }
	  } else {
	    if (val.equals(" ")) {
	      ins.append("NULL)");
	    } else {
	      ins.append("'").append(val).append("')");
	    }
	  }
	}     
	String insStr = ins.toString();
	if (executeAttendUpdate(insStr) != 1) {
	  cout.println(insStr + "  ... failed");
	} else {
	  ins_count++;
	  // cout.println(insStr);
	  mapOrigin.put(key, list2);
	}
      }
    }
    if (ins_count != 0) {
      cout.println("SERVER_INSERT_STRUCT: inserted " + ins_count + " lines");
    }
    
    keySet = mapOrigin.keySet();
    for (String key : keySet) {	    
      ArrayList<String> list1 = mapOrigin.get(key);
      ArrayList<String> list2 = mapChanged.get(key);
      if (!compareList(list1, list2)) {
	
	StringBuilder upd = new StringBuilder("update SERVER_INSERT_STRUCT set ");
	for (int i = 2; i < colSize; i++) {
	  String val = list2.get(i);
	  String colName = col[i];
	  if (i < colSize-1) {
	    if (val.equals(" ")) {
	      upd.append(colName).append(" = null, ");
	    } else {
	      upd.append(colName).append(" = '").append(val).append("', ");
	    }
	  } else {
	    if (val.equals(" ")) {
	      upd.append(colName).append(" = null ");
	    } else {
	      upd.append(colName).append(" = '").append(val).append("' ");
	    }
	  }
	}
	upd.append(" where SERVICE_NAME = '").append(list2.get(0));
	upd.append("' and INSERT_CODE = '").append(list2.get(1)).append("'");  
	String updStr = upd.toString();	
	if (executeAttendUpdate(updStr) != 1) {
	  cout.println(updStr + "  ... failed");
	} else {
	  upd_count++;
	  // cout.println(updStr); 
	}
      }		
    }
    if (upd_count != 0) {
      cout.println("SERVER_INSERT_STRUCT: updated " + upd_count + " lines");
    }
  }

  public void readServerSpecialStruct (BufferedReader cin,
				   PrintWriter cout) throws Exception { 	
    HashMap<String, ArrayList<String>> mapOrigin = new HashMap<String, ArrayList<String>>();
    HashMap<String, ArrayList<String>> mapChanged = new HashMap<String, ArrayList<String>>();
    int ins_count = 0;
    int upd_count = 0;
    int dlt_count = 0;   
	
    String[] col = { "SERVICE_NAME", "SPECIAL_CODE", 
		     "DB_NAME", 
		     "SPECIAL_PARAM_LIST", 
		     "QUALIFICATION_METHOD", 
		     "QUALIFICATION_METHOD_PARAM" };
    
    int colSize = col.length;

    String query = "select SERVICE_NAME, SPECIAL_CODE, DB_NAME, SPECIAL_PARAM_LIST, QUALIFICATION_METHOD, QUALIFICATION_METHOD_PARAM from SERVER_SPECIAL_STRUCT order by SERVICE_NAME, SPECIAL_CODE";  
    ResultSet resultSet = attendStatement.executeQuery(query); 
    while (resultSet.next()) {	
      ArrayList<String> list = new ArrayList<String>();
      String[] tokens = new String[colSize];		
      for (int i = 0; i < colSize; i++) {
	String val = resultSet.getString(i+1);
	if ((val == null) || (val.equals(""))) {
	  tokens[i] = " ";
	} else {
	  tokens[i] = val.trim();
	}
	list.add(tokens[i]);
      }	  
      String key = tokens[0]+"|"+tokens[1];
      mapOrigin.put(key, list);
    }
    resultSet.close();
	    
    String s;
    while ((s = cin.readLine()) != null) {
      if (s.equals(".")) break;
      if (!s.startsWith("//")) {
	String[] tokens = s.split("\\|");
	if (tokens.length == colSize) {
	  ArrayList<String> list = new ArrayList<String>();
	  for (String token : tokens) {
	    list.add(token);
	  }	  
	  String key = tokens[0]+"|"+tokens[1];
	  mapChanged.put(key, list);
	} else {
	  cout.println(s);
	}
      }
    }

    Set<String> keySet = mapOrigin.keySet();
    ArrayList<String> listToRemove = new ArrayList<String>();
    for (String key : keySet) {
      if (!mapChanged.containsKey(key)) {
	ArrayList<String> list1 = mapOrigin.get(key);
	StringBuilder del = new StringBuilder();
	del.append("delete from SERVER_SPECIAL_STRUCT where ");
	for (int i = 0; i < 2; i++) {
	  String colName = col[i];
	  String val = list1.get(i);
	  if (i < 1) {
	    del.append(colName).append(" = '").append(val).append("' and ");
	  } else {
	    del.append(colName).append(" = '").append(val).append("'");
	  }
	}
	String delString = del.toString();
	if (executeAttendUpdate(delString) != 1) {
	  cout.println(delString + "  ... failed");
	} else {
	  dlt_count++;
	  // cout.println(delString);
	  listToRemove.add(key);
	}
      }
    }
    if (dlt_count != 0) {
      cout.println("SERVER_SPECIAL_STRUCT: deleted " + dlt_count + " lines");
      for (String key : listToRemove) {
	mapOrigin.remove(key);
      }
    }
    
    keySet = mapChanged.keySet();
    for (String key : keySet) {
      if (!mapOrigin.containsKey(key)) {
	ArrayList<String> list2 = mapChanged.get(key);
	StringBuilder ins = new StringBuilder();
	ins.append("insert into SERVER_SPECIAL_STRUCT (SERVICE_NAME, SPECIAL_CODE, DB_NAME, SPECIAL_PARAM_LIST, QUALIFICATION_METHOD, QUALIFICATION_METHOD_PARAM) values (");
	for (int i = 0; i < colSize; i++) {
	  String colName = col[i];
	  String val = list2.get(i);
	  if (i < colSize-1) {
	    if (val.equals(" ")) {
	      ins.append("NULL, ");
	    } else {
	      ins.append("'").append(val).append("', ");
	    }
	  } else {
	    if (val.equals(" ")) {
	      ins.append("NULL)");
	    } else {
	      ins.append("'").append(val).append("')");
	    }
	  }
	}     
	String insStr = ins.toString();
	if (executeAttendUpdate(insStr) != 1) {
	  cout.println(insStr + "  ... failed");
	} else {
	  ins_count++;
	  // cout.println(insStr);
	  mapOrigin.put(key, list2);
	}
      }
    }
    if (ins_count != 0) {
      cout.println("SERVER_SPECIAL_STRUCT: inserted " + ins_count + " lines");
    }
    
    keySet = mapOrigin.keySet();
    for (String key : keySet) {	    
      ArrayList<String> list1 = mapOrigin.get(key);
      ArrayList<String> list2 = mapChanged.get(key);
      if (!compareList(list1, list2)) {	
	StringBuilder upd = new StringBuilder("update SERVER_SPECIAL_STRUCT set ");
	for (int i = 2; i < colSize; i++) {
	  String val = list2.get(i);
	  String colName = col[i];
	  if (i < colSize-1) {
	    if (val.equals(" ")) {
	      upd.append(colName).append(" = null, ");
	    } else {
	      upd.append(colName).append(" = '").append(val).append("', ");
	    }
	  } else {
	    if (val.equals(" ")) {
	      upd.append(colName).append(" = null ");
	    } else {
	      upd.append(colName).append(" = '").append(val).append("' ");
	    }
	  }
	}
	upd.append(" where SERVICE_NAME = '").append(list2.get(0));
	upd.append("' and SPECIAL_CODE = '").append(list2.get(1)).append("'");  
	String updStr = upd.toString();
	
	if (executeAttendUpdate(updStr) != 1) {
	  cout.println(updStr + "  ... failed");
	} else {
	  upd_count++;
	  // cout.println(updStr); 
	}
      }		
    }
    if (upd_count != 0) {
      cout.println("SERVER_SPECIAL_STRUCT: updated " + upd_count + " lines");
    }
  }

  public void readServerQueryStruct (BufferedReader cin,
				   PrintWriter cout) throws Exception { 	
    HashMap<String, ArrayList<String>> mapOrigin = new HashMap<String, ArrayList<String>>();
    HashMap<String, ArrayList<String>> mapChanged = new HashMap<String, ArrayList<String>>();
    int ins_count = 0;
    int upd_count = 0;
    int dlt_count = 0;  
	
    String[] col = { "SERVICE_NAME", 
		     "PANEL_ID", 
		     "SWITCH_FLAG", 
		     "DB_NAME", 
		     "PARAM_LIST", 
		     "QUERY_SQL", 
		     "QUALIFICATION_METHOD", 
		     "QUALIFICATION_METHOD_PARAM" };    
    int colSize = col.length;

    String query = "select SERVICE_NAME, PANEL_ID, SWITCH_FLAG, DB_NAME, PARAM_LIST, QUERY_SQL, QUALIFICATION_METHOD, QUALIFICATION_METHOD_PARAM from SERVER_QUERY_STRUCT order by SERVICE_NAME, PANEL_ID";  
    ResultSet resultSet = attendStatement.executeQuery(query); 
    while (resultSet.next()) {	
      ArrayList<String> list = new ArrayList<String>();
      String[] tokens = new String[colSize];		
      for (int i = 0; i < colSize; i++) {
	String val = resultSet.getString(i+1);
	if ((val == null) || (val.equals(""))) {
	  tokens[i] = " ";
	} else {
	  if (i == 5) {
	    val = changeQuotationCode(val); 
	  }
	  tokens[i] = val.trim();
	}
	list.add(tokens[i]);
      }	  
      String key = tokens[0]+"|"+tokens[1]+"|"+tokens[2];
      mapOrigin.put(key, list);
    }
    resultSet.close();
	    
    String s;
    while ((s = cin.readLine()) != null) {
      if (s.equals(".")) break;
      if (!s.startsWith("//")) {
	String[] tokens = s.split("\\|");
	if (tokens.length == colSize) {
	  ArrayList<String> list = new ArrayList<String>();
	  for (String token : tokens) {
	    String token2 = changeConcatCode(token);
	    list.add(token2);
	  }	  
	  String key = tokens[0]+"|"+tokens[1]+"|"+tokens[2];
	  mapChanged.put(key, list);
	} else {
	  cout.println(s);
	}
      }
    }

    Set<String> keySet = mapOrigin.keySet();
    ArrayList<String> listToRemove = new ArrayList<String>();
    for (String key : keySet) {
      if (!mapChanged.containsKey(key)) {
	ArrayList<String> list1 = mapOrigin.get(key);
	StringBuilder del = new StringBuilder();
	del.append("delete from SERVER_QUERY_STRUCT where ");
	for (int i = 0; i < 3; i++) {
	  String colName = col[i];
	  String val = list1.get(i);
	  if (i < 2) {
	    del.append(colName).append(" = '").append(val).append("' and ");
	  } else {
	    del.append(colName).append(" = '").append(val).append("'");
	  }
	}
	String delString = del.toString();
	if (executeAttendUpdate(delString) != 1) {
	  cout.println(delString + "  ... failed");
	} else {
	  dlt_count++;
	  // cout.println(delString);
	  listToRemove.add(key);
	}
      }
    }
    if (dlt_count != 0) {
      cout.println("SERVER_QUERY_STRUCT: deleted " + dlt_count + " lines");
      for (String key : listToRemove) {
	mapOrigin.remove(key);
      }
    }
    
    keySet = mapChanged.keySet();
    for (String key : keySet) {
      if (!mapOrigin.containsKey(key)) {
	ArrayList<String> list2 = mapChanged.get(key);
	StringBuilder ins = new StringBuilder();
	ins.append("insert into SERVER_QUERY_STRUCT (SERVICE_NAME, PANEL_ID, SWITCH_FLAG, DB_NAME, PARAM_LIST, QUERY_SQL, QUALIFICATION_METHOD, QUALIFICATION_METHOD_PARAM) values (");
	for (int i = 0; i < colSize; i++) {
	  String colName = col[i];
	  String val = list2.get(i);
	  if (i < colSize-1) {
	    if (val.equals(" ")) {
	      ins.append("NULL, ");
	    } else {
	      ins.append("'").append(val).append("', ");
	    }
	  } else {
	    if (val.equals(" ")) {
	      ins.append("NULL)");
	    } else {
	      ins.append("'").append(val).append("')");
	    }
	  }
	}     
	String insStr = ins.toString();
	if (executeAttendUpdate(insStr) != 1) {
	  cout.println(insStr + "  ... failed");
	} else {
	  ins_count++;
	  // cout.println(insStr);
	  mapOrigin.put(key, list2);
	}
      }
    }
    if (ins_count != 0) {
      cout.println("SERVER_QUERY_STRUCT: inserted " + ins_count + " lines");
    }
    
    keySet = mapOrigin.keySet();
    for (String key : keySet) {	    
      ArrayList<String> list1 = mapOrigin.get(key);
      ArrayList<String> list2 = mapChanged.get(key);
      if (!compareList(list1, list2)) {	
	StringBuilder upd = new StringBuilder("update SERVER_QUERY_STRUCT set ");
	for (int i = 3; i < colSize; i++) {
	  String val = list2.get(i);
	  String colName = col[i];
	  if (i < colSize-1) {
	    if (val.equals(" ")) {
	      upd.append(colName).append(" = null, ");
	    } else {
	      upd.append(colName).append(" = '").append(val).append("', ");
	    }
	  } else {
	    if (val.equals(" ")) {
	      upd.append(colName).append(" = null ");
	    } else {
	      upd.append(colName).append(" = '").append(val).append("' ");
	    }
	  }
	}
	upd.append(" where SERVICE_NAME = '").append(list2.get(0));
	upd.append("' and PANEL_ID = '").append(list2.get(1));
	upd.append("' and SWITCH_FLAG = '").append(list2.get(2)).append("'");  
	String updStr = upd.toString();	
	if (executeAttendUpdate(updStr) != 1) {
	  cout.println(updStr + "  ... failed");
	} else {
	  upd_count++;
	  // cout.println(updStr); 
	}
      }		
    }
    if (upd_count != 0) {
      cout.println("SERVER_QUERY_STRUCT: updated " + upd_count + " lines");
    }
  }

  public void readCommonQueryStruct (BufferedReader cin,
				   PrintWriter cout) throws Exception { 	
    HashMap<String, ArrayList<String>> mapOrigin = new HashMap<String, ArrayList<String>>();
    HashMap<String, ArrayList<String>> mapChanged = new HashMap<String, ArrayList<String>>();
    int ins_count = 0;
    int upd_count = 0;
    int dlt_count = 0;   
	
    String[] col = { "QUERY_NAME", "DB_NAME", 
		     "PARAM_LIST", "QUERY_SQL"  };
    int colSize = col.length;

    String query = "select QUERY_NAME, DB_NAME, PARAM_LIST, QUERY_SQL from COMMON_QUERY_STRUCT order by QUERY_NAME";  
    ResultSet resultSet = attendStatement.executeQuery(query); 
    while (resultSet.next()) {	
      ArrayList<String> list = new ArrayList<String>();
      String[] tokens = new String[colSize];		
      for (int i = 0; i < colSize; i++) {
	String val = resultSet.getString(i+1);
	if ((val == null) || (val.equals(""))) {
	  tokens[i] = " ";
	} else {
	  if (i == 3) {
	    val = changeQuotationCode(val);
	  }
	  tokens[i] = val.trim();
	}
	list.add(tokens[i]);
      }	  
      String key = tokens[0];
      mapOrigin.put(key, list);
    }
    resultSet.close();
	    
    String s;
    while ((s = cin.readLine()) != null) {
      if (s.equals(".")) break;
      if (!s.startsWith("//")) {
	String[] tokens = s.split("\\|");
	if (tokens.length == colSize) {
	  ArrayList<String> list = new ArrayList<String>();
	  for (String token : tokens) {
	    String token2 = changeConcatCode(token);
	    list.add(token2);
	  }	  
	  String key = tokens[0];
	  mapChanged.put(key, list);
	} else {
	  cout.println(s);
	}
      }
    }

    Set<String> keySet = mapOrigin.keySet();
    ArrayList<String> listToRemove = new ArrayList<String>();
    for (String key : keySet) {
      if (!mapChanged.containsKey(key)) {
	ArrayList<String> list1 = mapOrigin.get(key);
	StringBuilder del = new StringBuilder();
	del.append("delete from COMMON_QUERY_STRUCT where ");
	String colName = col[0];
	String val = list1.get(0);
	del.append(colName).append(" = '").append(val).append("'");
	String delString = del.toString();
	if (executeAttendUpdate(delString) != 1) {
	  cout.println(delString + "  ... failed");
	} else {
	  dlt_count++;
	  // cout.println(delString);
	  listToRemove.add(key);
	}
      }
    }
    if (dlt_count != 0) {
      cout.println("COMMON_QUERY_STRUCT: deleted " + dlt_count + " lines");
      for (String key : listToRemove) {
	mapOrigin.remove(key);
      }
    }
    
    keySet = mapChanged.keySet();
    for (String key : keySet) {
      if (!mapOrigin.containsKey(key)) {
	ArrayList<String> list2 = mapChanged.get(key);
	StringBuilder ins = new StringBuilder();
	ins.append("insert into COMMON_QUERY_STRUCT (QUERY_NAME, DB_NAME, PARAM_LIST, QUERY_SQL) values (");
	for (int i = 0; i < colSize; i++) {
	  String colName = col[i];
	  String val = list2.get(i);
	  if (i < colSize-1) {
	    if (val.equals(" ")) {
	      ins.append("NULL, ");
	    } else {
	      ins.append("'").append(val).append("', ");
	    }
	  } else {
	    if (val.equals(" ")) {
	      ins.append("NULL)");
	    } else {
	      ins.append("'").append(val).append("')");
	    }
	  }
	}     
	String insStr = ins.toString();
	if (executeAttendUpdate(insStr) != 1) {
	  cout.println(insStr + "  ... failed");
	} else {
	  ins_count++;
	  // cout.println(insStr);
	  mapOrigin.put(key, list2);
	}
      }
    }
    if (ins_count != 0) {
      cout.println("COMMON_QUERY_STRUCT: inserted " + ins_count + " lines");
    }
		
    keySet = mapOrigin.keySet();
    for (String key : keySet) {	    
      ArrayList<String> list1 = mapOrigin.get(key);
      ArrayList<String> list2 = mapChanged.get(key);
      if (!compareList(list1, list2)) {	
	StringBuilder upd = new StringBuilder("update COMMON_QUERY_STRUCT set ");
	for (int i = 1; i < colSize; i++) {
	  String val = list2.get(i);
	  String colName = col[i];
	  if (i < colSize-1) {
	    if (val.equals(" ")) {
	      upd.append(colName).append(" = null, ");
	    } else {
	      upd.append(colName).append(" = '").append(val).append("', ");
	    }
	  } else {
	    if (val.equals(" ")) {
	      upd.append(colName).append(" = null ");
	    } else {
	      upd.append(colName).append(" = '").append(val).append("' ");
	    }
	  }
	}
	upd.append(" where QUERY_NAME = '").append(list2.get(0)).append("'");  
	String updStr = upd.toString();
	
	if (executeAttendUpdate(updStr) != 1) {
	  cout.println(updStr + "  ... failed");
	} else {
	  upd_count++;
	  // cout.println(updStr); 
	}
      }		
    }
    if (upd_count != 0) {
      cout.println("COMMON_QUERY_STRUCT: updated " + upd_count + " lines");
    }
  }

  public void readGakumuCodeDef (BufferedReader cin,
				 PrintWriter cout) throws Exception { 	
    HashMap<String, ArrayList<String>> mapOrigin = new HashMap<String, ArrayList<String>>();
    HashMap<String, ArrayList<String>> mapChanged = new HashMap<String, ArrayList<String>>();
    int ins_count = 0;
    int upd_count = 0;
    int dlt_count = 0; 

    String[] col = { "CODE_CATEGORY", "GAKUMU_CODE", 
		     "GAKUMU_NAME", 
		     "SHORTER_NAME", 
		     "REMARK" };    
    int colSize = col.length;

    String query = "select CODE_CATEGORY, GAKUMU_CODE, GAKUMU_NAME, SHORTER_NAME, REMARK  from GAKUMU_CODE_DEF order by CODE_CATEGORY, GAKUMU_CODE";  
    ResultSet resultSet = attendStatement.executeQuery(query); 
    while (resultSet.next()) {	
      ArrayList<String> list = new ArrayList<String>();
      String[] tokens = new String[colSize];		
      for (int i = 0; i < colSize; i++) {
	String val = resultSet.getString(i+1);
	if ((val == null) || (val.equals(""))) {
	  tokens[i] = " ";
	} else {
	  tokens[i] = val.trim();
	}
	list.add(tokens[i]);
      }	  
      String key = tokens[0]+"|"+tokens[1];
      mapOrigin.put(key, list);
    }
    resultSet.close();
	    
    String s;
    while ((s = cin.readLine()) != null) {
      if (s.equals(".")) break;
      if (!s.startsWith("//")) {
	String[] tokens = s.split("\\|");
	if (tokens.length == colSize) {
	  ArrayList<String> list = new ArrayList<String>();
	  for (String token : tokens) {
	    list.add(token);
	  }	  
	  String key = tokens[0]+"|"+tokens[1];
	  mapChanged.put(key, list);
	} else {
	  cout.println(s);
	}
      }
    }

    Set<String> keySet = mapOrigin.keySet();
    ArrayList<String> listToRemove = new ArrayList<String>();
    for (String key : keySet) {
      if (!mapChanged.containsKey(key)) {
	ArrayList<String> list1 = mapOrigin.get(key);
	StringBuilder del = new StringBuilder();
	del.append("delete from GAKUMU_CODE_DEF where ");
	for (int i = 0; i < 2; i++) {
	  String colName = col[i];
	  String val = list1.get(i);
	  if (i < 1) {
	    del.append(colName).append(" = '").append(val).append("' and ");
	  } else {
	    del.append(colName).append(" = '").append(val).append("'");
	  }
	}
	String delString = del.toString();
	if (executeAttendUpdate(delString) != 1) {
	  cout.println(delString + "  ... failed");
	} else {
	  dlt_count++;
	  // cout.println(delString);
	  listToRemove.add(key);
	}
      }
    }
    if (dlt_count != 0) {
      cout.println("GAKUMU_CODE_DEF: deleted " + dlt_count + " lines");
      for (String key : listToRemove) {
	mapOrigin.remove(key);
      }
    }
    
    keySet = mapChanged.keySet();
    for (String key : keySet) {
      if (!mapOrigin.containsKey(key)) {
	ArrayList<String> list2 = mapChanged.get(key);
	StringBuilder ins = new StringBuilder();
	ins.append("insert into GAKUMU_CODE_DEF (CODE_CATEGORY, GAKUMU_CODE, GAKUMU_NAME, SHORTER_NAME, REMARK) values (");
	for (int i = 0; i < colSize; i++) {
	  String colName = col[i];
	  String val = list2.get(i);
	  if (i < colSize-1) {
	    if (val.equals(" ")) {
	      ins.append("NULL, ");
	    } else {
	      ins.append("'").append(val).append("', ");
	    }
	  } else {
	    if (val.equals(" ")) {
	      ins.append("NULL)");
	    } else {
	      ins.append("'").append(val).append("')");
	    }
	  }
	}     
	String insStr = ins.toString();
	if (executeAttendUpdate(insStr) != 1) {
	  cout.println(insStr + "  ... failed");
	} else {
	  ins_count++;
	  // cout.println(insStr);
	  mapOrigin.put(key, list2);
	}
      }
    }
    if (ins_count != 0) {
      cout.println("GAKUMU_CODE_DEF: inserted " + ins_count + " lines");
    }
    
    keySet = mapOrigin.keySet();
    for (String key : keySet) {	    
      ArrayList<String> list1 = mapOrigin.get(key);
      ArrayList<String> list2 = mapChanged.get(key);
      if (!compareList(list1, list2)) {
	StringBuilder upd = new StringBuilder("update GAKUMU_CODE_DEF set ");
	for (int i = 2; i < colSize; i++) {
	  String val = list2.get(i);
	  String colName = col[i];
	  if (i < colSize-1) {
	    if (val.equals(" ")) {
	      upd.append(colName).append(" = null, ");
	    } else {
	      upd.append(colName).append(" = '").append(val).append("', ");
	    }
	  } else {
	    if (val.equals(" ")) {
	      upd.append(colName).append(" = null ");
	    } else {
	      upd.append(colName).append(" = '").append(val).append("' ");
	    }
	  }
	}
	upd.append(" where CODE_CATEGORY = '").append(list2.get(0));
	upd.append("' and GAKUMU_CODE = '").append(list2.get(1)).append("'");  
	String updStr = upd.toString();
	if (executeAttendUpdate(updStr) != 1) {
	  cout.println(updStr + "  ... failed");
	} else {
	  upd_count++;
	  // cout.println(updStr); 
	}
      }		
    }
    if (upd_count != 0) {
      cout.println("GAKUMU_CODE_DEF: updated " + upd_count + " lines");
    }
  }

  public void readColorNameDef (BufferedReader cin,
				   PrintWriter cout) throws Exception { 	
    HashMap<String, ArrayList<String>> mapOrigin = new HashMap<String, ArrayList<String>>();
    HashMap<String, ArrayList<String>> mapChanged = new HashMap<String, ArrayList<String>>();
    int ins_count = 0;
    int upd_count = 0;
    int dlt_count = 0;  
	
    String[] col = { "COLOR_NAME", 
		     "RED", 
		     "GREEN", 
		     "BLUE" };
    int colSize = col.length;

    String query = "select COLOR_NAME, RED, GREEN, BLUE from COLOR_NAME_DEF order by COLOR_NAME";  
    ResultSet resultSet = attendStatement.executeQuery(query); 
    while (resultSet.next()) {	
      ArrayList<String> list = new ArrayList<String>();
      String[] tokens = new String[colSize];		
      for (int i = 0; i < colSize; i++) {
	String val = resultSet.getString(i+1);
	if ((val == null) || (val.equals(""))) {
	  tokens[i] = " ";
	} else {
	  tokens[i] = val.trim();
	}
	list.add(tokens[i]);
      }	  
      String key = tokens[0];
      mapOrigin.put(key, list);
    }
    resultSet.close();
	    
    String s;
    while ((s = cin.readLine()) != null) {
      if (s.equals(".")) break;
      if (!s.startsWith("//")) {
	String[] tokens = s.split("\\|");
	if (tokens.length == colSize) {
	  ArrayList<String> list = new ArrayList<String>();
	  for (String token : tokens) {
	    list.add(token);
	  }	  
	  String key = tokens[0];
	  mapChanged.put(key, list);
	} else {
	  cout.println(s);
	}
      }
    }

    Set<String> keySet = mapOrigin.keySet();
    ArrayList<String> listToRemove = new ArrayList<String>();
    for (String key : keySet) {
      if (!mapChanged.containsKey(key)) {
	ArrayList<String> list1 = mapOrigin.get(key);
	StringBuilder del = new StringBuilder();
	del.append("delete from COLOR_NAME_DEF where ");
	String colName = col[0];
	String val = list1.get(0);
	del.append(colName).append(" = '").append(val).append("'");
	
	String delString = del.toString();
	if (executeAttendUpdate(delString) != 1) {
	  cout.println(delString + "  ... failed");
	} else {
	  dlt_count++;
	  // cout.println(delString);
	  listToRemove.add(key);
	}
      }
    }
    if (dlt_count != 0) {
      cout.println("COLOR_NAME_DEF: deleted " + dlt_count + " lines");
      for (String key : listToRemove) {
	mapOrigin.remove(key);
      }
    }
    
    keySet = mapChanged.keySet();
    for (String key : keySet) {
      if (!mapOrigin.containsKey(key)) {
	ArrayList<String> list2 = mapChanged.get(key);
	StringBuilder ins = new StringBuilder();
	ins.append("insert into COLOR_NAME_DEF (COLOR_NAME, RED, GREEN, BLUE) values (");
	for (int i = 0; i < colSize; i++) {
	  String colName = col[i];
	  String val = list2.get(i);
	  if (i < colSize-1) {
	    if (val.equals(" ")) {
	      ins.append("NULL, ");
	    } else {
	      ins.append("'").append(val).append("', ");
	    }
	  } else {
	    if (val.equals(" ")) {
	      ins.append("NULL)");
	    } else {
	      ins.append("'").append(val).append("')");
	    }
	  }
	}     
	String insStr = ins.toString();
	if (executeAttendUpdate(insStr) != 1) {
	  cout.println(insStr + "  ... failed");
	} else {
	  ins_count++;
	  // cout.println(insStr);
	  mapOrigin.put(key, list2);
	}
      }
    }
    if (ins_count != 0) {
      cout.println("COLOR_NAME_DEF: inserted " + ins_count + " lines");
    }
    
    keySet = mapOrigin.keySet();
    for (String key : keySet) {	    
      ArrayList<String> list1 = mapOrigin.get(key);
      ArrayList<String> list2 = mapChanged.get(key);
      if (!compareList(list1, list2)) {	
	StringBuilder upd = new StringBuilder("update COLOR_NAME_DEF set ");
	for (int i = 1; i < colSize; i++) {
	  String val = list2.get(i);
	  String colName = col[i];
	  if (i < colSize-1) {
	    if (val.equals(" ")) {
	      upd.append(colName).append(" = null, ");
	    } else {
	      upd.append(colName).append(" = '").append(val).append("', ");
	    }
	  } else {
	    if (val.equals(" ")) {
	      upd.append(colName).append(" = null ");
	    } else {
	      upd.append(colName).append(" = '").append(val).append("' ");
	    }
	  }
	}
	upd.append(" where COLOR_NAME = '").append(list2.get(0)).append("'");
	String updStr = upd.toString();	
	if (executeAttendUpdate(updStr) != 1) {
	  cout.println(updStr + "  ... failed");
	} else {
	  upd_count++;
	  // cout.println(updStr); 
	}
      }		
    }
    if (upd_count != 0) {
      cout.println("COLOR_NAME_DEF: updated " + upd_count + " lines");
    }
  }
} 
