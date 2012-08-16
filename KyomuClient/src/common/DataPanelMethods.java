package common;

import clients.*;
import java.util.*;
import java.awt.*;
import java.io.*;
//import java.awt.event.*;
import javax.swing.*;
import javax.swing.text.*;
import javax.swing.border.*;
//import java.lang.reflect.*; 

public class DataPanelMethods {
  private String serviceName;
  private String nodePath;
  private CommonInfo commonInfo;
  private CommonInfoMethods commonMethods;
  private TabbedPaneBase parentTabbedPane;
  private DataPanelBase dataPanel;
  private TablePrinter tablePrinter;
  private JFileChooser fileChooser;
  private String lineSeparator = System.getProperty("line.separator");

  protected TableViewBase tableView = null;
  protected ReportTableView reportTableView = null;  
  protected NinteiTableView ninteiTableView = null;
  protected RegistrTableView registrTableView = null;
  protected StructTableView structTableView = null;
  protected GakusekiView gakusekiView = null;
  protected HtmlViewBase htmlView = null;
  protected JikanwariViewBase jikanwariView = null;
  protected PhotoViewBase photoView = null;
//  protected MailViewBase mailView = null;
  protected SyllabusView syllabusView = null;
  protected SyllabusView2 syllabusView2 = null;
  protected SyllabusEdit syllabusEdit = null;

  public DataPanelMethods(String serviceName, String nodePath,
			  CommonInfo commonInfo,
			  TabbedPaneBase parentTabbedPane,
			  DataPanelBase dataPanel) {
    this.serviceName = serviceName;
    this.nodePath = nodePath;
    this.commonInfo = commonInfo;
    this.parentTabbedPane = parentTabbedPane;
    this.dataPanel = dataPanel;
    this.tablePrinter = dataPanel.tablePrinter;
    this.commonMethods = commonInfo.commonInfoMethods;
  }  

  public String packZero(String str, int n) {
    StringBuffer sbuf = new StringBuffer();
    int m = str.length();
    if (n > m) {
      for (int i = m; i < n; i++) {
	sbuf.append("0");
      }
      sbuf.append(str);
      return sbuf.toString();
    } else {
      return str;
    }
  }

  public String packSpace(String str, int n) {
    StringBuffer sbuf = new StringBuffer();
    int m = str.length();
    if (n > m) {
      for (int i = m; i < n; i++) {
	sbuf.append(" ");
      }
      sbuf.append(str);
      return sbuf.toString();
    } else {
      return str;
    }
  }


  //******** Methods invoked by Simple Button Pressed ****************/////

  public void openTab(String nodeID) { 
    parentTabbedPane.openTab(nodeID);
  }  

  public void openOldPage(String param) { 
    parentTabbedPane.openOldTab();
  }

  public void setStructTableInfoAndOpenTab(String nodeID) {
    String[] columnName = { "COLUMN_NAME", "COLUMN_VALUE" };
    for (int i = 0; i < tableView.getRowCount(); i++) { 
      String key = tableView.getCodeAt(i, columnName[0]);
      String value = tableView.getCodeAt(i, columnName[1]);
      parentTabbedPane.addColumnCodeMap(key, value, value);
    }
    parentTabbedPane.openTab(nodeID);
  }  

  public void refreshTable(String param) {    
    tableView.refreshTable();
  }

  public void showSelectedChildInfo(String nodeID2) { 
    String serviceName = parentTabbedPane.getValueFromColumnCodeMap("SERVICE_NAME");
    String nodePath = parentTabbedPane.getValueFromColumnCodeMap("NODE_PATH");
    String componentType = parentTabbedPane.getValueFromColumnCodeMap("CHILD_COMPONENT_TYPE");
    String nodeID = parentTabbedPane.getValueFromColumnCodeMap("CHILD_NODE_ID");
    String panelID = parentTabbedPane.getValueFromColumnCodeMap("CHILD_PANEL_ID");

    if (componentType.equals("TabbedPane")) {
      if (!(nodeID.startsWith("/"))) {
	String newNodePath = nodePath + "." + nodeID;
	parentTabbedPane.addColumnCodeMap("NODE_PATH", newNodePath, newNodePath);  
	tableView.refreshTable();
      } else {
	int pos = nodeID.indexOf(".");
	String newServiceName = nodeID.substring(1, pos);
	String newNodePath = nodeID.substring(pos+1); 
	parentTabbedPane.addColumnCodeMap("SERVICE_NAME", newServiceName, newServiceName);
	parentTabbedPane.addColumnCodeMap("NODE_PATH", newNodePath, newNodePath); 
	tableView.refreshTable();
      } 
    } else if (componentType.equals("DataPanel")) {
      parentTabbedPane.openTab(nodeID2);
    }    
  }

  public void showParentTabInfo(String param) { 
    String serviceName = parentTabbedPane.getValueFromColumnCodeMap("SERVICE_NAME");
    String nodePath = parentTabbedPane.getValueFromColumnCodeMap("NODE_PATH");
    if (!nodePath.equals("root")) {
      int pos = nodePath.lastIndexOf(".");
      if (pos >= 0) {
	String newNodePath = nodePath.substring(0, pos);
	parentTabbedPane.addColumnCodeMap("NODE_PATH", newNodePath, newNodePath); 
	tableView.refreshTable();
      }
    }
  }

  public void openStructTableView(String nodeID) {
    if (tableView.getSelectedRowCount() == 0) {
      int rowCount = tableView.getRowCount();
      if (rowCount == 1) {
	tableView.rowSelectAction(0);
      }
    }
    parentTabbedPane.openTab(nodeID);
  }

  public void deleteStructTableContent(String parameter) {
    String[] tokens  = parameter.split("\\:");
    String deleteServiceName = tokens[0];
    String deleteCode = tokens[1];
    String delParams = commonInfo.getServerDeleteParams(tokens[0], tokens[1]);
    String[] ttokens =  delParams.split("\\|");    
    String[] whereParams = ttokens[0].split("\\:");
    HashSet<String> whereSet = new HashSet<String>();
    for (String param : whereParams) {
      whereSet.add(param);
    }    
    String[] columnName = { "COLUMN_NAME", "COLUMN_VALUE" };
    HashMap<String, String> valueMap = new HashMap<String, String>();
    ArrayList<Object> list = new ArrayList<Object>();
    list.add(" データ表の削除データの確認です。                           ");
    list.add(" UNDO 機能はないので削除を実行する前に十分に確認して下さい。");
    for (int i = 0; i < tableView.getRowCount(); i++) { 
      String key = tableView.getCodeAt(i, columnName[0]);
      String value = tableView.getCodeAt(i, columnName[1]);
      String text = " " + key + " = " + value;
      valueMap.put(key, value);
      if (whereSet.contains(key)) {
	JTextField editor = new JTextField(text);
	editor.setEditable(false); 
	editor.setFont(new Font("DialogInput", Font.PLAIN, 12));  
	editor.setBackground(Color.yellow);
	list.add(editor);
      }
    }
    int ans =  JOptionPane.showOptionDialog(commonInfo.getFrame(),
					    list.toArray(), 
					    "削除データの設定", 
					    JOptionPane.OK_CANCEL_OPTION,
					    JOptionPane.QUESTION_MESSAGE,
					    null,
					    null,
					    null);
    if (ans == JOptionPane.OK_OPTION) {
      StringBuilder paramValues = new StringBuilder();   
      for (String param : whereParams) {
	String value = valueMap.get(param);
	if ((value == null) || (value.trim().equals(""))) {
	  paramValues.append(" |"); 
	} else {
	  paramValues.append(value.trim()).append("|"); 
	}
      }
      int ret = commonInfo.getDeleteResult(deleteServiceName, deleteCode, 
					   paramValues.toString());
      if (ret != 1) {
	commonInfo.showMessage("データの削除に失敗しました");
      } else {
	tableView.refreshTable();
      }
    } 
  }

  public void updateStructTableContent(String parameter) {
    String[] tokens  = parameter.split("\\:");
    String updateServiceName = tokens[0];
    String updateCode = tokens[1];
    String updateParams = commonInfo.getServerUpdateParams(tokens[0], tokens[1]);
    String[] ttokens = updateParams.split("\\|");    
    String[] setParams = ttokens[0].split("\\:");
    String[] whereParams = ttokens[2].split("\\:");
    HashSet<String> setSet = new HashSet<String>();
    HashSet<String> whereSet = new HashSet<String>();
    for (String param : setParams) {
      setSet.add(param);
    }
    for (String param : whereParams) {
      whereSet.add(param);
    }
    String[] columnName = { "COLUMN_NAME", "COLUMN_VALUE" };
    HashMap<String, String> valueMap = new HashMap<String, String>();
    HashMap<String, JTextComponent> editorMap = new HashMap<String, JTextComponent>();
    ArrayList<Object> list = new ArrayList<Object>();
    list.add(" データ表の更新データを設定して下さい。                     ");
    list.add(" UNDO 機能はないので更新を実行する前に十分に確認して下さい。");
    for (int i = 0; i < tableView.getRowCount(); i++) { 
      String key = tableView.getCodeAt(i, columnName[0]);
      String value = tableView.getCodeAt(i, columnName[1]);
      String text = " " + key + " = " + value;
      valueMap.put(key, value);
      if (whereSet.contains(key)) {
	JTextField editor = new JTextField(text);
	editor.setEditable(false); 
	editor.setFont(new Font("DialogInput", Font.PLAIN, 12));  
	editor.setBackground(Color.yellow);
	list.add(editor);
      } else if (setSet.contains(key)) {
	JTextField editor = new JTextField(100);
	editor.setFont(new Font("DialogInput", Font.PLAIN, 12));
	editor.setText("  " + value);
	editor.setEditable(true);
	TitledBorder border = new TitledBorder(null, text, TitledBorder.LEFT, TitledBorder.TOP,
					       new Font("DialogInput", Font.PLAIN, 10));
	editor.setBorder(border);
	JScrollPane scroll = new JScrollPane(editor);
	list.add(scroll);
	editorMap.put(key, editor);
      }
    }
    int ans =  JOptionPane.showOptionDialog(commonInfo.getFrame(),
					    list.toArray(), 
					    "更新値の設定", 
					    JOptionPane.OK_CANCEL_OPTION,
					    JOptionPane.QUESTION_MESSAGE,
					    null,
					    null,
					    null);
    if (ans == JOptionPane.OK_OPTION) {
      StringBuilder paramValues = new StringBuilder();      
      for (String param : setParams) {
	JTextComponent editor = editorMap.get(param);
	String value = editor.getText();
	if ((value == null) || (value.trim().equals(""))) {
	  paramValues.append(" |"); 
	} else {
	  paramValues.append(value.trim()).append("|"); 
	}
      }
      for (String param : whereParams) {
	String value = valueMap.get(param);
	if ((value == null) || (value.trim().equals(""))) {
	  paramValues.append(" |"); 
	} else {
	  paramValues.append(value.trim()).append("|"); 
	}
      }

      int ret = commonInfo.getUpdateResult(updateServiceName, updateCode, 
					   paramValues.toString());
      if (ret != 1) {
	commonInfo.showMessage("データの更新に失敗しました");
      } else {
	tableView.refreshTable();
      }
    } 
  }
 
  public void updateStructTableSQL(String parameter) {
    String[] tokens  = parameter.split("\\:");
    String updateServiceName = tokens[0];
    String updateCode = tokens[1];
    String updateParams = commonInfo.getServerUpdateParams(tokens[0], tokens[1]);
    String[] ttokens = updateParams.split("\\|");    
    String[] setParams = ttokens[0].split("\\:");
    String[] whereParams = ttokens[2].split("\\:");
    HashSet<String> setSet = new HashSet<String>();
    HashSet<String> whereSet = new HashSet<String>();
    for (String param : setParams) {
      setSet.add(param);
    }
    for (String param : whereParams) {
      whereSet.add(param);
    }
    String[] columnName = { "COLUMN_NAME", "COLUMN_VALUE" };
    HashMap<String, String> valueMap = new HashMap<String, String>();
    HashMap<String, JTextComponent> editorMap = new HashMap<String, JTextComponent>();
    ArrayList<Object> list = new ArrayList<Object>();
    list.add(" データ表の更新データを設定して下さい。                     ");
    list.add(" UNDO 機能はないので更新を実行する前に十分に確認して下さい。");
    for (int i = 0; i < tableView.getRowCount(); i++) { 
      String key = tableView.getCodeAt(i, columnName[0]);
      String value = tableView.getCodeAt(i, columnName[1]);
      String text = " " + key + " = " + value;
      valueMap.put(key, value);
      if (whereSet.contains(key)) {
	JTextField editor = new JTextField(text);
	editor.setEditable(false); 
	editor.setFont(new Font("DialogInput", Font.PLAIN, 12));  
	editor.setBackground(Color.yellow);
	list.add(editor);
      } else if (setSet.contains(key)) {
	if ((key.equals("SIDE_EFFECT")) || (key.indexOf("SQL") > 0)) {
	  JTextArea editor = new JTextArea(4, 95);
	  editor.setFont(new Font("DialogInput", Font.PLAIN, 12));
	  editor.setLineWrap(true);
	  editor.setText(value);
	  editor.setEditable(true);
	  TitledBorder border = new TitledBorder(null, key, TitledBorder.LEFT, TitledBorder.TOP,
						 new Font("DialogInput", Font.PLAIN, 10));
	  editor.setBorder(border);
	  JScrollPane scroll = new JScrollPane(editor);
	  list.add(scroll);
	  editorMap.put(key, editor);
	} else {
	  JTextField editor = new JTextField(100);
	  editor.setFont(new Font("DialogInput", Font.PLAIN, 12));
	  editor.setText("  " + value);
	  editor.setEditable(true);
	  TitledBorder border = new TitledBorder(null, text, TitledBorder.LEFT, TitledBorder.TOP,
						 new Font("DialogInput", Font.PLAIN, 10));
	  editor.setBorder(border);
	  JScrollPane scroll = new JScrollPane(editor);
	  list.add(scroll);
	  editorMap.put(key, editor);
	}
      }
    }
    int ans =  JOptionPane.showOptionDialog(commonInfo.getFrame(),
					    list.toArray(), 
					    "更新値の設定", 
					    JOptionPane.OK_CANCEL_OPTION,
					    JOptionPane.QUESTION_MESSAGE,
					    null,
					    null,
					    null);
    if (ans == JOptionPane.OK_OPTION) {
      StringBuilder paramValues = new StringBuilder();      
      for (String param : setParams) {
	JTextComponent editor = editorMap.get(param);
	String value = editor.getText();
	if ((value == null) || (value.trim().equals(""))) {
	  paramValues.append(" |"); 
	} else {
	  paramValues.append(value.trim()).append("|"); 
	}
      }
      for (String param : whereParams) {
	String value = valueMap.get(param);
	if ((value == null) || (value.trim().equals(""))) {
	  paramValues.append(" |"); 
	} else {
	  paramValues.append(value.trim()).append("|"); 
	}
      }
      
      int ret = commonInfo.getUpdateResult(updateServiceName, updateCode, 
					   paramValues.toString());
      if (ret != 1) {
	commonInfo.showMessage("データの更新に失敗しました");
      } else {
	tableView.refreshTable();
      }
    } 
  }

  public void insertStructTableIfEmpty(String parameter) {
    if (tableView.getRowCount() != 0) {
      commonInfo.showMessage("この画面の設定表は定義済みです。");
      return;
    }
    String[] tokens  = parameter.split("\\:");
    String insertServiceName = tokens[0];
    String insertCode = tokens[1];
    String insParams = commonInfo.getServerInsertParams(tokens[0], tokens[1]);
    String[] ttokens = insParams.split("\\|");    
    String[] insertParams = ttokens[0].split("\\:");
    String[] whereParams = ttokens[2].split("\\:");

    HashMap<String, JTextComponent> editorMap = new HashMap<String, JTextComponent>();
    ArrayList<Object> list = new ArrayList<Object>();
    list.add(" データ表に挿入するデータを設定して下さい。                 ");
    list.add(" UNDO 機能はないので挿入を実行する前に十分に確認して下さい。");

    for (String param : whereParams) {
      String key = param;
      String value = parentTabbedPane.getValueFromColumnCodeMap(key);
      String text = " " + key + " = " + value;
      JTextField editor = new JTextField(text);
      editor.setEditable(false); 
      editor.setFont(new Font("DialogInput", Font.PLAIN, 12));  
      editor.setBackground(Color.yellow);
      list.add(editor);
    }
    for (String param : insertParams) {
      String key = param;
      String text = " " + key + " = ";      
      JTextField editor = new JTextField(100);
      editor.setFont(new Font("DialogInput", Font.PLAIN, 12));
      editor.setText("  ");
      editor.setEditable(true);
      TitledBorder border = new TitledBorder(null, text, TitledBorder.LEFT, TitledBorder.TOP,
					     new Font("DialogInput", Font.PLAIN, 10));
      editor.setBorder(border);
      JScrollPane scroll = new JScrollPane(editor);
      list.add(scroll);
      editorMap.put(key, editor);
    }
    int ans =  JOptionPane.showOptionDialog(commonInfo.getFrame(),
					    list.toArray(), 
					    "挿入値の設定", 
					    JOptionPane.OK_CANCEL_OPTION,
					    JOptionPane.QUESTION_MESSAGE,
					    null,
					    null,
					    null);
    if (ans == JOptionPane.OK_OPTION) {
      StringBuilder paramValues = new StringBuilder();      
      for (String param : insertParams) {
	JTextComponent editor = editorMap.get(param);
	String value = editor.getText();
	if ((value == null) || (value.trim().equals(""))) {
	  paramValues.append(" |"); 
	} else {
	  paramValues.append(value.trim()).append("|"); 
	}
      }
      for (String param : whereParams) {
	String value = parentTabbedPane.getValueFromColumnCodeMap(param);
	if ((value == null) || (value.trim().equals(""))) {
	  paramValues.append(" |"); 
	} else {
	  paramValues.append(value.trim()).append("|"); 
	}
      }
      int ret = commonInfo.getInsertResult(insertServiceName, insertCode, 
					   paramValues.toString());
      if (ret != 1) {
	commonInfo.showMessage("データの挿入に失敗しました");
      } else {
	tableView.refreshTable();
      }
    } 
  } 

  public void showHtmlDebugView(String dummy) {
    String[] columnName = { "COLUMN_NAME", "COLUMN_VALUE" };
    HashMap<String, String> valueMap = new HashMap<String, String>();
    for (int i = 0; i < tableView.getRowCount(); i++) { 
      String key = tableView.getCodeAt(i, columnName[0]);
      String value = tableView.getCodeAt(i, columnName[1]);
      valueMap.put(key, value);
    }
    String urlPath = valueMap.get("URL");
    HtmlViewBase htmlView = new HtmlViewBase(commonInfo, urlPath);
    Dimension dim = new Dimension(900, 550);
    htmlView.setPreferredSize(dim);
    htmlView.setMinimumSize(dim);
    htmlView.setBackground(commonInfo.getColor("Khaki"));
    htmlView.pageOpened();
    JOptionPane.showMessageDialog(commonInfo.getFrame(), htmlView, 
				  "HTML画面のデバッグ表示", 
				  JOptionPane.INFORMATION_MESSAGE); 
  }

  public void showSQLDebugResult(String dummy) {
    String[] columnName = { "COLUMN_NAME", "COLUMN_VALUE" };
    HashMap<String, String> valueMap = new HashMap<String, String>();
    for (int i = 0; i < tableView.getRowCount(); i++) { 
      String key = tableView.getCodeAt(i, columnName[0]);
      String value = tableView.getCodeAt(i, columnName[1]);
      valueMap.put(key, value);
    }
    String queryName = valueMap.get("QUERY_NAME");
    String paramList = valueMap.get("PARAM_LIST");    
    StringBuilder paramValues = new StringBuilder();  
    if ((paramList == null) || (paramList.trim().equals(""))) {
      paramValues.append(" |");
    } else {      
      String[] params = paramList.split("\\:");
      HashMap<String, JTextComponent> editorMap = new HashMap<String, JTextComponent>();
      ArrayList<Object> list = new ArrayList<Object>();
      list.add(" パラメータの値を設定して下さい。                          ");
      for (String param : params) {
	String key = param;
	String text = " " + param + " = ";
	JTextField editor = new JTextField(100);
	editor.setFont(new Font("DialogInput", Font.PLAIN, 12));
	editor.setText("  ");
	editor.setEditable(true);
	TitledBorder border = new TitledBorder(null, text, TitledBorder.LEFT, TitledBorder.TOP,
					       new Font("DialogInput", Font.PLAIN, 10));
	editor.setBorder(border);
	list.add(editor);
	editorMap.put(key, editor);
      }
      int ans =  JOptionPane.showOptionDialog(commonInfo.getFrame(),
					      list.toArray(), 
					      "パラメータ値の設定", 
					      JOptionPane.OK_CANCEL_OPTION,
					      JOptionPane.QUESTION_MESSAGE,
					      null,
					      null,
					      null);
      if (ans == JOptionPane.OK_OPTION) {  
	for (String param : params) {
	  JTextComponent editor = editorMap.get(param);
	  String value = editor.getText();
	  if ((value == null) || (value.trim().equals(""))) {
	    paramValues.append(" |"); 
	  } else {
	    paramValues.append(value.trim()).append("|"); 
	  }
	}
      } else {
	return;
      }
    }    
    ArrayList<Object> list = new ArrayList<Object>();
    JTextArea editor = new JTextArea(16, 100);
    editor.setFont(new Font("DialogInput", Font.PLAIN, 12));
    editor.setLineWrap(true);
    editor.setEditable(true);
    TitledBorder border = new TitledBorder(null, "queryResult", TitledBorder.LEFT, TitledBorder.TOP,
					   new Font("DialogInput", Font.PLAIN, 10));
    editor.setBorder(border);
    JScrollPane scroll = new JScrollPane(editor);
    list.add(scroll);    
    String result = commonInfo.getCommonQueryResult(queryName, paramValues.toString());    
    String[] lines = result.split("\\$");
    for (String line : lines) {
      editor.append(" " + line + "\n");
    }
    JOptionPane.showMessageDialog(commonInfo.getFrame(),
				  list.toArray(), 
				  "SQL文の実行結果の確認", 
				  JOptionPane.INFORMATION_MESSAGE); 
  }

  public void showServerQueryResult(String dummy) {
    String[] columnName = { "COLUMN_NAME", "COLUMN_VALUE" };
    HashMap<String, String> valueMap = new HashMap<String, String>();
    for (int i = 0; i < tableView.getRowCount(); i++) { 
      String key = tableView.getCodeAt(i, columnName[0]);
      String value = tableView.getCodeAt(i, columnName[1]);
      valueMap.put(key, value);
    }
    String serviceName = valueMap.get("SERVICE_NAME");
    String panelID = valueMap.get("PANEL_ID");
    String switchFlag = valueMap.get("SWITCH_FLAG");
    String paramList = valueMap.get("PARAM_LIST");    
    StringBuilder paramValues = new StringBuilder();  
    if ((paramList == null) || (paramList.trim().equals(""))) {
      paramValues.append(" |");
    } else {      
      String[] params = paramList.split("\\:");
      HashMap<String, JTextComponent> editorMap = new HashMap<String, JTextComponent>();
      ArrayList<Object> list = new ArrayList<Object>();
      list.add(" パラメータの値を設定して下さい。                          ");
      for (String param : params) {
	String key = param;
	String text = " " + param + " = ";
	JTextField editor = new JTextField(100);
	editor.setFont(new Font("DialogInput", Font.PLAIN, 12));
	editor.setText("  ");
	editor.setEditable(true);
	TitledBorder border = new TitledBorder(null, text, TitledBorder.LEFT, TitledBorder.TOP,
					       new Font("DialogInput", Font.PLAIN, 10));
	editor.setBorder(border);
	list.add(editor);
	editorMap.put(key, editor);
      }
      int ans =  JOptionPane.showOptionDialog(commonInfo.getFrame(),
					      list.toArray(), 
					      "パラメータ値の設定", 
					      JOptionPane.OK_CANCEL_OPTION,
					      JOptionPane.QUESTION_MESSAGE,
					      null,
					      null,
					      null);
      if (ans == JOptionPane.OK_OPTION) {  
	for (String param : params) {
	  JTextComponent editor = editorMap.get(param);
	  String value = editor.getText();
	  if ((value == null) || (value.trim().equals(""))) {
	    paramValues.append(" |"); 
	  } else {
	    paramValues.append(value.trim()).append("|"); 
	  }
	}
      } else {
	return;
      }
    }    
    ArrayList<Object> list = new ArrayList<Object>();
    JTextArea editor = new JTextArea(16, 100);
    editor.setFont(new Font("DialogInput", Font.PLAIN, 12));
    editor.setLineWrap(true);
    editor.setEditable(true);
    TitledBorder border = new TitledBorder(null, "queryResult", TitledBorder.LEFT, TitledBorder.TOP,
					   new Font("DialogInput", Font.PLAIN, 10));
    editor.setBorder(border);
    JScrollPane scroll = new JScrollPane(editor);
    list.add(scroll);   


    String result = commonInfo.getQueryResult(serviceName, panelID, switchFlag,
					      paramValues.toString());
    String[] lines = result.split("\\$");
    for (String line : lines) {
      editor.append(" " + line + "\n");
    }
    JOptionPane.showMessageDialog(commonInfo.getFrame(),
				  list.toArray(), 
				  "SQL文の実行結果の確認", 
				  JOptionPane.INFORMATION_MESSAGE); 
  }

  public void showSelectedUpdateCommand(String dummy) {
    if (tableView.getSelectedRowCount() != 0) {
      String commandType = parentTabbedPane.getValueFromColumnCodeMap("COMMAND_TYPE");
      if (commandType.equals("DELETE")) {
	parentTabbedPane.openTab("delete");
      } else if (commandType.equals("UPDATE")) {
	parentTabbedPane.openTab("update");
      } else if (commandType.equals("INSERT")) {
	parentTabbedPane.openTab("insert");
      } else if (commandType.equals("SPECIAL")) {
	parentTabbedPane.openTab("special");
      } 
    }
  }

  public void showInvokedUpdateCommand(String dummy) {
    if (tableView.getRowCount() != 0) {
      String[] columnName = { "COLUMN_NAME", "COLUMN_VALUE" };
      HashMap<String, String> valueMap = new HashMap<String, String>();
      for (int i = 0; i < tableView.getRowCount(); i++) { 
	String key = tableView.getCodeAt(i, columnName[0]);
	String value = tableView.getCodeAt(i, columnName[1]);
	valueMap.put(key, value);
      }
      String commandType = valueMap.get("COMMAND_TYPE");
      if (commandType.equals("DELETE")) {
	parentTabbedPane.openTab("delete");
      } else if (commandType.equals("UPDATE")) {
	parentTabbedPane.openTab("update");
      } else if (commandType.equals("INSERT")) {
	parentTabbedPane.openTab("insert");
      } else if (commandType.equals("SPECIAL")) {
	parentTabbedPane.openTab("special");
      } 
    }
  }

 
  public void makeBackupOfStructTable(String dummy) {
    if (tableView.getSelectedRowCount() == 0) {
      commonInfo.showMessage("バックアップをとるべきシステム構成表が指定されていません。" );
      return;
    } else {
      String tableName = parentTabbedPane.getValueFromColumnCodeMap("TABLE_NAME");
      if (commonInfo.showNotice("システム構成表 " + tableName + " のバックアップを作成しますか？")) {
	commonMethods.makeBackupOfStructTable(tableName);
      }
    }
  }
 
  public void makeBackupOfStructTable2(String dummy) {
    String tableName = parentTabbedPane.getValueFromColumnCodeMap("TABLE_NAME");
    if (commonInfo.showNotice("システム構成表 " + tableName + " のバックアップを作成しますか？")) {
      commonMethods.makeBackupOfStructTable(tableName);
    }
  }
 
  public void readBackupOfStructTable(String dummy) {
    if (tableView.getSelectedRowCount() == 0) {
      commonInfo.showMessage("バックアップから読み込むべきシステム構成表が指定されていません。" );
      return;
    } else {
      String tableName = parentTabbedPane.getValueFromColumnCodeMap("TABLE_NAME");
      if (commonInfo.showNotice("バックアップからシステム構成表 " + tableName + " を読み込みますか？")) {
	commonMethods.readBackupOfStructTable(tableName);
      }
    }
  }

  public void readBackupOfStructTable2(String dummy) {
    String tableName = parentTabbedPane.getValueFromColumnCodeMap("TABLE_NAME");
    if (commonInfo.showNotice("バックアップからシステム構成表 " + tableName + " を読み込みますか？")) {
      commonMethods.readBackupOfStructTable(tableName);
    }
  }

  public void showRootDebugTabbedPane(String dummy) {  
    if (tableView.getSelectedRowCount() == 0) {
      commonInfo.showMessage("デバッグ画面を表示すべきサービスのサービス名を指定して下さい。" );
      return;
    } else {
      String serviceName = parentTabbedPane.getValueFromColumnCodeMap("SERVICE_NAME");
      String nodePath = parentTabbedPane.getValueFromColumnCodeMap("NODE_PATH");  
  
      commonInfo.setDebugMode();
      TabbedPaneBase tabbedPane = new TabbedPaneBase(serviceName, nodePath, commonInfo, null, null);
      Dimension dim = new Dimension(900, 550);
      tabbedPane.setPreferredSize(dim);
      tabbedPane.setMinimumSize(dim);
      tabbedPane.setBackground(commonInfo.getColor("Khaki"));
      tabbedPane.pageOpened();
      JOptionPane.showMessageDialog(commonInfo.getFrame(), tabbedPane, 
				    "サービスのルート画面のデバッグ表示", 
				    JOptionPane.INFORMATION_MESSAGE); 
      commonInfo.resetDebugMode();
    }
  }

  public void showSelectedDebugView(String dummy) {  
    if (tableView.getSelectedRowCount() == 0) {
      commonInfo.showMessage("デバッグ画面を表示すべきサービスのサービス名を指定して下さい。" );
      return;
    } else {
      String serviceName = parentTabbedPane.getValueFromColumnCodeMap("SERVICE_NAME");
      String nodePath = parentTabbedPane.getValueFromColumnCodeMap("NODE_PATH");
      String nodeID = parentTabbedPane.getValueFromColumnCodeMap("CHILD_NODE_ID");
      String componentType = parentTabbedPane.getValueFromColumnCodeMap("CHILD_COMPONENT_TYPE");
      String panelID = parentTabbedPane.getValueFromColumnCodeMap("CHILD_PANEL_ID");
      String methodWhenSelected = parentTabbedPane.getValueFromColumnCodeMap("CHILD_METHOD_WHEN_SELECTED");
      if (methodWhenSelected.trim().equals("")) {
	methodWhenSelected = null;
      }

      TabbedPaneBase rootTabbedPane = new TabbedPaneBase(serviceName, "root", commonInfo, null, null);

      if (componentType.equals("TabbedPane")) {
	if (!(nodeID.startsWith("/"))) {
	  nodePath = nodePath + "." + nodeID;
	} else {
	  int pos = nodeID.indexOf(".");
	  serviceName = nodeID.substring(1, pos);
	  nodePath = nodeID.substring(pos+1); 
	} 

	commonInfo.setDebugMode();
	TabbedPaneBase tabbedPane = new TabbedPaneBase(serviceName, nodePath, commonInfo, 
						       rootTabbedPane, methodWhenSelected);
	Dimension dim = new Dimension(900, 550);
	tabbedPane.setPreferredSize(dim);
	tabbedPane.setMinimumSize(dim);
	tabbedPane.setBackground(commonInfo.getColor("Khaki"));
	tabbedPane.pageOpened();
	JOptionPane.showMessageDialog(commonInfo.getFrame(), tabbedPane, 
				      "選択された選択画面のデバッグ表示", 
				      JOptionPane.INFORMATION_MESSAGE); 
	commonInfo.resetDebugMode();
      } else if (componentType.equals("DataPanel")) {
	commonInfo.setDebugMode();
	DataPanelBase dataPanel = new DataPanelBase(serviceName, nodePath, panelID,
						    commonInfo, rootTabbedPane, methodWhenSelected);
	Dimension dim = new Dimension(900, 550);
	dataPanel.setPreferredSize(dim);
	dataPanel.setMinimumSize(dim);
	dataPanel.setBackground(commonInfo.getColor("Khaki"));
	dataPanel.setBorder(new CompoundBorder(BorderFactory.createRaisedBevelBorder(), 
					       new EmptyBorder(2,2,2,2)));
	dataPanel.pageOpened();
	JOptionPane.showMessageDialog(commonInfo.getFrame(), dataPanel, 
				      "選択されたパネル画面のデバッグ表示", 
				      JOptionPane.INFORMATION_MESSAGE); 
	commonInfo.resetDebugMode();
      }    
    }
  }

  public void showThisDebugView(String dummy) {  
    String[] columnName = { "COLUMN_NAME", "COLUMN_VALUE" };
    HashMap<String, String> map = new HashMap<String, String>();
    for (int i = 0; i < tableView.getRowCount(); i++) { 
      String key = tableView.getCodeAt(i, columnName[0]);
      String val = tableView.getCodeAt(i, columnName[1]);
      map.put(key, val);
    }
    String serviceName = map.get("SERVICE_NAME");
    String nodePath = map.get("NODE_PATH");
    String nodeID = map.get("CHILD_NODE_ID");    
    String componentType = map.get("CHILD_COMPONENT_TYPE"); 
    String panelID = map.get("CHILD_PANEL_ID");
    String methodWhenSelected = map.get("CHILD_METHOD_WHEN_SELECTED");
    if (methodWhenSelected != null) {
      if (methodWhenSelected.trim().equals("")) {
	methodWhenSelected = null;
      }
    }

    TabbedPaneBase rootTabbedPane = new TabbedPaneBase(serviceName, "root", commonInfo, null, null);

    if (componentType.equals("TabbedPane")) {
      if (!(nodeID.startsWith("/"))) {
	nodePath = nodePath + "." + nodeID;
      } else {
	int pos = nodeID.indexOf(".");
	serviceName = nodeID.substring(1, pos);
	nodePath = nodeID.substring(pos+1); 
      } 

      commonInfo.setDebugMode();
      TabbedPaneBase tabbedPane = new TabbedPaneBase(serviceName, nodePath, commonInfo, 
						     rootTabbedPane, methodWhenSelected);
      Dimension dim = new Dimension(900, 550);
      tabbedPane.setPreferredSize(dim);
      tabbedPane.setMinimumSize(dim);
      tabbedPane.setBackground(commonInfo.getColor("Khaki"));
      tabbedPane.pageOpened();
      JOptionPane.showMessageDialog(commonInfo.getFrame(), tabbedPane, 
				    "選択された選択画面のデバッグ表示", 
				    JOptionPane.INFORMATION_MESSAGE); 
      commonInfo.resetDebugMode();
    } else if (componentType.equals("DataPanel")) {
      commonInfo.setDebugMode();
      DataPanelBase dataPanel = new DataPanelBase(serviceName, nodePath, panelID,
						  commonInfo, rootTabbedPane, methodWhenSelected);
      Dimension dim = new Dimension(900, 550);
      dataPanel.setPreferredSize(dim);
      dataPanel.setMinimumSize(dim);
      dataPanel.setBackground(commonInfo.getColor("Khaki"));
      dataPanel.pageOpened();
      dataPanel.setBorder(new CompoundBorder(BorderFactory.createRaisedBevelBorder(), 
					     new EmptyBorder(2,2,2,2)));
      JOptionPane.showMessageDialog(commonInfo.getFrame(), dataPanel, 
				    "選択されたパネル画面のデバッグ表示", 
				    JOptionPane.INFORMATION_MESSAGE); 
      commonInfo.resetDebugMode();	
    } 
  }

  public void showDebugDataPanel(String dummy) {  
    String[] columnName = { "COLUMN_NAME", "COLUMN_VALUE" };
    HashMap<String, String> map = new HashMap<String, String>();
    for (int i = 0; i < tableView.getRowCount(); i++) { 
      String key = tableView.getCodeAt(i, columnName[0]);
      String val = tableView.getCodeAt(i, columnName[1]);
      map.put(key, val);
    }
    String serviceName = map.get("SERVICE_NAME");
    String panelID = map.get("PANEL_ID");

    String nodePath = parentTabbedPane.getValueFromColumnCodeMap("NODE_PATH");
    String nodeID = parentTabbedPane.getValueFromColumnCodeMap("NODE_ID");
    nodePath = nodePath + "." + nodeID;
    commonInfo.structControlDebugMode = false;
    String methodWhenSelected = parentTabbedPane.getValueFromColumnCodeMap("METHOD_WHEN_SELECTED");
    if (methodWhenSelected.trim().equals("")) {
      methodWhenSelected = null;
    }
    TabbedPaneBase rootTabbedPane = new TabbedPaneBase(serviceName, "root", commonInfo, null, null);
    commonInfo.setDebugMode();	
    DataPanelBase dataPanel = new DataPanelBase(serviceName, nodePath, panelID,
						commonInfo, rootTabbedPane, methodWhenSelected);
    Dimension dim = new Dimension(900, 550);
    dataPanel.setPreferredSize(dim);
    dataPanel.setMinimumSize(dim);
    dataPanel.setBackground(commonInfo.getColor("Khaki"));
    dataPanel.setBorder(new CompoundBorder(BorderFactory.createRaisedBevelBorder(), 
					   new EmptyBorder(2,2,2,2)));
    dataPanel.pageOpened();
    JOptionPane.showMessageDialog(commonInfo.getFrame(), dataPanel, 
				  "選択されたパネル画面のデバッグ表示", 
				  JOptionPane.INFORMATION_MESSAGE); 	
    commonInfo.resetDebugMode();	
  }

  public void setStaffAsDebugUser(String dummy) {
    String userID = parentTabbedPane.getValueFromColumnCodeMap("USER_ID");
    String staffQualification = parentTabbedPane.getValueFromColumnCodeMap("QUALIFICATION");
    String passwdStatus = parentTabbedPane.getValueFromColumnCodeMap("PASSWORD_STATUS");
    String staffName = parentTabbedPane.getDisplayFromColumnCodeMap("STAFF_CODE");
    if ((userID.trim().equals("")) || (staffQualification.trim().equals("")) ||
	(passwdStatus.trim().equals(""))) {
      commonInfo.showMessage(" " + staffName + " をデバッグ用ユーザに設定することはできません。");
    } else {
      commonInfo.USER_ID_D = parentTabbedPane.getValueFromColumnCodeMap("USER_ID");
      commonInfo.STAFF_CODE_D = parentTabbedPane.getValueFromColumnCodeMap("STAFF_CODE");
      commonInfo.STAFF_NAME_D = parentTabbedPane.getDisplayFromColumnCodeMap("STAFF_CODE");
      commonInfo.STAFF_ATTRIB_D = parentTabbedPane.getValueFromColumnCodeMap("STAFF_ATTRIB");
      commonInfo.STAFF_QUALIFICATION_D = parentTabbedPane.getValueFromColumnCodeMap("QUALIFICATION");
      commonInfo.LOCAL_ATTRIB_D = parentTabbedPane.getValueFromColumnCodeMap("LOCAL_ATTRIB");
      commonInfo.STAFF_DEPARTMENT_D = commonInfo.STAFF_ATTRIB_D;
      commonInfo.studentDebugUserSelected = false;
      commonInfo.staffDebugUserSelected = true;
      commonInfo.addStaffAttribToCommonCodeMapD(); 
      
      StringBuffer sbuf = new StringBuffer();
      sbuf.append("USER_ID       : ").append(commonInfo.USER_ID_D).append("$");
      sbuf.append("STAFF_CODE    : ").append(commonInfo.STAFF_CODE_D).append("$");
      sbuf.append("STAFF_ATTRIB  : ").append(commonInfo.STAFF_ATTRIB_D).append("$");
      sbuf.append("QUALIFICATION : ").append(commonInfo.STAFF_QUALIFICATION_D).append("$");
      sbuf.append("LOCAL_ATTRIB  : ").append(commonInfo.LOCAL_ATTRIB_D).append("$");
      String[] strArray = sbuf.toString().split("\\$");
      JOptionPane.showMessageDialog(commonInfo.getFrame(), strArray, 
				    "Warning", JOptionPane.WARNING_MESSAGE);
    }
  }  

  public void setStudentAsDebugUser(String dummy) {
    commonInfo.STUDENT_CODE_D = parentTabbedPane.getValueFromColumnCodeMap("STUDENT_CODE");
    commonInfo.STUDENT_NAME_D = parentTabbedPane.getDisplayFromColumnCodeMap("STUDENT_CODE");
    commonInfo.STUDENT_STATUS_D = parentTabbedPane.getValueFromColumnCodeMap("STUDENT_STATUS");
    commonInfo.STUDENT_FACULTY_D = parentTabbedPane.getValueFromColumnCodeMap("FACULTY");
    commonInfo.STUDENT_DEPARTMENT_D = parentTabbedPane.getValueFromColumnCodeMap("DEPARTMENT");
    commonInfo.STUDENT_COURSE_D = parentTabbedPane.getValueFromColumnCodeMap("COURSE");
    commonInfo.STUDENT_COURSE_2_D = parentTabbedPane.getValueFromColumnCodeMap("COURSE_2");
    commonInfo.STUDENT_GAKUNEN_D = parentTabbedPane.getValueFromColumnCodeMap("GAKUNEN");
    commonInfo.STUDENT_CURRICULUM_YEAR_D = parentTabbedPane.getValueFromColumnCodeMap("CURRICULUM_YEAR");
    commonInfo.STUDENT_SUPERVISOR_D = parentTabbedPane.getValueFromColumnCodeMap("SUPERVISOR");
    commonInfo.STUDENT_SUPERVISOR_NAME_D = parentTabbedPane.getDisplayFromColumnCodeMap("SUPERVISOR");
    commonInfo.studentDebugUserSelected = true;
    commonInfo.staffDebugUserSelected = false;
    commonInfo.addStudentAttribToCommonCodeMapD();  

    StringBuffer sbuf = new StringBuffer();
    sbuf.append("STUDENT_CODE       : ").append(commonInfo.STUDENT_CODE_D).append("$");
    sbuf.append("STUDENT_STATUS     : ").append(commonInfo.STUDENT_STATUS_D).append("$");
    sbuf.append("STUDENT_FACULTY    : ").append(commonInfo.STUDENT_FACULTY_D).append("$");
    sbuf.append("STUDENT_DEPARTMENT : ").append(commonInfo.STUDENT_DEPARTMENT_D).append("$");
    sbuf.append("STUDENT_COURSE     : ").append(commonInfo.STUDENT_COURSE_D).append("$");
    sbuf.append("STUDENT_SUPERVISOR : ").append(commonInfo.STUDENT_SUPERVISOR_D).append("$");
    String[] strArray = sbuf.toString().split("\\$");
    JOptionPane.showMessageDialog(commonInfo.getFrame(), strArray, 
                                  "Warning", JOptionPane.WARNING_MESSAGE); 
  }
 
  public void rtorokuToSeiseki(String params) {
    if (commonInfo.showNotice("成績報告データを成績表に記入してよいですか？")) {

      String SCHOOL_YEAR  = parentTabbedPane.getValueFromColumnCodeMap("SCHOOL_YEAR");
      String SUBJECT_CODE = parentTabbedPane.getValueFromColumnCodeMap("SUBJECT_CODE");
      String SUBJECT_NAME = parentTabbedPane.getDisplayFromColumnCodeMap("SUBJECT_CODE");
      String CLASS_CODE   = parentTabbedPane.getValueFromColumnCodeMap("CLASS_CODE");
      String TEACHER_CODE = parentTabbedPane.getValueFromColumnCodeMap("TEACHER_CODE");
      String TEACHER_NAME = parentTabbedPane.getDisplayFromColumnCodeMap("TEACHER_CODE");

      String[] columnName = { "DEPARTMENT", "COURSE", "GAKUNEN", "STUDENT_STATUS",
			      "STUDENT_CODE", "MARKS" };
      String[] val  = new String[columnName.length];
      String[] disp = new String[columnName.length];

      int marks;
      String deptName, courseName, gakunenName, statusName;
      String studentCode, studentName, marksName, marksName2;
      StringBuilder sbuf = new StringBuilder();

      sbuf.append("                    (期末試験の成績報告データの処理に関するお知らせ)  $ $   TO： ");
      sbuf.append(TEACHER_NAME).append(" 様 $ FROM： 情報工学部学務係  $ $");
      sbuf.append(" 次の授業科目の成績報告データを「成績マスター表」に $ 記入する処理を行ないました。 $ ");
      sbuf.append("$  授業科目名： " + SUBJECT_NAME);
      sbuf.append("$    担当教官： " + TEACHER_NAME);
      sbuf.append("$  担当クラス： ").append(CLASS_CODE);
      sbuf.append("$    処理日時： " + commonInfo.thisYear+ "年 ");
      sbuf.append(commonInfo.thisMonth + "月 " + commonInfo.thisDay +  "日 " + " $ $");
    
      for (int i = 0; i < tableView.getRowCount(); i++) { 
	for (int m = 0; m < columnName.length; m++) {
	  val[m] = tableView.getCodeAt(i, columnName[m]);
	  disp[m] = tableView.getNameAt(i, columnName[m]);
	  if ((val[m] == null) || (val[m].equals(""))) {
	    val[m] = " ";
	    disp[m] = " ";
	  }
	}
	deptName    = disp[0];
	courseName  = disp[1];
	gakunenName = disp[2];
	statusName  = disp[3];
	studentCode = val[4];
	studentName = disp[4];
	marksName = val[5];

	try {
	  marks = Integer.parseInt(marksName);
	} catch (NumberFormatException e) {
	  System.out.println(e.toString());
	  marks = 0;
	}
	if (marks <= 0) {
	  marks = 0;
	} else if (marks < 60) {
	  marks = 1;
	}
	marksName  = packZero(""+marks, 3);
	if (marks == 0) {
	  marksName2 = "不合格";
	} else if (marks == 1) {
	  marksName2 = "再試験";
	} else {	
	  marksName2 = packSpace(""+marks, 5);
	}

	if (i == 0) {
	  sbuf.append("  学科/専攻  コース  学年 ").append("\t").append("学生番号 学生氏名   得点").append("\t").append("在籍状況 ").append("$ $");
	}
	sbuf.append(deptName).append(" ").append(courseName).append(" ").append(gakunenName).append("\t").append(studentCode).append(" ").append(studentName).append("  ").append(marksName2).append("\t ").append(statusName).append("$");
      }
      commonMethods.sendTransferNotice(TEACHER_CODE, sbuf.toString()); 

      String param = SCHOOL_YEAR+"|"+SUBJECT_CODE+"|"+CLASS_CODE;
      commonMethods.rtorokuToSeiseki(param, SUBJECT_NAME, TEACHER_NAME);
      tableView.refreshTable();
    }
  }

  public void saishiToSeiseki(String params) {
    if (commonInfo.showNotice("再試の成績報告データを成績表に記入してよいですか？")) {

      String SCHOOL_YEAR  = parentTabbedPane.getValueFromColumnCodeMap("SCHOOL_YEAR");
      String SUBJECT_CODE = parentTabbedPane.getValueFromColumnCodeMap("SUBJECT_CODE");
      String SUBJECT_NAME = parentTabbedPane.getDisplayFromColumnCodeMap("SUBJECT_CODE");
      String CLASS_CODE   = parentTabbedPane.getValueFromColumnCodeMap("CLASS_CODE");
      String TEACHER_CODE = parentTabbedPane.getValueFromColumnCodeMap("TEACHER_CODE");
      String TEACHER_NAME = parentTabbedPane.getDisplayFromColumnCodeMap("TEACHER_CODE");

      String[] columnName = { "DEPARTMENT", "COURSE", "GAKUNEN", "STUDENT_STATUS",
			      "STUDENT_CODE", "MARKS" };
      String[] val  = new String[columnName.length];
      String[] disp = new String[columnName.length];

      int marks;
      String deptName, courseName, gakunenName, statusName;
      String studentCode, studentName, marksName, marksName2;
      StringBuilder sbuf = new StringBuilder();

      sbuf.append("                    (再試験の成績報告データの処理に関するお知らせ)  $ $   TO： ");
      sbuf.append(TEACHER_NAME).append(" 様 $ FROM： 情報工学部学務係  $ $");
      sbuf.append(" 次の再試験の成績報告データを「成績マスター表」に $ 記入する処理を行ないました。 $ ");
      sbuf.append("$    履修年度： " + SCHOOL_YEAR);
      sbuf.append("$  授業科目名： " + SUBJECT_NAME);
      sbuf.append("$    担当教官： " + TEACHER_NAME);
      sbuf.append("$  担当クラス： ").append(CLASS_CODE);
      sbuf.append("$    処理日時： " + commonInfo.thisYear+ "年 ");
      sbuf.append(commonInfo.thisMonth + "月 " + commonInfo.thisDay +  "日 " + " $ $");
    
      for (int i = 0; i < tableView.getRowCount(); i++) { 
	for (int m = 0; m < columnName.length; m++) {
	  val[m] = tableView.getCodeAt(i, columnName[m]);
	  disp[m] = tableView.getNameAt(i, columnName[m]);
	  if ((val[m] == null) || (val[m].equals(""))) {
	    val[m] = " ";
	    disp[m] = " ";
	  }
	}
	deptName    = disp[0];
	courseName  = disp[1];
	gakunenName = disp[2];
	statusName  = disp[3];
	studentCode = val[4];
	studentName = disp[4];
	marksName = val[5];

	try {
	  marks = Integer.parseInt(marksName);
	} catch (NumberFormatException e) {
	  System.out.println(e.toString());
	  marks = 0;
	}
	if (marks <= 0) {
	  marks = 0;
	} else if (marks < 60) {
	  marks = 1;
	}
	marksName  = packZero(""+marks, 3);
	if (marks == 0) {
	  marksName2 = "不合格";
	} else if (marks == 1) {
	  marksName2 = "再試験";
	} else {	
	  marksName2 = packSpace(""+marks, 5);
	}

	if (i == 0) {
	  sbuf.append("  学科/専攻  コース  学年 ").append("\t").append("学生番号 学生氏名   得点").append("\t").append("在籍状況 ").append("$ $");
	}
	sbuf.append(deptName).append(" ").append(courseName).append(" ").append(gakunenName).append("\t").append(studentCode).append(" ").append(studentName).append("  ").append(marksName2).append("\t ").append(statusName).append("$");

      }  
      commonMethods.sendTransferNotice(TEACHER_CODE, sbuf.toString());  
      
      String param = SCHOOL_YEAR+"|"+SUBJECT_CODE+"|"+CLASS_CODE+"|"+TEACHER_CODE;
      commonMethods.saishiToSeiseki(param, SUBJECT_NAME, TEACHER_NAME);
      tableView.refreshTable();
    }
  }

  public void ninteiToSeiseki(String param2) {   
    String SCHOOL_YEAR = "" + commonInfo.thisSchoolYear;
    String STUDENT_CODE = parentTabbedPane.getValueFromColumnCodeMap("STUDENT_CODE");
    String STUDENT_NAME = parentTabbedPane.getDisplayFromColumnCodeMap("STUDENT_CODE");
    String FACULTY      = parentTabbedPane.getValueFromColumnCodeMap("FACULTY");
    String DEPARTMENT   = parentTabbedPane.getValueFromColumnCodeMap("DEPARTMENT");
    String COURSE       = parentTabbedPane.getValueFromColumnCodeMap("COURSE");

    String TEACHER = " ";
    if (DEPARTMENT.equals("31")) {
      TEACHER = "82895044";
    } else if (DEPARTMENT.equals("32")) {
      TEACHER = "82895145";
    } else if (DEPARTMENT.equals("33")) {
      TEACHER = "82895246";
    } else if (DEPARTMENT.equals("34")) {
      TEACHER = "82895347";
    } else if (DEPARTMENT.equals("35")) {
      TEACHER = "82895448";
    }  

    if (commonInfo.showNotice( STUDENT_NAME + " の単位認定データを成績表に記入してよいですか？" )) {
      String param = STUDENT_CODE+"|"+TEACHER+"|"+SCHOOL_YEAR;
      int count = commonMethods.ninteiToSeiseki(param);
      commonInfo.showMessage(STUDENT_NAME + " の " + count + " 科目を認定しました。");
    }
  }

  public void addSelectedStudentsToListToPass() { 
    ArrayList<String> list = new ArrayList<String>();
    int[] rows = tableView.getSelectedRows();
    for (int i = 0; i < rows.length; i++) {   
      String dname = tableView.getNameAt(rows[i], "DEPARTMENT");
      String gname = tableView.getNameAt(rows[i], "GAKUNEN");
      String code = tableView.getCodeAt(rows[i], "STUDENT_CODE");
      String name = tableView.getNameAt(rows[i], "STUDENT_CODE");
      String data = dname + "|" + gname  + "|" + code + "|" + name;
      list.add(data);
    }    
    parentTabbedPane.setStudentListToPass(list);
  }

  public void addToPhotoList(String param) {
    addSelectedStudentsToListToPass();
    parentTabbedPane.openTab("photoPanel");
  }

  public void addToMailList(String param) {
    addSelectedStudentsToListToPass();
    parentTabbedPane.openTab("mailPanel");
  }

  public void selectAllList(String param) { 
    tableView.selectAll();
  }

  public void clearPhotoView(String param) { 
    if (photoView != null) {
      photoView.clearPhotoView();
    }
  }

  public void clearAddressList(String param) { 
//    if (mailView != null) {
//      mailView.clearAddressList();
//    }
  }

  public void removeSelectedAddress(String param) { 
//    if (mailView != null) {
//      mailView.removeSelectedAddress();
 //   }
  }

  public void sendMailToAll(String param) { 
//    if (mailView != null) {
//      mailView.sendMailToAll();
//    }
  }

  public void sendMailToSelected(String param) { 
//    if (mailView != null) {
//      mailView.sendMailToSelected();
//    }
  }

  public void changeSMTPSite(String param) { 
//    if (mailView != null) {
//      mailView.changeSMTPSite();
//    }
  }

  public void printSyllabus(String param) { 
    if (syllabusView != null) {
      StringBuffer sbuf = new StringBuffer();
      syllabusView.makeSyllabusLatexText(sbuf);
      
      try {
	if (tablePrinter == null) {
	  tablePrinter = new TablePrinter(commonInfo);
	}
	tablePrinter.latexToPrinter(sbuf);
      } catch (Exception e) {
	e.printStackTrace();
      }
    }
  }

  public void makeSyllabusLatex(String param) {
    if (syllabusView != null) {
      syllabusView.makeSyllabusLatex();
    }
  }

  public void makeSyllabusText(String param) {
    if (syllabusView != null) {
      syllabusView.makeSyllabusText();
    }
  }

  public void makeUndergradSyllabusLatex(String param) {
    if ((serviceName.equals("SyllabusTool")) &&
	(nodePath.equals("root.syllabusPrint.course"))) {
      if (tablePrinter == null) {
	tablePrinter = new TablePrinter(tableView);
      }
      tablePrinter.makeUndergradSyllabusLatex();
    }
  }

  public void makeGraduateSyllabusLatex(String param) {
    if ((serviceName.equals("SyllabusTool")) &&
	(nodePath.equals("root.syllabusPrint.course"))) {
      if (tablePrinter == null) {
	tablePrinter = new TablePrinter(tableView);
      }
      tablePrinter.makeGraduateSyllabusLatex();
    }
  }

  public void makeUndergradSyllabusHtml(String param) {
    if ((serviceName.equals("SyllabusTool")) &&
	(nodePath.equals("root.syllabusPrint.course"))) {
      if (tablePrinter == null) {
	tablePrinter = new TablePrinter(tableView);
      }
      tablePrinter.makeUndergradSyllabusHtml();
    }
  }

  public void makeGraduateSyllabusHtml(String param) {
    if ((serviceName.equals("SyllabusTool")) &&
	(nodePath.equals("root.syllabusPrint.course"))) {
      if (tablePrinter == null) {
	tablePrinter = new TablePrinter(tableView);
      }
      tablePrinter.makeGraduateSyllabusHtml();
    }
  }



  public void saveKimatsuData(String param) {
    if (reportTableView != null) {
      reportTableView.saveKimatsuData();
    }
  }

  public void doKimatsuReport(String param) {
    if (reportTableView != null) {
      reportTableView.doKimatsuReport();
    }
  }

  public void makeKimatsuReportFile(String param) {
    if (reportTableView != null) {
      reportTableView.makeKimatsuReportFile();
    }
  }

  public void readKimatsuReportFile(String param) {
    if (reportTableView != null) {
      reportTableView.readKimatsuReportFile();
    }
  }

  public void cancelInputData(String param) {
    if (reportTableView != null) {
      reportTableView.cancelInputData();
    }
  }

  public void saveSaishiData(String param) {
    if (reportTableView != null) {
      reportTableView.saveSaishiData();
    }
  }

  public void doSaishiReport(String param) {
    if (reportTableView != null) {
      reportTableView.doSaishiReport();
    }
  }

  public void makeSaishiReportFile(String param) {
    if (reportTableView != null) {
      reportTableView.makeSaishiReportFile();
    }
  }

  public void readSaishiReportFile(String param) {
    if (reportTableView != null) {
      reportTableView.readSaishiReportFile();
    }
  }

  public void makeNinteiForm(String param) {
    if (ninteiTableView != null) {
      ninteiTableView.makeNinteiForm();
    }
  }

  public void readNinteiForm(String param) {
    if (ninteiTableView != null) {
      ninteiTableView.readNinteiForm();
    }
  }
  
  public void cancelSelectedRegistr(String param) { 
    if (registrTableView != null) {
      registrTableView.cancelSelectedRegistr();
    } else if (jikanwariView != null) {
      jikanwariView.cancelSelectedRegistr();
    }
  }
   
  public void registrSelectedSubject(String param) {
    if (registrTableView != null) {
      registrTableView.registrSelectedSubject();
    } else if (jikanwariView != null) {
      jikanwariView.registrSelectedSubject();
    }
  } 

  public void setGakuseki(String param) {
    if (gakusekiView != null) {
      gakusekiView.setGakusekiData(param);
    }
  } 

  public void printAttendCheckSheet(String param) {
    try {
      if (tablePrinter == null) {
	tablePrinter = new TablePrinter(tableView);
      }
      tablePrinter.printAttendCheckSheet();
    } catch (Exception e) {
      e.printStackTrace();
    }
  }
    
  public void makeAttendCheckSheetLatex(String param) {
    try {
      if (tablePrinter == null) {
	tablePrinter = new TablePrinter(tableView);
      }
      tablePrinter.makeAttendCheckSheetLatex();
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  public void setMonday(String param) {
    int row = tableView.getSelectedRow();
    if (row >= 0) {     
      int[] rows = tableView.getSelectedRows(); 
      ((CalendarTableView)tableView).setMonday(rows);
    }
  }

  public void setTuesday(String param) {
    int row = tableView.getSelectedRow();
    if (row >= 0) {   
      int[] rows = tableView.getSelectedRows();    
      ((CalendarTableView)tableView).setTuesday(rows);
    }
  }

  public void setWednesday(String param) {
    int row = tableView.getSelectedRow();
    if (row >= 0) {      
      int[] rows = tableView.getSelectedRows(); 
      ((CalendarTableView)tableView).setWednesday(rows);
    }
  }

  public void setThursday(String param) {
    int row = tableView.getSelectedRow();
    if (row >= 0) {    
      int[] rows = tableView.getSelectedRows();   
      ((CalendarTableView)tableView).setThursday(rows);
    }
  }

  public void setFriday(String param) {
    int row = tableView.getSelectedRow();
    if (row >= 0) {      
      int[] rows = tableView.getSelectedRows(); 
      ((CalendarTableView)tableView).setFriday(rows);
    }
  }

  public void deleteClassWeek(String param) {
    int row = tableView.getSelectedRow();
    if (row >= 0) {     
      int[] rows = tableView.getSelectedRows();  
      ((CalendarTableView)tableView).deleteClassWeek(rows);
    }
  }

  public void addHolidayInfo(String param) {
    int row = tableView.getSelectedRow();
    if (row >= 0) {    
      int[] rows = tableView.getSelectedRows();   
      ((CalendarTableView)tableView).addHolidayInfo(rows);
    }
  }

  public void deleteHolidayInfo(String param) {
    int row = tableView.getSelectedRow();
    if (row >= 0) {   
      int[] rows = tableView.getSelectedRows();    
      ((CalendarTableView)tableView).deleteHolidayInfo(rows);
    }
  }

  public void addSchoolEvent(String param) {
    int row = tableView.getSelectedRow();
    if (row >= 0) {   
      int[] rows = tableView.getSelectedRows();    
      ((CalendarTableView)tableView).addSchoolEvent(rows);
    }
  }

  public void deleteSchoolEvent(String param) {
    int row = tableView.getSelectedRow();
    if (row >= 0) {   
      int[] rows = tableView.getSelectedRows();    
      ((CalendarTableView)tableView).deleteSchoolEvent(rows);
    }
  }

  public void addNewHolidayInfo(String param) {  
    int row = tableView.getSelectedRow();
    if (row >= 0) {  
      int[] rows = tableView.getSelectedRows();     
      ((CalendarTableView)tableView).addNewHolidayInfo(rows);
    }
  }

  public void addNewSchoolEvent(String param) {
    int row = tableView.getSelectedRow();
    if (row >= 0) {   
      int[] rows = tableView.getSelectedRows();    
      ((CalendarTableView)tableView).addNewSchoolEvent(rows);
    }
  }

  public void resetAttendControlParamsOfAllSubjects(String param) {
    if (commonInfo.showNotice("全科目の出欠調査の設定のリセットには計算時間を要します")) {
      String schoolYear = parentTabbedPane.getValueFromColumnCodeMap("SCHOOL_YEAR");
      String semester   = parentTabbedPane.getValueFromColumnCodeMap("SEMESTER");      
      int res = commonMethods.resetAttendControlParamsOfAllSubjects(schoolYear, semester); 
    }
  }

  public void resetAttendControlParamsOfSelectedSubjects(String param) {
    int row1 = tableView.getSelectedRow();
    if (row1 >= 0) {   
      String schoolYear = parentTabbedPane.getValueFromColumnCodeMap("SCHOOL_YEAR");
      String semester   = parentTabbedPane.getValueFromColumnCodeMap("SEMESTER");  
      int[] rows = tableView.getSelectedRows();        
      for (int row : rows) {
	String subjectCode = tableView.getCodeAt(row, "SUBJECT_CODE");
	String classCode   = tableView.getCodeAt(row, "CLASS_CODE");   
	int res = commonMethods.resetAttendControlParams(schoolYear, semester, subjectCode, classCode); 
      }
    }
  }

  public void copyCurriculumToNextYear(String param) {
    int thisSchoolYear = commonInfo.thisSchoolYear;
    int nextSchoolYear = thisSchoolYear + 1;
    String message = " "+thisSchoolYear+"年度の履修課程表を全て"+nextSchoolYear+"年度 $ の履修課程表にコピーします。$ 既に"+nextSchoolYear+"年度の履修課程表にデータが入っている場合、 $ そのデータは一旦リセットされて"+thisSchoolYear+"年度の設定がコピーされます。 $ その操作を実行して良いでか？ $$ なお、履修課程表に関連する操作を実行するためには履修課程の集計データ $ の作成(毎日深夜に作成)が必要になります。従って、履修課程表関連の操作が $ 実行できるのは明日以降になります。";
    
    boolean res = commonInfo.showNoticeLong(message);
    if (res) {
      commonMethods.copyCurriculumToNextYear(thisSchoolYear, nextSchoolYear);
    }    
  }



  public void copyEduCurriculumToNextYear(String param) { 
    int thisSchoolYear = commonInfo.thisSchoolYear;
    int nextSchoolYear = thisSchoolYear + 1;
    String message = " "+thisSchoolYear+"年度の教職履修課程表全て"+nextSchoolYear+"年度 $ の教職履修課程にコピーします。$ 既に"+nextSchoolYear+"年度の教職履修課程にデータが入っている場合、 $ そのデータは一旦リセットされて"+thisSchoolYear+"年度の設定がコピーされます。 $ その操作を実行して良いでか？ $$ なお、教職履修課程に関連する操作を実行するためには教職履修課程の集計データ $ の作成(毎日深夜に作成)が必要になります。従って、教職履修課程表関連の操作が $ 実行できるのは明日以降になります。";
    
    boolean res = commonInfo.showNoticeLong(message);
    if (res) {
      commonMethods.copyEduCurriculumToNextYear(thisSchoolYear, nextSchoolYear);
    }    
  }

  public void copyClassInfoToNextYear(String param) {
    int thisSchoolYear = commonInfo.thisSchoolYear;
    int nextSchoolYear = thisSchoolYear + 1;
    String message = " "+thisSchoolYear+"年度のクラス定義表を全て"+nextSchoolYear+"年度 $ のクラス定義表にコピーします。$ 既に"+nextSchoolYear+"年度のクラス定義表にデータが入っている場合、 $ そのデータは一旦リセットされて"+thisSchoolYear+"年度の設定がコピーされます。 $ その操作を実行して良いでか？";    
    boolean res = commonInfo.showNoticeLong(message);
    if (res) {
      commonMethods.copyClassInfoToNextYear(thisSchoolYear, nextSchoolYear);
    }      
  }

  public void copyJikanwariToNextYear(String param) {
    int thisSchoolYear = commonInfo.thisSchoolYear;
    int nextSchoolYear = thisSchoolYear + 1;
    String message = " "+thisSchoolYear+"年度の時間割表を全て"+nextSchoolYear+"年度 $ の時間割表にコピーします。$ 既に"+nextSchoolYear+"年度の時間割表にデータが入っている場合、 $ そのデータは一旦リセットされて"+thisSchoolYear+"年度の設定がコピーされます。 $ その操作を実行して良いでか？ $$ なお、時間割表に関連する操作を実行するためには時間割表の集計データ $ の作成(毎日深夜に作成)が必要になります。従って、時間割関連の操作が $ 実行できるのは明日以降になります。";    
    boolean res = commonInfo.showNoticeLong(message);
    if (res) {
      commonMethods.copyJikanwariToNextYear(thisSchoolYear, nextSchoolYear);
    }      
  }

  public void copyJikanwariOverlapToNextYear(String param) {
    int thisSchoolYear = commonInfo.thisSchoolYear;
    int nextSchoolYear = thisSchoolYear + 1;
    String message = " "+thisSchoolYear+"年度の時間割重複科目を全て"+nextSchoolYear+"年度 $ の時間割重複科目にコピーします。$ 既に"+nextSchoolYear+"年度の時間割重複科目にデータが入っている場合、 $ そのデータは一旦リセットされて"+thisSchoolYear+"年度の設定がコピーされます。 $ その操作を実行して良いでか？ ";    
    boolean res = commonInfo.showNoticeLong(message);
    if (res) {
      commonMethods.copyJikanwariOverlapToNextYear(thisSchoolYear, nextSchoolYear);
    } 
  }

  public void copyGradYokenNextYear(String param) {
    int thisSchoolYear = commonInfo.thisSchoolYear;
    int nextSchoolYear = thisSchoolYear + 1;
    String message = " "+thisSchoolYear+"年度の卒業修了要件を全て"+nextSchoolYear+"年度 $ の卒業修了要件にコピーします。$ 既に"+nextSchoolYear+"年度の卒業修了要件にデータが入っている場合、 $ そのデータは一旦リセットされて"+thisSchoolYear+"年度の設定がコピーされます。 $ その操作を実行して良いでか？ ";    
    boolean res = commonInfo.showNoticeLong(message);
    if (res) {
      commonMethods.copyGradYokenNextYear(thisSchoolYear, nextSchoolYear);
    }     
  }

  public void copyEdYokenToNextYear(String param) {
    int thisSchoolYear = commonInfo.thisSchoolYear;
    int nextSchoolYear = thisSchoolYear + 1;
    String message = " "+thisSchoolYear+"年度の教職修了要件を全て"+nextSchoolYear+"年度 $ の教職修了要件にコピーします。$ 既に"+nextSchoolYear+"年度の教職修了要件にデータが入っている場合、 $ そのデータは一旦リセットされて"+thisSchoolYear+"年度の設定がコピーされます。 $ その操作を実行して良いでか？ ";    
    boolean res = commonInfo.showNoticeLong(message);
    if (res) {
      commonMethods.copyEdYokenToNextYear(thisSchoolYear, nextSchoolYear);
    }     
  }

  public void copyModuleDefToNextYear(String param) {
    int thisSchoolYear = commonInfo.thisSchoolYear;
    int nextSchoolYear = thisSchoolYear + 1;
    String message = " "+thisSchoolYear+"年度のモジュール要件を全て"+nextSchoolYear+"年度 $ のモジュール要件にコピーします。$ 既に"+nextSchoolYear+"年度のモジュール要件にデータが入っている場合、 $ そのデータは一旦リセットされて"+thisSchoolYear+"年度の設定がコピーされます。 $ その操作を実行して良いでか？ ";    
    boolean res = commonInfo.showNoticeLong(message);
    if (res) {
      commonMethods.copyModuleDefToNextYear(thisSchoolYear, nextSchoolYear);
    }     
  }

  public void copyIIFCurriculumToNextYear(String param) {
    int thisSchoolYear = commonInfo.thisSchoolYear;
    int nextSchoolYear = thisSchoolYear + 1;
    String message = " "+thisSchoolYear+"年度のIIF履修課程表を全て"+nextSchoolYear+"年度 $ のIIF履修課程表にコピーします。$ 既に"+nextSchoolYear+"年度のIIF履修課程表にデータが入っている場合、 $ そのデータは一旦リセットされて"+thisSchoolYear+"年度の設定がコピーされます。 $ その操作を実行して良いでか？ ";    
    boolean res = commonInfo.showNoticeLong(message);
    if (res) {
      commonMethods.copyIIFCurriculumToNextYear(thisSchoolYear, nextSchoolYear);
    }     
  }

  public void copyYomikaeToNextYear(String param) {
    int thisSchoolYear = commonInfo.thisSchoolYear;
    int nextSchoolYear = thisSchoolYear + 1;
    String message = " "+thisSchoolYear+"年度の「新旧科目の読み替え」を"+nextSchoolYear+"年度 $ の「新旧科目の読み替え」にコピーします。$ 既に"+nextSchoolYear+"年度の「新旧科目の読み替え」にデータが入っている場合、 $ そのデータは一旦リセットされて"+thisSchoolYear+"年度の設定がコピーされます。 $ その操作を実行して良いでか？ ";    
    boolean res = commonInfo.showNoticeLong(message);
    if (res) {
      commonMethods.copyYomikaeToNextYear(thisSchoolYear, nextSchoolYear);
    } 
  }

  public void copySyllabusToNextYear(String param) {
    int thisSchoolYear = commonInfo.thisSchoolYear;
    int nextSchoolYear = thisSchoolYear + 1;
    String message = " "+thisSchoolYear+"年度のシラバスを"+nextSchoolYear+"年度 $ のシラバスにコピーします。$ 最近の４年間に開講された授業科目の「担当教官・担当科目」のシラバスは $ "+nextSchoolYear+"年度には開講されない場合であっても "+nextSchoolYear+"年度のシラバス $ として保存されます。この操作を実行して良いでか？ ";    
    boolean res = commonInfo.showNoticeLong(message);
    if (res) {
      commonMethods.copySyllabusToNextYear(thisSchoolYear, nextSchoolYear);
    }     
  }


  public void copyClassTimeZoneToNextYear(String param) {
    int thisSchoolYear = commonInfo.thisSchoolYear;
    int nextSchoolYear = thisSchoolYear + 1;
    String message = " "+thisSchoolYear+"年度のデフォルトの出欠調査時刻の設定を全て"+nextSchoolYear+"年度 $ のデフォルトの出欠調査時刻の設定にコピーします。$ 既に"+nextSchoolYear+"年度のデフォルトの出欠調査時刻の設定にデータが置かれている場合、 $ そのデータは一旦リセットされて"+thisSchoolYear+"年度の設定内容がコピーされます。 $ その操作を実行して良いでか？ ";    
    boolean res = commonInfo.showNoticeLong(message);
    if (res) {
      commonMethods.copyClassTimeZoneToNextYear(thisSchoolYear, nextSchoolYear);
    }     
  }
  
  public void setTeacherEnglishName(String param) {
    if (tableView.getSelectedRowCount() == 0) return;
    
    ArrayList<Object> list = new ArrayList<Object>();
    list.add(" 教員名の英文表記は  Taro YAMADA  の形式で設定して下さい。");

    String columnCode;
    String code = "", name = "", text = "";
    String teacherCode = "";

    String whereParams = "SUBJECT_CODE:TEACHER_CODE";
    String whereParamValues = "";
    StringTokenizer stk = new StringTokenizer(whereParams, ":");
    while (stk.hasMoreTokens()) {
      columnCode = stk.nextToken();
      if (parentTabbedPane.columnCodeMapContains(columnCode)) {
	code = parentTabbedPane.getValueFromColumnCodeMap(columnCode);
	name = parentTabbedPane.getDisplayFromColumnCodeMap(columnCode);
	if (code.equals("")) {
	  code = " ";
	}
      } else if (commonInfo.commonCodeMapContains(columnCode)) {
	code = commonInfo.getValueFromCommonCodeMap(columnCode);
	name = commonInfo.getDisplayFromCommonCodeMap(columnCode);
	if (code.equals("")) {
	  code = " ";
	}
      }
      text = " " + columnCode + " = " + code + " ( " + name + " )";
      if (columnCode.equals("TEACHER_CODE")) {
	teacherCode = code;
      }
      whereParamValues = whereParamValues + code + "|";
      JTextField editor = new JTextField(text);
      editor.setEditable(false); 
      editor.setFont(new Font("DialogInput", Font.PLAIN, 12));  
      editor.setBackground(Color.yellow);
      list.add(editor);
    }
  
    columnCode = "TEACHER_NAME_ENGLISH"; 
    name = parentTabbedPane.getDisplayFromColumnCodeMap(columnCode);
    JTextField nameEditor = new JTextField(100);
    nameEditor.setFont(new Font("DialogInput", Font.PLAIN, 12));
    nameEditor.setText(" " + name);
    nameEditor.setEditable(true);
    nameEditor.setBorder(new TitledBorder(columnCode + " = " + name));
    list.add(nameEditor);
  
    int ans =  JOptionPane.showOptionDialog(commonInfo.getFrame(),
					    list.toArray(), 
					    "教員名の英文表記", 
					    JOptionPane.OK_CANCEL_OPTION,
					    JOptionPane.QUESTION_MESSAGE,
					    null,
					    null,
					    null);
    if (ans != JOptionPane.OK_OPTION) {
      return;
    }

    String teacherNameEnglish = nameEditor.getText().trim();
    if (!teacherNameEnglish.equals("")) {
      whereParamValues = whereParamValues + teacherNameEnglish + "|";
    } else {
      whereParamValues = whereParamValues + " |";
    }

    if (!commonInfo.structControlDebugMode) {    
      if ((commonInfo.STAFF_QUALIFICATION.equals("3")) ||
	  (commonInfo.STAFF_QUALIFICATION.equals("4")) ||
	  (commonInfo.STAFF_QUALIFICATION.equals("8")) ||
	  (commonInfo.STAFF_QUALIFICATION.equals("9")) ||
	  (commonInfo.STAFF_CODE.equals(teacherCode))) {
	int cnt = commonMethods.setTeacherEnglishName(whereParamValues); 
	if (cnt != 0) {
	  tableView.refreshTable();
	}
      } else {
	commonInfo.showMessage("あなたには英文表記を設定する権限が与えられていません。");
      }  
    } else {   
      if ((commonInfo.STAFF_QUALIFICATION_D.equals("3")) ||
	  (commonInfo.STAFF_QUALIFICATION_D.equals("4")) ||
	  (commonInfo.STAFF_QUALIFICATION_D.equals("8")) ||
	  (commonInfo.STAFF_QUALIFICATION_D.equals("9")) ||
	  (commonInfo.STAFF_CODE_D.equals(teacherCode))) {
	int cnt = commonMethods.setTeacherEnglishName(whereParamValues); 
	if (cnt != 0) {
	  tableView.refreshTable();
	}
      } else {
	commonInfo.showMessage("あなたには英文表記を設定する権限が与えられていません。");
      }  
    }
  }
  
  public void setSubjectEnglishName(String param) {
    if (tableView.getSelectedRowCount() == 0) return;
    
    ArrayList<Object> list = new ArrayList<Object>();
    list.add(" 授業科目の英文表記を設定して下さい。 ");

    String columnCode;
    String  code = "", name = "", text = "";
    String teacherCode = "";

    String whereParams = "SUBJECT_CODE:TEACHER_CODE";
    String whereParamValues = "";
    StringTokenizer stk = new StringTokenizer(whereParams, ":");
    while (stk.hasMoreTokens()) {
      columnCode = stk.nextToken();
      if (parentTabbedPane.columnCodeMapContains(columnCode)) {
	code = parentTabbedPane.getValueFromColumnCodeMap(columnCode);
	name = parentTabbedPane.getDisplayFromColumnCodeMap(columnCode);
	if (code.equals("")) {
	  code = " ";
	}
      } else if (commonInfo.commonCodeMapContains(columnCode)) {
	code = commonInfo.getValueFromCommonCodeMap(columnCode);
	name = commonInfo.getDisplayFromCommonCodeMap(columnCode);
	if (code.equals("")) {
	  code = " ";
	}
      }
      text = " " + columnCode + " = " + code + " ( " + name + " )";
      if (columnCode.equals("TEACHER_CODE")) {
	teacherCode = code;
      }
      whereParamValues = whereParamValues + code + "|";
      JTextField editor = new JTextField(text);
      editor.setEditable(false); 
      editor.setFont(new Font("DialogInput", Font.PLAIN, 12));  
      editor.setBackground(Color.yellow);
      list.add(editor);
    }
  
    columnCode = "SUBJECT_NAME_ENGLISH"; 
    name = parentTabbedPane.getDisplayFromColumnCodeMap(columnCode);
    JTextField nameEditor = new JTextField(100);
    nameEditor.setFont(new Font("DialogInput", Font.PLAIN, 12));
    nameEditor.setText(" " + name);
    nameEditor.setEditable(true);
    nameEditor.setBorder(new TitledBorder(columnCode + " = " + name));
    list.add(nameEditor);
  
    int ans =  JOptionPane.showOptionDialog(commonInfo.getFrame(), 
					    list.toArray(), 
					    "授業科目の英文表記", 
					    JOptionPane.OK_CANCEL_OPTION,
					    JOptionPane.QUESTION_MESSAGE,
					    null,
					    null,
					    null);
    if (ans != JOptionPane.OK_OPTION) {
      return;
    }

    String subjectNameEnglish = nameEditor.getText().trim();
    if (!subjectNameEnglish.equals("")) {
      whereParamValues = whereParamValues + subjectNameEnglish + "|";
    } else {
      whereParamValues = whereParamValues + " |";
    }

    if (!commonInfo.structControlDebugMode) {  
      if ((commonInfo.STAFF_QUALIFICATION.equals("3")) ||
	  (commonInfo.STAFF_QUALIFICATION.equals("4")) ||
	  (commonInfo.STAFF_QUALIFICATION.equals("8")) ||
	  (commonInfo.STAFF_QUALIFICATION.equals("9")) ||
	  (commonInfo.STAFF_CODE.equals(teacherCode))) {
	int cnt = commonMethods.setSubjectEnglishName(whereParamValues);
	if (cnt != 0) {
	  tableView.refreshTable();
	}
      } else {
	commonInfo.showMessage("あなたには英文表記を設定する権限が与えられていません。");
      }    
    } else {
      if ((commonInfo.STAFF_QUALIFICATION_D.equals("3")) ||
	  (commonInfo.STAFF_QUALIFICATION_D.equals("4")) ||
	  (commonInfo.STAFF_QUALIFICATION_D.equals("8")) ||
	  (commonInfo.STAFF_QUALIFICATION_D.equals("9")) ||
	  (commonInfo.STAFF_CODE_D.equals(teacherCode))) {
	int cnt = commonMethods.setSubjectEnglishName(whereParamValues);
	if (cnt != 0) {
	  tableView.refreshTable();
	}
      } else {
	commonInfo.showMessage("あなたには英文表記を設定する権限が与えられていません。");
      }    
    }
  } 




  public void makeTextFileOfThisUnderGradJikanwari(String param) { 
    File f = null;
    if (fileChooser == null) {
      String homeDir = System.getProperty("user.home");
      fileChooser = new JFileChooser(homeDir);
    }

    String name = "";
    String titleText = "";
    String whereParams = "SCHOOL_YEAR:FACULTY:SEMESTER:GAKUNEN:WEEK";    
    StringTokenizer sstk = new StringTokenizer(whereParams, ":");
    while (sstk.hasMoreTokens()) {
      String columnCode = sstk.nextToken();
      if (parentTabbedPane.columnCodeMapContains(columnCode)) {
	name = parentTabbedPane.getDisplayFromColumnCodeMap(columnCode);
      } else if (commonInfo.commonCodeMapContains(columnCode)) {
	name = commonInfo.getDisplayFromCommonCodeMap(columnCode);
      }
      if (titleText.equals("")) titleText = name;
      else titleText = titleText + "-" + name;      
    }

    File dir = fileChooser.getCurrentDirectory();
    String fname = titleText + ".txt";
    
    f = new File(dir, fname);
    fileChooser.setSelectedFile(f);

    int ret = fileChooser.showSaveDialog(commonInfo.getFrame());
    if (ret == JFileChooser.APPROVE_OPTION) {
      f = fileChooser.getSelectedFile(); 
    } else {
      return;
    }

    StringBuffer sbuf = new StringBuffer();  
    sbuf.append(titleText).append("  時間割表のデータ").append(lineSeparator);
    sbuf.append("作成日： "+commonInfo.thisYear+"年").append(""+commonInfo.thisMonth+"月").append(""+commonInfo.thisDay+"日").append(lineSeparator).append(lineSeparator);
    makeUnderGradJikanwariInfoText(sbuf);
   
    try {
      PrintWriter fout = new PrintWriter(new BufferedWriter(new FileWriter(f)));
      fout.print(sbuf.toString());
      fout.close();
    } catch (Exception e) {
      commonInfo.showMessage(e.toString());
      return;
    }
  }

  public void makeTextFileOfThisGraduateJikanwari(String param) { 
    File f = null;
    if (fileChooser == null) {
      String homeDir = System.getProperty("user.home");
      fileChooser = new JFileChooser(homeDir);
    }

    String name = "";
    String titleText = "";
    String whereParams = "SCHOOL_YEAR:FACULTY:SEMESTER:GAKUNEN:WEEK";    
    StringTokenizer sstk = new StringTokenizer(whereParams, ":");
    while (sstk.hasMoreTokens()) {
      String columnCode = sstk.nextToken();
      if (parentTabbedPane.columnCodeMapContains(columnCode)) {
	name = parentTabbedPane.getDisplayFromColumnCodeMap(columnCode);
      } else if (commonInfo.commonCodeMapContains(columnCode)) {
	name = commonInfo.getDisplayFromCommonCodeMap(columnCode);
      }
      if (titleText.equals("")) titleText = name;
      else titleText = titleText + "-" + name;   
    }

    File dir = fileChooser.getCurrentDirectory();
    String fname = titleText + ".txt";
    
    f = new File(dir, fname);
    fileChooser.setSelectedFile(f);

    int ret = fileChooser.showSaveDialog(commonInfo.getFrame());
    if (ret == JFileChooser.APPROVE_OPTION) {
      f = fileChooser.getSelectedFile(); 
    } else {
      return;
    }

    StringBuffer sbuf = new StringBuffer();  
    sbuf.append(titleText).append("  時間割表のデータ").append(lineSeparator);  
    sbuf.append("作成日： "+commonInfo.thisYear+"年").append(""+commonInfo.thisMonth+"月").append(""+commonInfo.thisDay+"日").append(lineSeparator).append(lineSeparator); 
    makeGraduateJikanwariInfoText(sbuf);
   
    try {
      PrintWriter fout = new PrintWriter(new BufferedWriter(new FileWriter(f)));
      fout.print(sbuf.toString());
      fout.close();
    } catch (Exception e) {
      commonInfo.showMessage(e.toString());
      return;
    }
  }

  private void makeUnderGradJikanwariInfoText(StringBuffer sbuf) {
    String[] deptName = { " ", "知能情報", "電子情報", "システム創成", "機械情報", "生命情報" };
    String[] hourName = { " ", "1限目", "2限目", "3限目", "4限目", "5限目", "6限目" };

    JikanwariModelBase jikanwariModel = jikanwariView.getJikanwariModel();

    String titleText;
    for (int col = 1; col <= 6; col++) {
      for (int row = 1; row <= 5; row++) {
	titleText = deptName[row] + " : " + hourName[col];
	sbuf.append("**** [ ").append(titleText).append(" ] ****").append(lineSeparator).append(lineSeparator);
	ArrayList<String> komaInfo = jikanwariModel.getKomaInfo(row, col);
	for (int i = 0; i < komaInfo.size(); i++) {
	  String subjectInfo = komaInfo.get(i);
	  if (subjectInfo.equals("empty")) {
	    sbuf.append("    ").append(lineSeparator);
	  } else {

	    StringTokenizer stk = new StringTokenizer(subjectInfo, "|");
	    String SCHOOL_YEAR = stk.nextToken();
	    String SEMESTER = stk.nextToken();
	    String FACULTY = stk.nextToken();
	    String DEPARTMENT = stk.nextToken();
	    String GAKUNEN = stk.nextToken();
	    String WEEK = stk.nextToken();
	    String HOUR = stk.nextToken();
	    String SUBJECT_CODE = stk.nextToken();
	    String TEACHER_CODE = stk.nextToken();
	    String CLASS_CODE = stk.nextToken();
	    String KUBUN_CODE = stk.nextToken();
	    String REQ_CODE = stk.nextToken();
	    String UNIT = stk.nextToken();
	    String ROOM = stk.nextToken();
	    String SUBJECT_NAME = stk.nextToken();
	    String TEACHER_NAME = stk.nextToken();
	    String FLAG = stk.nextToken();

	    String roomName = commonInfo.getGakumuCodeShorterName("ROOM", ROOM);
	    String reqName = commonInfo.getGakumuCodeShorterName("REQ_CODE", REQ_CODE);
	    String kubunName = commonInfo.getGakumuCodeShorterName("KUBUN_CODE", KUBUN_CODE);
	    StringTokenizer stk3 = new StringTokenizer(TEACHER_NAME, "　 ");
	    String teanerName2 = stk3.nextToken();

	    sbuf.append(SUBJECT_NAME).append(lineSeparator);
	    sbuf.append("(").append(CLASS_CODE).append(") ");
	    sbuf.append("(").append(teanerName2).append(")").append(lineSeparator);
	    sbuf.append(roomName).append(lineSeparator);
	    sbuf.append(reqName).append(" : ").append(kubunName).append(" : ").append(TEACHER_NAME).append(lineSeparator).append(lineSeparator);
	  }
	}
      }
    }	 
  }

  private void makeGraduateJikanwariInfoText(StringBuffer sbuf) {
    String[] deptName = { " ", "共通科目", "情報科学", "情報システム", "情報創成" };
    String[] hourName = { " ", "1限目", "2限目", "3限目", "4限目", "5限目", "6限目" };

    JikanwariModelBase jikanwariModel = jikanwariView.getJikanwariModel();

    String titleText;
    for (int col = 1; col <= 6; col++) {
      for (int row = 1; row <= 4; row++) {
	titleText = deptName[row] + " : " + hourName[col];
	sbuf.append("**** [ ").append(titleText).append(" ] ****").append(lineSeparator).append(lineSeparator);
	ArrayList<String> komaInfo = jikanwariModel.getKomaInfo(row, col);
	for (int i = 0; i < komaInfo.size(); i++) {
	  String subjectInfo = komaInfo.get(i);
	  if (subjectInfo.equals("empty")) {
	    sbuf.append(lineSeparator);
	  } else {

	    StringTokenizer stk = new StringTokenizer(subjectInfo, "|");
	    String SCHOOL_YEAR = stk.nextToken();
	    String SEMESTER = stk.nextToken();
	    String FACULTY = stk.nextToken();
	    String DEPARTMENT = stk.nextToken();
	    String GAKUNEN = stk.nextToken();
	    String WEEK = stk.nextToken();
	    String HOUR = stk.nextToken();
	    String SUBJECT_CODE = stk.nextToken();
	    String TEACHER_CODE = stk.nextToken();
	    String CLASS_CODE = stk.nextToken();
	    String KUBUN_CODE = stk.nextToken();
	    String REQ_CODE = stk.nextToken();
	    String UNIT = stk.nextToken();
	    String ROOM = stk.nextToken();
	    String SUBJECT_NAME = stk.nextToken();
	    String TEACHER_NAME = stk.nextToken();
	    String FLAG = stk.nextToken();

	    String roomName = commonInfo.getGakumuCodeShorterName("ROOM", ROOM);
	    String reqName = commonInfo.getGakumuCodeShorterName("REQ_CODE", REQ_CODE);
	    String kubunName = commonInfo.getGakumuCodeShorterName("KUBUN_CODE", KUBUN_CODE);
	    StringTokenizer stk3 = new StringTokenizer(TEACHER_NAME, "　 ");
	    String teanerName2 = stk3.nextToken();

	    sbuf.append(SUBJECT_NAME).append(lineSeparator);
	    sbuf.append("(").append(CLASS_CODE).append(") ");
	    sbuf.append("(").append(teanerName2).append(")").append(lineSeparator);
	    sbuf.append(roomName).append(lineSeparator);
	    sbuf.append(reqName).append(" : ").append(kubunName).append(" : ").append(TEACHER_NAME).append(lineSeparator).append(lineSeparator);
	  }
	}
      }
    }	 
  }

  private String getSelectedGakubuKomaInfo() {
    String[] deptCodeArray = { " ", "31", "32", "33", "34", "35" };
    String[] deptNameArray = { " ", "知能情報", "電子情報", "システム創成", "機械情報", "生命情報" };
    String[] hourCodeArray = { "0", "1", "2", "3", "4", "5", "6" };
    String[] hourNameArray = { " ", "1限目", "2限目", "3限目", "4限目", "5限目", "6限目" };

    String whereParamValues = "";
    String whereParams;
    String columnCode;
    String code = "", name = "", text;
    
    ArrayList<Object> list = new ArrayList<Object>();
    list.add(" 次の時間割表の「コマ」が選択されている状態にあります。 ");
    list.add("  ");

    whereParams = "SCHOOL_YEAR:FACULTY:SEMESTER:GAKUNEN:WEEK";
    StringTokenizer stk = new StringTokenizer(whereParams, ":");
    while (stk.hasMoreTokens()) {
      columnCode = stk.nextToken();
      if (parentTabbedPane.columnCodeMapContains(columnCode)) {
	code = parentTabbedPane.getValueFromColumnCodeMap(columnCode);
	name = parentTabbedPane.getDisplayFromColumnCodeMap(columnCode);
	if (code.equals("")) {
	  code = " ";
	}
      } else if (commonInfo.commonCodeMapContains(columnCode)) {
	code = commonInfo.getValueFromCommonCodeMap(columnCode);
	name = commonInfo.getDisplayFromCommonCodeMap(columnCode);
	if (code.equals("")) {
	  code = " ";
	}
      }
      if (code == null) {
	commonInfo.showMessageLong("追加データを特定するパラメータ値が設定されていません。$ " + columnCode);	  
	return null;
      }
      if (code.equals(name)) {
	text = " " + columnCode + " = " + code;
      } else {
	text = " " + columnCode + " = " + code + " ( " + name + " )";
      }
      whereParamValues = whereParamValues + code + "|";
      JTextField editor = new JTextField(text);
      editor.setEditable(false); 
      editor.setFont(new Font("DialogInput", Font.PLAIN, 12));  
      editor.setBackground(Color.yellow);
      list.add(editor);
    }
    int row, col;
    try {
      String rowSel = parentTabbedPane.getValueFromColumnCodeMap("ROW_SELECTED");
      String colSel = parentTabbedPane.getValueFromColumnCodeMap("COL_SELECTED");
      row = Integer.parseInt(rowSel);
      col = Integer.parseInt(colSel);
    } catch (Exception e) {
      return null;
    }
    String dept = deptCodeArray[row];
    String hour = hourCodeArray[col];
    whereParamValues =  whereParamValues + dept + "|" + hour + "|";
    String deptName  = deptNameArray[row];
    String hourName  = hourNameArray[col];
    
    text = " DEPARTMENT = " + dept + " ( " + deptName + " )";
    JTextField editor1 = new JTextField(text);
    editor1.setEditable(false); 
    editor1.setFont(new Font("DialogInput", Font.PLAIN, 12));  
    editor1.setBackground(Color.yellow);
    list.add(editor1);
    
    text = " HOUR = " + hour + " ( " + hourName + " )";
    JTextField editor2 = new JTextField(text);
    editor2.setEditable(false); 
    editor2.setFont(new Font("DialogInput", Font.PLAIN, 12));  
    editor2.setBackground(Color.yellow);
    list.add(editor2);
    
    int ans =  JOptionPane.showOptionDialog(commonInfo.getFrame(), 
					    list.toArray(), 
					    "時間割コマの確認", 
					    JOptionPane.OK_CANCEL_OPTION,
					    JOptionPane.QUESTION_MESSAGE,
					    null,
					    null,
					    null);
    if (ans != JOptionPane.OK_OPTION) {
      return null;
    }
    // whereParamValues = SCHOOL_YEAR|FACULTY|SEMESTER|GAKUNEN|WEEK|DEPARTMENT|HOUR|
    return whereParamValues;
  }


  private String getSelectedGraduateKomaInfo() {
    String[] deptCodeArray = { " ", "70", "73", "74", "75" };
    String[] deptNameArray = { " ", "共通科目", "情報科学", "情報システム", "情報創成" };
    String[] hourCodeArray = { "0", "1", "2", "3", "4", "5", "6" };
    String[] hourNameArray = { " ", "1限目", "2限目", "3限目", "4限目", "5限目", "6限目" };

    String whereParamValues = "";
    String whereParams;
    String columnCode;
    String code = "", name = "", text;
    
    ArrayList<Object> list = new ArrayList<Object>();
    list.add(" 次の時間割表の「コマ」が選択されている状態にあります。 ");
    list.add("  ");

    whereParams = "SCHOOL_YEAR:FACULTY:SEMESTER:GAKUNEN:WEEK";
    StringTokenizer stk = new StringTokenizer(whereParams, ":");
    while (stk.hasMoreTokens()) {
      columnCode = stk.nextToken();
      if (parentTabbedPane.columnCodeMapContains(columnCode)) {
	code = parentTabbedPane.getValueFromColumnCodeMap(columnCode);
	name = parentTabbedPane.getDisplayFromColumnCodeMap(columnCode);
	if (code.equals("")) {
	  code = " ";
	}
      } else if (commonInfo.commonCodeMapContains(columnCode)) {
	code = commonInfo.getValueFromCommonCodeMap(columnCode);
	name = commonInfo.getDisplayFromCommonCodeMap(columnCode);
	if (code.equals("")) {
	  code = " ";
	}
      }
      if (code == null) {
	commonInfo.showMessageLong("追加データを特定するパラメータ値が設定されていません。$ " + columnCode);	  
	return null;
      }
      if (code.equals(name)) {
	text = " " + columnCode + " = " + code;
      } else {
	text = " " + columnCode + " = " + code + " ( " + name + " )";
      }
      whereParamValues = whereParamValues + code + "|";
      JTextField editor = new JTextField(text);
      editor.setEditable(false); 
      editor.setFont(new Font("DialogInput", Font.PLAIN, 12));  
      editor.setBackground(Color.yellow);
      list.add(editor);
    }
    int row, col;
    try {
      String rowSel = parentTabbedPane.getValueFromColumnCodeMap("ROW_SELECTED");
      String colSel = parentTabbedPane.getValueFromColumnCodeMap("COL_SELECTED");
      row = Integer.parseInt(rowSel);
      col = Integer.parseInt(colSel);
    } catch (Exception e) {
      return null;
    }
    String dept = deptCodeArray[row];
    String hour = hourCodeArray[col];
    whereParamValues =  whereParamValues + dept + "|" + hour + "|";
    String deptName  = deptNameArray[row];
    String hourName  = hourNameArray[col];
    
    text = " DEPARTMENT = " + dept + " ( " + deptName + " )";
    JTextField editor1 = new JTextField(text);
    editor1.setEditable(false); 
    editor1.setFont(new Font("DialogInput", Font.PLAIN, 12));  
    editor1.setBackground(Color.yellow);
    list.add(editor1);
    
    text = " HOUR = " + hour + " ( " + hourName + " )";
    JTextField editor2 = new JTextField(text);
    editor2.setEditable(false); 
    editor2.setFont(new Font("DialogInput", Font.PLAIN, 12));  
    editor2.setBackground(Color.yellow);
    list.add(editor2);
    
    int ans =  JOptionPane.showOptionDialog(commonInfo.getFrame(), 
					    list.toArray(), 
					    "時間割コマの確認", 
					    JOptionPane.OK_CANCEL_OPTION,
					    JOptionPane.QUESTION_MESSAGE,
					    null,
					    null,
					    null);
    if (ans != JOptionPane.OK_OPTION) {
      return null;
    }
    // whereParamValues = SCHOOL_YEAR|FACULTY|SEMESTER|GAKUNEN|WEEK|DEPARTMENT|HOUR|
    return whereParamValues;
  }


  private String getSelectedSubjectClassInfo() {
    ArrayList<Object> list = new ArrayList<Object>();
    JComponent comp = commonInfo.getSelector("ClassInfoSelector", null, null, null, null, 1); 
    list.add(comp);
    int ans =  JOptionPane.showOptionDialog(commonInfo.getFrame(),
					    list.toArray(), 
					    "追加するクラス", 
					    JOptionPane.OK_CANCEL_OPTION,
					    JOptionPane.QUESTION_MESSAGE,
					    null,
					    null,
					    null);
    if (ans == JOptionPane.OK_OPTION) {
      String params1 = "SCHOOL_YEAR#FACULTY"; 
      String paramValues1 = commonInfo.getSelectorValue("ClassInfoSelector", params1);      
      String schoolYear1 = parentTabbedPane.getValueFromColumnCodeMap("SCHOOL_YEAR");
      String faculty1 = parentTabbedPane.getValueFromColumnCodeMap("FACULTY");
      String text1 = schoolYear1+"|"+faculty1+"|";
      if (paramValues1.equals(text1)) {
	String params2 = "SUBJECT_CODE#CLASS_CODE"; 
	String paramValues2 = commonInfo.getSelectorValue("ClassInfoSelector", params2);
	// paramValues2 = SUBJECT_CODE|CLASS_CODE|
	return paramValues2;
      } else {
	commonInfo.showMessage("「時間割」と「開講クラス表」の開講年度または学部が一致していません。");	  
	return null;
      }
    } else {
      return null;
    }
  }  

  private String getSelectedRoom() {
    ArrayList<Object> list = new ArrayList<Object>();
    JComponent comp = commonInfo.getSelector("RoomSelector", null, null, null, null, 1); 
    list.add(comp);
    int ans =  JOptionPane.showOptionDialog(commonInfo.getFrame(),
					    list.toArray(), 
					    "設定する講義室", 
					    JOptionPane.OK_CANCEL_OPTION,
					    JOptionPane.QUESTION_MESSAGE,
					    null,
					    null,
					    null);
    if (ans == JOptionPane.OK_OPTION) {
      String room = commonInfo.getSelectorValue("RoomSelector", "ROOM"); 
      return room;
    } else {
      return null;
    }
  }  

  private String getSelectedGakunen() {
    ArrayList<Object> list = new ArrayList<Object>();
    JComponent comp = commonInfo.getSelector("GakunenSelector", null, null, null, null, 1); 
    list.add(comp);
    int ans =  JOptionPane.showOptionDialog(commonInfo.getFrame(),
					    list.toArray(), 
					    "設定する学年", 
					    JOptionPane.OK_CANCEL_OPTION,
					    JOptionPane.QUESTION_MESSAGE,
					    null,
					    null,
					    null);
    if (ans == JOptionPane.OK_OPTION) {
      String gakunen = commonInfo.getSelectorValue("GakunenSelector", "GAKUNEN"); 
      return gakunen;
    } else {
      return null;
    }
  }  

  private String getSelectedJikanwariSubjectClass() {
    String ret = parentTabbedPane.getValueFromColumnCodeMap("EMPTY_KOMA_SELECTED");
    if (ret.equals("true")) return null;
    
    String whereParams;
    String whereParamValues = "";
    String columnCode;
    String code = "", name = "", text;

    ArrayList<Object> list = new ArrayList<Object>();
    list.add(" 次の時間割表の「コマ」が選択されている状態にあります。 ");
    list.add("  ");

    whereParams = "SCHOOL_YEAR:FACULTY:SEMESTER:GAKUNEN:WEEK:SUBJECT_CODE:CLASS_CODE";
    StringTokenizer stk = new StringTokenizer(whereParams, ":");
    while (stk.hasMoreTokens()) {
      columnCode = stk.nextToken();
      if (parentTabbedPane.columnCodeMapContains(columnCode)) {
	code = parentTabbedPane.getValueFromColumnCodeMap(columnCode);
	name = parentTabbedPane.getDisplayFromColumnCodeMap(columnCode);
	if (code.equals("")) {
	  code = " ";
	}
      } else if (commonInfo.commonCodeMapContains(columnCode)) {
	code = commonInfo.getValueFromCommonCodeMap(columnCode);
	name = commonInfo.getDisplayFromCommonCodeMap(columnCode);
	if (code.equals("")) {
	  code = " ";
	}
      }
      if (code == null) {
	commonInfo.showMessageLong("削除データを特定するパラメータ値が設定されていません。$ " + columnCode);	  
	return null;
      }
      if (code.equals(name)) {
	text = " " + columnCode + " = " + code;
      } else {
	text = " " + columnCode + " = " + code + " ( " + name + " )";
      }
      whereParamValues = whereParamValues + code + "|";
      JTextField editor = new JTextField(text);
      editor.setEditable(false); 
      editor.setFont(new Font("DialogInput", Font.PLAIN, 12));  
      editor.setBackground(Color.yellow);
      list.add(editor);
    }

    String[] deptCodeArray = { " ", "31", "32", "33", "34", "35" };
    String[] deptNameArray = { " ", "知能情報", "電子情報", "システム創成", "機械情報", "生命情報" };
    String[] hourCodeArray = { "0", "1", "2", "3", "4", "5", "6" };
    String[] hourNameArray = { " ", "1限目", "2限目", "3限目", "4限目", "5限目", "6限目" };
   
    int row, col;
    try {
      String rowSel = parentTabbedPane.getValueFromColumnCodeMap("ROW_SELECTED");
      String colSel = parentTabbedPane.getValueFromColumnCodeMap("COL_SELECTED");
      row = Integer.parseInt(rowSel);
      col = Integer.parseInt(colSel);
    } catch (Exception e) {
      return null;
    }
    String dept = deptCodeArray[row];
    String hour = hourCodeArray[col];
    whereParamValues =  whereParamValues + dept + "|" + hour + "|";
    String deptName  = deptNameArray[row];
    String hourName  = hourNameArray[col];
    
    text = " DEPARTMENT = " + dept + " ( " + deptName + " )";
    JTextField editor1 = new JTextField(text);
    editor1.setEditable(false); 
    editor1.setFont(new Font("DialogInput", Font.PLAIN, 12));  
    editor1.setBackground(Color.yellow);
    list.add(editor1);
    
    text = " HOUR = " + hour + " ( " + hourName + " )";
    JTextField editor2 = new JTextField(text);
    editor2.setEditable(false); 
    editor2.setFont(new Font("DialogInput", Font.PLAIN, 12));  
    editor2.setBackground(Color.yellow);
    list.add(editor2);
    
    int ans =  JOptionPane.showOptionDialog(commonInfo.getFrame(), 
					    list.toArray(), 
					    "時間割コマの確認", 
					    JOptionPane.OK_CANCEL_OPTION,
					    JOptionPane.QUESTION_MESSAGE,
					    null,
					    null,
					    null);
    if (ans != JOptionPane.OK_OPTION) {
      return null;
    }
    // whereParamValues = SCHOOL_YEAR|FACULTY|SEMESTER|GAKUNEN|WEEK|SUBJECT_CODE|CLASS_CODE|DEPARTMENT|HOUR|
    return whereParamValues;
  }
 

  private String getSelectedJikanwariListSubjectClass() {  
    if (tableView.getSelectedRowCount() == 0) return null;
    
    String whereParams;
    String whereParamValues = "";
    String columnCode;
    String code = "", name = "", text;

    ArrayList<Object> list = new ArrayList<Object>();
    list.add(" 次の時間割表の「コマ」が選択されている状態にあります。 ");
    list.add("  ");

    whereParams = "SCHOOL_YEAR:FACULTY:SEMESTER:GAKUNEN:WEEK:SUBJECT_CODE:CLASS_CODE:DEPARTMENT:HOUR";
    StringTokenizer stk = new StringTokenizer(whereParams, ":");
    while (stk.hasMoreTokens()) {
      columnCode = stk.nextToken();
      if (parentTabbedPane.columnCodeMapContains(columnCode)) {
	code = parentTabbedPane.getValueFromColumnCodeMap(columnCode);
	name = parentTabbedPane.getDisplayFromColumnCodeMap(columnCode);
	if (code.equals("")) {
	  code = " ";
	}
      } else if (commonInfo.commonCodeMapContains(columnCode)) {
	code = commonInfo.getValueFromCommonCodeMap(columnCode);
	name = commonInfo.getDisplayFromCommonCodeMap(columnCode);
	if (code.equals("")) {
	  code = " ";
	}
      }
      if (code == null) {
	commonInfo.showMessageLong("データを特定するパラメータ値が設定されていません。$ " + columnCode);	  
	return null;
      }
      if (code.equals(name)) {
	text = " " + columnCode + " = " + code;
      } else {
	text = " " + columnCode + " = " + code + " ( " + name + " )";
      }
      whereParamValues = whereParamValues + code + "|";
      JTextField editor = new JTextField(text);
      editor.setEditable(false); 
      editor.setFont(new Font("DialogInput", Font.PLAIN, 12));  
      editor.setBackground(Color.yellow);
      list.add(editor);
    }
       
    int ans =  JOptionPane.showOptionDialog(commonInfo.getFrame(), 
					    list.toArray(), 
					    "時間割コマの確認", 
					    JOptionPane.OK_CANCEL_OPTION,
					    JOptionPane.QUESTION_MESSAGE,
					    null,
					    null,
					    null);
    if (ans != JOptionPane.OK_OPTION) {
      return null;
    }
    // whereParamValues = SCHOOL_YEAR|FACULTY|SEMESTER|GAKUNEN|WEEK|SUBJECT_CODE|CLASS_CODE|DEPARTMENT|HOUR|
    return whereParamValues;
  }

  private String getSelectedGraduateJikanwariSubjectClass() {
    String ret = parentTabbedPane.getValueFromColumnCodeMap("EMPTY_KOMA_SELECTED");
    if (ret.equals("true")) return null;
    
    String whereParams;
    String whereParamValues = "";
    String columnCode;
    String code = "", name = "", text;

    ArrayList<Object> list = new ArrayList<Object>();
    list.add(" 次の時間割表の「コマ」が選択されている状態にあります。 ");
    list.add("  ");

    whereParams = "SCHOOL_YEAR:FACULTY:SEMESTER:GAKUNEN:WEEK:SUBJECT_CODE:CLASS_CODE";
    StringTokenizer stk = new StringTokenizer(whereParams, ":");
    while (stk.hasMoreTokens()) {
      columnCode = stk.nextToken();
      if (parentTabbedPane.columnCodeMapContains(columnCode)) {
	code = parentTabbedPane.getValueFromColumnCodeMap(columnCode);
	name = parentTabbedPane.getDisplayFromColumnCodeMap(columnCode);
	if (code.equals("")) {
	  code = " ";
	}
      } else if (commonInfo.commonCodeMapContains(columnCode)) {
	code = commonInfo.getValueFromCommonCodeMap(columnCode);
	name = commonInfo.getDisplayFromCommonCodeMap(columnCode);
	if (code.equals("")) {
	  code = " ";
	}
      }
      if (code == null) {
	commonInfo.showMessageLong("削除データを特定するパラメータ値が設定されていません。$ " + columnCode);	  
	return null;
      }
      if (code.equals(name)) {
	text = " " + columnCode + " = " + code;
      } else {
	text = " " + columnCode + " = " + code + " ( " + name + " )";
      }
      whereParamValues = whereParamValues + code + "|";
      JTextField editor = new JTextField(text);
      editor.setEditable(false); 
      editor.setFont(new Font("DialogInput", Font.PLAIN, 12));  
      editor.setBackground(Color.yellow);
      list.add(editor);
    }

    String[] deptCodeArray = { " ", "70", "73", "74", "75" };
    String[] deptNameArray = { " ", "共通科目", "情報科学", "情報システム", "情報創成" };
    String[] hourCodeArray = { "0", "1", "2", "3", "4", "5", "6" };
    String[] hourNameArray = { " ", "1限目", "2限目", "3限目", "4限目", "5限目", "6限目" };
   
    int row, col;
    try {
      String rowSel = parentTabbedPane.getValueFromColumnCodeMap("ROW_SELECTED");
      String colSel = parentTabbedPane.getValueFromColumnCodeMap("COL_SELECTED");
      row = Integer.parseInt(rowSel);
      col = Integer.parseInt(colSel);
    } catch (Exception e) {
      return null;
    }
    String dept = deptCodeArray[row];
    String hour = hourCodeArray[col];
    whereParamValues =  whereParamValues + dept + "|" + hour + "|";
    String deptName  = deptNameArray[row];
    String hourName  = hourNameArray[col];
    
    text = " DEPARTMENT = " + dept + " ( " + deptName + " )";
    JTextField editor1 = new JTextField(text);
    editor1.setEditable(false); 
    editor1.setFont(new Font("DialogInput", Font.PLAIN, 12));  
    editor1.setBackground(Color.yellow);
    list.add(editor1);
    
    text = " HOUR = " + hour + " ( " + hourName + " )";
    JTextField editor2 = new JTextField(text);
    editor2.setEditable(false); 
    editor2.setFont(new Font("DialogInput", Font.PLAIN, 12));  
    editor2.setBackground(Color.yellow);
    list.add(editor2);
    
    int ans =  JOptionPane.showOptionDialog(commonInfo.getFrame(), 
					    list.toArray(), 
					    "時間割コマの確認", 
					    JOptionPane.OK_CANCEL_OPTION,
					    JOptionPane.QUESTION_MESSAGE,
					    null,
					    null,
					    null);
    if (ans != JOptionPane.OK_OPTION) {
      return null;
    }
    // whereParamValues = SCHOOL_YEAR|FACULTY|SEMESTER|GAKUNEN|WEEK|SUBJECT_CODE|CLASS_CODE|DEPARTMENT|HOUR|
    return whereParamValues;
  }

  private String getSelectedShuchuSubjectClass() {   
    if (tableView.getSelectedRowCount() == 0) return null;

    String whereParams;
    String whereParamValues = "";
    String columnCode;
    String code = "", name = "", text;

    whereParams = "SCHOOL_YEAR:FACULTY:SEMESTER:DEPARTMENT:GAKUNEN:SUBJECT_CODE:CLASS_CODE";
    StringTokenizer stk = new StringTokenizer(whereParams, ":");
    while (stk.hasMoreTokens()) {
      columnCode = stk.nextToken();
      if (parentTabbedPane.columnCodeMapContains(columnCode)) {
	code = parentTabbedPane.getValueFromColumnCodeMap(columnCode);
	name = parentTabbedPane.getDisplayFromColumnCodeMap(columnCode);
	if (code.equals("")) {
	  code = " ";
	}
      } else if (commonInfo.commonCodeMapContains(columnCode)) {
	code = commonInfo.getValueFromCommonCodeMap(columnCode);
	name = commonInfo.getDisplayFromCommonCodeMap(columnCode);
	if (code.equals("")) {
	  code = " ";
	}
      }
      if (code == null) {
	commonInfo.showMessageLong("削除データを特定するパラメータ値が設定されていません。$ " + columnCode);	  
	return null;
      }
      whereParamValues = whereParamValues + code + "|";
    }
    
    // whereParamValues = SCHOOL_YEAR|FACULTY|SEMESTER|DEPARTMENT|GAKUNEN|SUBJECT_CODE|CLASS_CODE|
    return whereParamValues;
  }
  
  private String getSelectedShuchuDept() {   
    String whereParams;
    String whereParamValues = "";
    String columnCode;
    String code = "", name = "", text;

    whereParams = "SCHOOL_YEAR:FACULTY:SEMESTER:DEPARTMENT";
    StringTokenizer stk = new StringTokenizer(whereParams, ":");
    while (stk.hasMoreTokens()) {
      columnCode = stk.nextToken();
      if (parentTabbedPane.columnCodeMapContains(columnCode)) {
	code = parentTabbedPane.getValueFromColumnCodeMap(columnCode);
	name = parentTabbedPane.getDisplayFromColumnCodeMap(columnCode);
	if (code.equals("")) {
	  code = " ";
	}
      } else if (commonInfo.commonCodeMapContains(columnCode)) {
	code = commonInfo.getValueFromCommonCodeMap(columnCode);
	name = commonInfo.getDisplayFromCommonCodeMap(columnCode);
	if (code.equals("")) {
	  code = " ";
	}
      }
      if (code == null) {
	commonInfo.showMessageLong("集中講義を特定するパラメータ値が設定されていません。$ " + columnCode);	  
	return null;
      }
      whereParamValues = whereParamValues + code + "|";
    }
    
    // whereParamValues = SCHOOL_YEAR|FACULTY|SEMESTER|DEPARTMENT|
    return whereParamValues;
  }
  
  public void addClassToGakunenWeekJikanwari(String param) { 
    String paramValues1 = getSelectedGakubuKomaInfo();
    if (paramValues1 == null) return;
    String paramValues2 = getSelectedSubjectClassInfo();
    if (paramValues2 == null) return;

    String paramValues = paramValues1 + paramValues2;
    // paramValues = SCHOOL_YEAR|FACULTY|SEMESTER|GAKUNEN|WEEK|DEPARTMENT|HOUR|SUBJECT_CODE|CLASS_CODE|
    int cnt = commonMethods.addSubjectClassToJikanwari(paramValues);
    if (cnt != 0) {
      jikanwariView.refreshTable();
    } else {
      commonInfo.showMessage("記入しようとるす科目は当該学科の履修課程表に存在しません。");	  
      return;
    }
  }

  public void addClassToGakunenWeekJikanwari2(String param) { 
    int totalCnt = 0;
    String[] deptCodeArray = { " ", "31", "32", "33", "34", "35" };
    String[] deptNameArray = { " ", "知能情報", "電子情報", "システム創成", "機械情報", "生命情報" };

    String paramValues1 = getSelectedGakubuKomaInfo();
    if (paramValues1 == null) return;
    // paramValues1 = SCHOOL_YEAR|FACULTY|SEMESTER|GAKUNEN|WEEK|DEPARTMENT|HOUR|
    String paramValues2 = getSelectedSubjectClassInfo();
    if (paramValues2 == null) return;

    StringTokenizer stk = new StringTokenizer(paramValues1, "|");
    String schoolYear = stk.nextToken();
    String faculty = stk.nextToken();
    String semester = stk.nextToken();
    String gakunen = stk.nextToken();
    String week = stk.nextToken();
    String department = stk.nextToken();
    String hour = stk.nextToken();
    for (int i = 1; i <= 5; i++) {
      String deptCode = deptCodeArray[i];
      String deptName = deptNameArray[i];
      String paramValues3 = schoolYear+"|"+faculty+"|"+semester+"|"+gakunen+"|"+week+"|"+deptCode+"|"+hour+"|";
      String paramValues = paramValues3 + paramValues2;
      // paramValues = SCHOOL_YEAR|FACULTY|SEMESTER|GAKUNEN|WEEK|DEPARTMENT|HOUR|SUBJECT_CODE|CLASS_CODE|
      int cnt = commonMethods.addSubjectClassToJikanwari(paramValues);
      if (cnt == 0) {
	commonInfo.showMessage("記入しようとるす科目は " + deptName + " の履修課程表に存在しません。");	
      } else {
	totalCnt++;
      }
    }
    if (totalCnt > 0) {
      jikanwariView.refreshTable();
    }
  }

  public void delClassFromGakunenWeekJikanwari(String param) { 
    String paramValues = getSelectedJikanwariSubjectClass();
    if (paramValues == null) return;

    // paramValues = SCHOOL_YEAR|FACULTY|SEMESTER|GAKUNEN|WEEK|SUBJECT_CODE|CLASS_CODE|DEPARTMENT|HOUR|
    int cnt = commonMethods.deleteSubjectClassFromJikanwari(paramValues);
    if (cnt != 0) {
      jikanwariView.refreshTable();
    } else {
      commonInfo.showMessage("指定された科目は時間割表に配置されていません。");	  
      return;
    }
  }

  public void delClassFromGakunenWeekJikanwari2(String param) { 
    String paramValues = getSelectedJikanwariListSubjectClass();
    if (paramValues == null) return;

    // paramValues = SCHOOL_YEAR|FACULTY|SEMESTER|GAKUNEN|WEEK|SUBJECT_CODE|CLASS_CODE|DEPARTMENT|HOUR|
    int cnt = commonMethods.deleteSubjectClassFromJikanwari(paramValues);
    if (cnt != 0) {
      tableView.refreshTable();
    } else {
      commonInfo.showMessage("指定された科目は時間割表に配置されていません。");	  
      return;
    }
  }


  public void delSelectedSubjectClassFromJikanwari(String param) { // 選択した「クラス」を全学科・学年から削除 (時間割表) 
    String paramValues = getSelectedJikanwariSubjectClass();
    if (paramValues == null) return;

    // paramValues = SCHOOL_YEAR|FACULTY|SEMESTER|GAKUNEN|WEEK|SUBJECT_CODE|CLASS_CODE|DEPARTMENT|HOUR|
    int cnt = commonMethods.deleteSubjectClassFromJikanwari2(paramValues);
    if (cnt != 0) {
      jikanwariView.refreshTable();
    } else {
      commonInfo.showMessage("指定された科目は時間割表に配置されていません。");	  
      return;
    }
  }



  public void delSelectedSubjectClassFromJikanwari2(String param) { // 選択した「クラス」を全学科・学年から削除 (時間割リスト)
    String paramValues = getSelectedJikanwariListSubjectClass();
    if (paramValues == null) return;

    // paramValues = SCHOOL_YEAR|FACULTY|SEMESTER|GAKUNEN|WEEK|SUBJECT_CODE|CLASS_CODE|DEPARTMENT|HOUR|
    int cnt = commonMethods.deleteSubjectClassFromJikanwari2(paramValues);
    if (cnt != 0) {
      tableView.refreshTable();
    } else {
      commonInfo.showMessage("指定された科目は時間割表に配置されていません。");	  
      return;
    }
  }

  public void setRoomToSelectedSubjectClassOfJikanwari(String param) { 
    String paramValues1 = getSelectedJikanwariSubjectClass();
    if (paramValues1 == null) return;

    String room = getSelectedRoom();
    if (room == null) return;

    String paramValues = paramValues1 + room;
    // paramValues = SCHOOL_YEAR|FACULTY|SEMESTER|GAKUNEN|WEEK|SUBJECT_CODE|CLASS_CODE|DEPARTMENT|HOUR|ROOM|
    int cnt = commonMethods.setRoomToSubjectClassOfJikanwari2(paramValues); 
    if (cnt != 0) {
      jikanwariView.refreshTable();
    } else {
      commonInfo.showMessage("講義室が設定できません。");	  
      return;
    }
  }

  public void setRoomToSelectedSubjectClassOfJikanwari2(String param) { 
    String paramValues1 = getSelectedJikanwariListSubjectClass();
    if (paramValues1 == null) return;

    String room = getSelectedRoom();
    if (room == null) return;

    String paramValues = paramValues1 + room;
    // paramValues = SCHOOL_YEAR|FACULTY|SEMESTER|GAKUNEN|WEEK|SUBJECT_CODE|CLASS_CODE|DEPARTMENT|HOUR|ROOM|
    int cnt = commonMethods.setRoomToSubjectClassOfJikanwari2(paramValues); 
    if (cnt != 0) {
      tableView.refreshTable();
    } else {
      commonInfo.showMessage("講義室が設定できません。");	  
      return;
    }
  }

  public void setRoomToGakunenWeekJikanwari(String param) { 
    String paramValues1 = getSelectedJikanwariSubjectClass();
    if (paramValues1 == null) return;

    String room = getSelectedRoom();
    if (room == null) return;

    String paramValues = paramValues1 + room;
    // paramValues = SCHOOL_YEAR|FACULTY|SEMESTER|GAKUNEN|WEEK|SUBJECT_CODE|CLASS_CODE|DEPARTMENT|HOUR|ROOM|
    int cnt = commonMethods.setRoomToSubjectClassOfJikanwari(paramValues);
    if (cnt != 0) {
      jikanwariView.refreshTable();
    } else {
      commonInfo.showMessage("講義室が設定できません。");	  
      return;
    }
  }
  
  
  public void setRoomToGakunenWeekJikanwari2(String param) {  
    String paramValues1 = getSelectedJikanwariListSubjectClass();
    if (paramValues1 == null) return;

    String room = getSelectedRoom();
    if (room == null) return;

    String paramValues = paramValues1 + room;
    // paramValues = SCHOOL_YEAR|FACULTY|SEMESTER|GAKUNEN|WEEK|SUBJECT_CODE|CLASS_CODE|DEPARTMENT|HOUR|ROOM|
    int cnt = commonMethods.setRoomToSubjectClassOfJikanwari(paramValues);
    if (cnt != 0) {
      tableView.refreshTable();
    } else {
      commonInfo.showMessage("講義室が設定できません。");	  
      return;
    }
  }

  public void addClassToShuchuList(String param) { 
    String paramValues1 = getSelectedShuchuDept();
    if (paramValues1 == null) return;
    // paramValues1 = SCHOOL_YEAR|FACULTY|SEMESTER|DEPARTMENT|
    String paramValues2 = getSelectedGakunen();
    if (paramValues2 == null) return;
    // paramValues2 = GAKUNEN|
    
    String paramValues3 = getSelectedSubjectClassInfo();
    if (paramValues3 == null) return;
    // paramValues3 = SUBJECT_CODE|CLASS_CODE|
    
    String paramValues = paramValues1 + paramValues2 + paramValues3;
    // paramValues = SCHOOL_YEAR|FACULTY|SEMESTER|DEPARTMENT|GAKUNEN|SUBJECT_CODE|CLASS_CODE|
    int cnt = commonMethods.addSubjectClassToShuchuList(paramValues);
    if (cnt != 0) {
      tableView.refreshTable();
    } else {
      commonInfo.showMessage("記入しようとるす科目は当該学科の履修課程表に存在しません。");	  
      return;
    }
  }
    
  public void addClassToShuchuList2(String param) {
    int totalCnt = 0;
    String[] deptCodeArray = { " ", "31", "32", "33", "34", "35" };
    String[] deptNameArray = { " ", "知能情報", "電子情報", "システム創成", "機械情報", "生命情報" };

    String paramValues1 = getSelectedShuchuDept();
    if (paramValues1 == null) return;
    // paramValues1 = SCHOOL_YEAR|FACULTY|SEMESTER|DEPARTMENT|

    String paramValues2 = getSelectedGakunen();
    if (paramValues2 == null) return;
    // paramValues2 = GAKUNEN|
    
    String paramValues3 = getSelectedSubjectClassInfo();
    if (paramValues3 == null) return;
    // paramValues3 = SUBJECT_CODE|CLASS_CODE|

    StringTokenizer stk = new StringTokenizer(paramValues1, "|");
    String schoolYear = stk.nextToken();
    String faculty = stk.nextToken();
    String semester = stk.nextToken();

    for (int i = 1; i <= 5; i++) {
      String deptCode = deptCodeArray[i];
      String deptName = deptNameArray[i];
      String paramValues4 = schoolYear+"|"+faculty+"|"+semester+"|"+deptCode+"|";
      String paramValues = paramValues4 + paramValues2 + paramValues3;

      // paramValues = SCHOOL_YEAR|FACULTY|SEMESTER|DEPARTMENT|GAKUNEN|SUBJECT_CODE|CLASS_CODE|
      int cnt = commonMethods.addSubjectClassToShuchuList(paramValues);
      if (cnt == 0) {
	commonInfo.showMessage("記入しようとるす科目は " + deptName + " の履修課程表に存在しません。");	
      } else {
	totalCnt++;
      }
    }
    if (totalCnt > 0) {
      tableView.refreshTable();
    }
  }


  public void delSubjectClassFromShuchuList2(String param) { 
    String paramValues = getSelectedShuchuSubjectClass();
    if (paramValues == null) return;
    // paramValues = SCHOOL_YEAR|FACULTY|SEMESTER|DEPARTMENT|GAKUNEN|SUBJECT_CODE|CLASS_CODE|

    int cnt = commonMethods.deleteSubjectClassFromShuchuList2(paramValues); 
    if (cnt != 0) {
      tableView.refreshTable();
    } else {
      commonInfo.showMessage("指定された科目は時間割表に配置されていません。");	  
      return;
    }
  }


  public void delSubjectClassFromShuchuList(String param) { 
    String paramValues = getSelectedShuchuSubjectClass();
    if (paramValues == null) return;
    // paramValues = SCHOOL_YEAR|FACULTY|SEMESTER|DEPARTMENT|GAKUNEN|SUBJECT_CODE|CLASS_CODE|

    int cnt = commonMethods.deleteSubjectClassFromShuchuList(paramValues);
    if (cnt != 0) {
      tableView.refreshTable();
    } else {
      commonInfo.showMessage("指定された科目は時間割表に配置されていません。");	  
      return;
    }
  }



  public void addClassToGraduateGakunenWeekJikanwari(String param) {
    String paramValues1 = getSelectedGraduateKomaInfo();
    if (paramValues1 == null) return;
    String paramValues2 = getSelectedSubjectClassInfo();
    if (paramValues2 == null) return;

    String paramValues = paramValues1 + paramValues2;
    // paramValues = SCHOOL_YEAR|FACULTY|SEMESTER|GAKUNEN|WEEK|DEPARTMENT|HOUR|SUBJECT_CODE|CLASS_CODE|
    int cnt = commonMethods.addSubjectClassToGraduateJikanwari(paramValues);
    if (cnt != 0) {
      jikanwariView.refreshTable();
    } else {
      commonInfo.showMessage("記入しようとるす科目は当該学科の履修課程表に存在しません。");	  
      return;
    }
  }  


  public void delClassFromGraduateGakunenWeekJikanwari(String param) { 
    String paramValues = getSelectedGraduateJikanwariSubjectClass();
    if (paramValues == null) return;

    // paramValues = SCHOOL_YEAR|FACULTY|SEMESTER|GAKUNEN|WEEK|SUBJECT_CODE|CLASS_CODE|DEPARTMENT|HOUR|
    int cnt = commonMethods.deleteSubjectClassFromJikanwari(paramValues);
    if (cnt != 0) {
      jikanwariView.refreshTable();
    } else {
      commonInfo.showMessage("指定された科目は時間割表に配置されていません。");	  
      return;
    }
  }
  
  public void setRoomToGraduateGakunenWeekJikanwari(String param) { 
    String paramValues1 = getSelectedGraduateJikanwariSubjectClass();
    if (paramValues1 == null) return;

    String room = getSelectedRoom();
    if (room == null) return;

    String paramValues = paramValues1 + room;
    // paramValues = SCHOOL_YEAR|FACULTY|SEMESTER|GAKUNEN|WEEK|SUBJECT_CODE|CLASS_CODE|DEPARTMENT|HOUR|ROOM|
    int cnt = commonMethods.setRoomToSubjectClassOfJikanwari(paramValues);
    if (cnt != 0) {
      jikanwariView.refreshTable();
    } else {
      commonInfo.showMessage("講義室が設定できません。");	  
      return;
    }
  }
  
  public void addClassToGraduateShuchuList(String param) {
    String paramValues1 = getSelectedShuchuDept();
    if (paramValues1 == null) return;
    // paramValues1 = SCHOOL_YEAR|FACULTY|SEMESTER|DEPARTMENT|
    String paramValues2 = getSelectedGakunen();
    if (paramValues2 == null) return;
    // paramValues2 = GAKUNEN|
    
    String paramValues3 = getSelectedSubjectClassInfo();
    if (paramValues3 == null) return;
    // paramValues3 = SUBJECT_CODE|CLASS_CODE|
    
    String paramValues = paramValues1 + paramValues2 + paramValues3;
    // paramValues = SCHOOL_YEAR|FACULTY|SEMESTER|DEPARTMENT|GAKUNEN|SUBJECT_CODE|CLASS_CODE|
    int cnt = commonMethods.addSubjectClassToGraduateShuchuList(paramValues);
    if (cnt != 0) {
      tableView.refreshTable();
    } else {
      commonInfo.showMessage("記入しようとるす科目は当該学科の履修課程表に存在しません。");	  
      return;
    }
  }
    

  public void printTable(String param) {
    try {
      if (tablePrinter == null) {
	tablePrinter = new TablePrinter(tableView);
      }
      tablePrinter.printTable();
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  public void printAttendTable(String param) {
    try {
      if (tablePrinter == null) {
	tablePrinter = new TablePrinter(tableView);
      }
      tablePrinter.printAttendTable();
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  public void makeLatexSource(String param) { 
    try {
      if (tablePrinter == null) {
	tablePrinter = new TablePrinter(tableView);
      }
      tablePrinter.makeLatexSource();
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  public void makeTextFile(String param) {
    if (tablePrinter == null) {
      tablePrinter = new TablePrinter(tableView);
    }
    tablePrinter.makeTextFile();
  }


  public void printGakusekiTable(String param) {  
    try {
      if (tablePrinter == null) {
	tablePrinter = new TablePrinter(gakusekiView.tableView);
      }
      tablePrinter.printTable();
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  public void makeGakusekiTextFile(String param) {
    if (tablePrinter == null) {
      tablePrinter = new TablePrinter(gakusekiView.tableView);
    }
    tablePrinter.makeTextFile();
  }

  
  public void makeNinteiFormTableLatex(String param) {
    if (tablePrinter == null) {
      tablePrinter = new TablePrinter(tableView);
    }
    tablePrinter.makeNinteiFormTableLatex();
  }

  public void makeCurriculum1TableLatex(String param) {
    if (tablePrinter == null) {
      tablePrinter = new TablePrinter(tableView);
    }
    tablePrinter.makeCurriculum1TableLatex();
  }


  public void makeCurriculum3TableLatex(String param) {
    if (tablePrinter == null) {
      tablePrinter = new TablePrinter(tableView);
    }
    tablePrinter.makeCurriculum3TableLatex();
  }

  public void makeNinteiTableLatex(String param) {
    if (tablePrinter == null) {
      tablePrinter = new TablePrinter(tableView);
    }
    tablePrinter.makeNinteiTableLatex();
  }

  public void printNinteiTable(String param) {
    if (tablePrinter == null) {
      tablePrinter = new TablePrinter(tableView);
    }
    tablePrinter.printNinteiTable();
  }

  
  public void printStudentInitialPassword(String param) {  
    int row = tableView.getSelectedRow();
    if (row < 0) return;
    if (tablePrinter == null) {
      tablePrinter = new TablePrinter(tableView);
    }

    String studentCode = tableView.getCodeAt(row, "STUDENT_CODE");
    String studentName = tableView.getNameAt(row, "STUDENT_CODE");
    String initPasswd  = tableView.getCodeAt(row, "INIT_PASSWD");
    String studentStatus  = tableView.getNameAt(row, "STUDENT_STATUS");
    tablePrinter.printStudentInitialPassword(studentCode, studentName, 
					     initPasswd, studentStatus);
  }

  public void printStudentInitPasswdList(String param) {
    if (tablePrinter == null) {
      tablePrinter = new TablePrinter(tableView);
    }
    tablePrinter.printStudentInitPasswdList();
  }

  public void initAndPrintStaffInitialPassword(String param) {
    int row = tableView.getSelectedRow();
    if (row < 0) return;

    if (tablePrinter == null) {
      tablePrinter = new TablePrinter(tableView);
    }

    String staffCode = tableView.getCodeAt(row, "STAFF_CODE");
    String staffName = tableView.getNameAt(row, "STAFF_CODE");
    String userID = tableView.getCodeAt(row, "USER_ID");

    if (userID == null) return;
    if (userID.trim().equals("")) return;
    
    String initPasswd = commonMethods.initializeStaffPassword(userID);

    if (initPasswd != null) {
      tableView.refreshTable();
      tablePrinter.printStaffInitialPassword(staffCode, staffName, 
					     userID, initPasswd);
    }
  }

  public void registrateAndPrintStaffInitialPassword(String param) {
    int row = tableView.getSelectedRow();
    if (row < 0) return;
    if (tablePrinter == null) {
      tablePrinter = new TablePrinter(tableView);
    }

    String staffCode = tableView.getCodeAt(row, "STAFF_CODE");
    String staffName = tableView.getNameAt(row, "STAFF_CODE"); 

    ArrayList<Object> list = new ArrayList<Object>();
    list.add(" ユーザ登録すべき職員の「ユーザID」と「アクセス権限」を ");
    list.add(" 設定して下さい。");

    JTextField editor = new JTextField( staffCode + " (" + staffName + ")" );
    editor.setEditable(false); 
    editor.setFont(new Font("DialogInput", Font.PLAIN, 12));  
    editor.setBackground(Color.yellow);
    list.add(editor);

    JTextField editor2 = new JTextField(100);
    editor2.setFont(new Font("DialogInput", Font.PLAIN, 12));
    editor2.setText("  ");
    editor2.setEditable(true);
    editor2.setBorder(new TitledBorder(" ユーザID "));
    list.add(editor2);
    
    TabbedPaneBase tab = new TabbedPaneBase("QualSelector", "root", commonInfo, null, null);
    tab.setPreferredSize(new Dimension(500, 210));
    tab.pageOpened();
    list.add(tab);

    int ans =  JOptionPane.showOptionDialog(commonInfo.getFrame(),
					    list.toArray(), 
					    "追加の確認", 
					    JOptionPane.OK_CANCEL_OPTION,
					    JOptionPane.QUESTION_MESSAGE,
					    null,
					    null,
					    null);
    if (ans == JOptionPane.OK_OPTION) {
      String staffID = editor2.getText().trim();
      String qual = tab.getValueFromColumnCodeMap("QUALIFICATION");

      if (staffID.equals("")) return;
      if (qual == null) return;
      
      String qualification = qual.trim();

      String param2 = staffID + "|" + staffCode + "|" + qualification;

      String initPasswd = commonMethods.registrateStaffPassword(param2);
      if (initPasswd != null) {
	tableView.refreshTable();
	tablePrinter.printStaffInitialPassword(staffCode, staffName, staffID, initPasswd);
      }
    } 
  }

  public void showSubjectSeisekiGraph(String param) {  
    if (tableView.getSelectedRowCount() == 0) return;

    String SUBJECT_CODE = parentTabbedPane.getValueFromColumnCodeMap("SUBJECT_CODE");
    String SUBJECT_NAME = parentTabbedPane.getDisplayFromColumnCodeMap("SUBJECT_CODE");
    String TEACHER_CODE = parentTabbedPane.getValueFromColumnCodeMap("TEACHER_CODE");
    String TEACHER_NAME = parentTabbedPane.getDisplayFromColumnCodeMap("TEACHER_CODE");
    String SCHOOL_YEAR  = parentTabbedPane.getValueFromColumnCodeMap("SCHOOL_YEAR");
    String MARKS        = parentTabbedPane.getValueFromColumnCodeMap("MARKS");
    int marks = Integer.parseInt(MARKS);

    commonMethods.showSubjectSeisekiGraph(marks, 
				       SUBJECT_CODE, TEACHER_CODE, SCHOOL_YEAR,
				       SUBJECT_NAME, TEACHER_NAME);    
  }
  
  public void showYokenGPAGraph(String param) { 
    if (tableView.getSelectedRowCount() == 0) return;

    String YOKEN_GPA   = parentTabbedPane.getValueFromColumnCodeMap("YOKEN_GPA");
    String YOKEN_CODE  = parentTabbedPane.getValueFromColumnCodeMap("YOKEN_CODE");
    String FACULTY     = parentTabbedPane.getValueFromColumnCodeMap("FACULTY");
    String DEPARTMENT  = parentTabbedPane.getValueFromColumnCodeMap("DEPARTMENT");
    String COURSE      = parentTabbedPane.getValueFromColumnCodeMap("COURSE");
    String GAKUNEN     = parentTabbedPane.getValueFromColumnCodeMap("GAKUNEN");
    if (FACULTY == null) {
      FACULTY     = commonInfo.getValueFromCommonCodeMap("MY_FACULTY");
      DEPARTMENT  = commonInfo.getValueFromCommonCodeMap("MY_DEPARTMENT");
      COURSE      = commonInfo.getValueFromCommonCodeMap("MY_COURSE");
      GAKUNEN     = commonInfo.getValueFromCommonCodeMap("MY_GAKUNEN");
    }
    String YOKEN_CODE_NAME  = parentTabbedPane.getDisplayFromColumnCodeMap("YOKEN_CODE");
    if (YOKEN_GPA == null) return;
    if (YOKEN_GPA.equals("")) return;
    if (YOKEN_GPA.equals(" ")) return;
    if ((YOKEN_CODE.equals("440000")) || (YOKEN_CODE.equals("446700")) || 
	(YOKEN_CODE.equals("440455")) || (YOKEN_CODE.equals("635001")) || 
	(YOKEN_CODE.equals("636001")) || (YOKEN_CODE.equals("637001")) || 
	(YOKEN_CODE.equals("840000"))) {
      double gpa = Double.parseDouble(YOKEN_GPA);
      commonMethods.showYokenGPAGraph(gpa, 
				   YOKEN_CODE, FACULTY, DEPARTMENT, COURSE, GAKUNEN, 
				   YOKEN_CODE_NAME);
    }
  }
   
  public void showGokakuGPAGraph(String param) {  
    int index = -1;
    for (int i = 0; i < tableView.getColumnCount(); i++) {
      if (tableView.getColumnName(i).equals("GPA評価値（１）")) {
	index = i;
      }
    }
    if (index < 0) return;
    
    Double GPA = (Double)((CellObject) tableView.getValueAt(0, index)).getCode();
    double gpa = GPA.doubleValue();
    
    String FACULTY     = parentTabbedPane.getValueFromColumnCodeMap("FACULTY");
    String DEPARTMENT  = parentTabbedPane.getValueFromColumnCodeMap("DEPARTMENT");
    String COURSE      = parentTabbedPane.getValueFromColumnCodeMap("COURSE");
    String GAKUNEN     = parentTabbedPane.getValueFromColumnCodeMap("GAKUNEN");
    if (FACULTY == null) {
      FACULTY     = commonInfo.getValueFromCommonCodeMap("MY_FACULTY");
      DEPARTMENT  = commonInfo.getValueFromCommonCodeMap("MY_DEPARTMENT");
      COURSE      = commonInfo.getValueFromCommonCodeMap("MY_COURSE");
      GAKUNEN     = commonInfo.getValueFromCommonCodeMap("MY_GAKUNEN");
    }
    
    commonMethods.showGokakuGPAGraph(gpa, FACULTY, DEPARTMENT, COURSE, GAKUNEN );
  }
  
  public void showTotalGPAGraph(String param) { 
    int index = -1;
    for (int i = 0; i < tableView.getColumnCount(); i++) {
      if (tableView.getColumnName(i).equals("GPA評価値（２）")) {
	index = i;
      }
    }
    if (index < 0) return;
    
    Double GPA = (Double)((CellObject) tableView.getValueAt(0, index)).getCode();
    double gpa = GPA.doubleValue();
    
    String FACULTY     = parentTabbedPane.getValueFromColumnCodeMap("FACULTY");
    String DEPARTMENT  = parentTabbedPane.getValueFromColumnCodeMap("DEPARTMENT");
    String COURSE      = parentTabbedPane.getValueFromColumnCodeMap("COURSE");
    String GAKUNEN     = parentTabbedPane.getValueFromColumnCodeMap("GAKUNEN");
    if (FACULTY == null) {
      FACULTY     = commonInfo.getValueFromCommonCodeMap("MY_FACULTY");
      DEPARTMENT  = commonInfo.getValueFromCommonCodeMap("MY_DEPARTMENT");
      COURSE      = commonInfo.getValueFromCommonCodeMap("MY_COURSE");
      GAKUNEN     = commonInfo.getValueFromCommonCodeMap("MY_GAKUNEN");
    }
    
    commonMethods.showTotalGPAGraph(gpa, FACULTY, DEPARTMENT, COURSE, GAKUNEN );
  }
}
