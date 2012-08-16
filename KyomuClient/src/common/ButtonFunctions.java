package common;
import clients.*;
import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.border.*;
//import java.lang.reflect.*;

public class ButtonFunctions {
  private String serviceName;
  private String nodePath;
  private String panelID;
  private CommonInfo commonInfo;
  private TabbedPaneBase parentTabbedPane;
  private DataPanelMethods dataPanelMethods;

  private TableViewBase tableView;
  private String lineSeparator = System.getProperty("line.separator");
  
  private ArrayList<ArrayList<JButton>> buttonList = new ArrayList<ArrayList<JButton>>();
  private Map<String, ButtonInfo> buttonInfoMap = new HashMap<String, ButtonInfo>();
    
  private ActionListener buttonListener = new ActionHandler();

  public ButtonFunctions(String serviceName,
			 String nodePath,
			 String panelID,
			 CommonInfo commonInfo, 
			 TabbedPaneBase parentTabbedPane,
			 DataPanelMethods dataPanelMethods) {
    this.serviceName = serviceName;
    this.nodePath = nodePath;
    this.panelID = panelID;
    this.commonInfo = commonInfo;
    this.parentTabbedPane = parentTabbedPane;
    this.dataPanelMethods = dataPanelMethods;

    for (int i = 0; i < 3; i++) {
      ArrayList<JButton> list = new ArrayList<JButton>();
      buttonList.add(list);
    }
    makeSimpleButtonInfo();
    makeUpdateButtonInfo();
  }

  public void setTargetTableView(TableViewBase tableView) {
    this.tableView = tableView;
  }

  public ArrayList<JButton> getButtonList(int i) {
    return buttonList.get(i);
  }

  public ButtonInfo getButtonInfo(String buttonTitle) {
    return buttonInfoMap.get(buttonTitle);
  }

  private void makeSimpleButtonInfo() {
    String buttonTitle = null;
    int buttonRow = 0;
    int buttonCol = 0;
    Color buttonFgColor = null;
    Color buttonBgColor = null;
    String qualificationMethod = null;
    String qualificationParam = null;
    String methodToInvoke = null;
    String methodToInvokeParam = null;
    
    String buttonsInfos = commonInfo.getSimpleButtonStruct(serviceName, panelID);
    if (buttonsInfos != null) {
      String[] buttonsInfoArray = buttonsInfos.split("\\$");
      for (String buttonsInfo : buttonsInfoArray) {
	String[] tokens =  buttonsInfo.split("\\|");
	if (tokens.length == 9) { 
	  for (int i = 0; i < 9; i++) {
	    String token = tokens[i].trim();
	    if (token.equals("")) token = null;
	    switch (i) {
	    case 0:
	      buttonTitle = token; break;
	    case 1:
	      buttonRow = Integer.parseInt(token); break;
	    case 2:
	      buttonCol = Integer.parseInt(token); break;
	    case 3:
	      buttonFgColor = commonInfo.getSimpleButtonFgColor(token); break;
	    case 4:
	      buttonBgColor = commonInfo.getSimpleButtonBgColor(token); break;
	    case 5:
	      qualificationMethod = token; break;
	    case 6:
	      qualificationParam = token; break;
	    case 7:
	      methodToInvoke = token; break;
	    case 8:
	      methodToInvokeParam = token; break;
	    }
	  }
	  boolean enabled = checkQualification(qualificationMethod, qualificationParam);

	  if (enabled) {
	    JButton button = new JButton(buttonTitle);
	    button.setForeground(buttonFgColor);
	    button.setBackground(buttonBgColor);
	    button.setFont(new Font("DialogInput", Font.PLAIN, 12));
	    button.setBorder(new CompoundBorder(BorderFactory.createRaisedBevelBorder(), 
						new EmptyBorder(2,4,2,4)));
	    button.addActionListener(buttonListener);
	    (buttonList.get(buttonRow)).add(button);
	    ButtonInfo buttonInfo = new ButtonInfo("SIMPLE", methodToInvoke, methodToInvokeParam);
	    buttonInfoMap.put(buttonTitle, buttonInfo);
	  }
	} else {
	  commonInfo.showMessageLong("SimpleButtonStruct: $ format error $ " + buttonsInfo);
	}
      }
    }
  }

  private void makeUpdateButtonInfo() {
    String buttonTitle = null;
    int buttonRow = 0;
    int buttonCol = 0;
    Color buttonFgColor = null;
    Color buttonBgColor = null;
    String qualificationMethod = null;
    String qualificationParam = null;
    String commandType = null;
    String commandCode = null;
    
    String buttonsInfos = commonInfo.getUpdateButtonStruct(serviceName, panelID);
    if (buttonsInfos != null) {
      String[] buttonsInfoArray = buttonsInfos.split("\\$");
      for (String buttonsInfo : buttonsInfoArray) {
	String[] tokens =  buttonsInfo.split("\\|");
	if (tokens.length == 9) { 
	  for (int i = 0; i < 9; i++) {
	    String token = tokens[i].trim();
	    if (token.equals("")) token = null;
	    switch (i) {
	    case 0:
	      buttonTitle = token; break;
	    case 1:
	      buttonRow = Integer.parseInt(token); break;
	    case 2:
	      buttonCol = Integer.parseInt(token); break;
	    case 3:
	      buttonFgColor = commonInfo.getUpdateButtonFgColor(token); break;
	    case 4:
	      buttonBgColor = commonInfo.getUpdateButtonBgColor(token); break;
	    case 5:
	      qualificationMethod = token; break;
	    case 6:
	      qualificationParam = token; break;
	    case 7:
	      commandType = token; break;
	    case 8:
	      commandCode = token; break;
	    }
	  }
	  boolean enabled = checkQualification(qualificationMethod, qualificationParam);
	  if (enabled) {
	    JButton button = new JButton(buttonTitle);
	    button.setForeground(buttonFgColor);
	    button.setBackground(buttonBgColor);
	    button.setFont(new Font("DialogInput", Font.PLAIN, 12));
	    button.setBorder(new CompoundBorder(BorderFactory.createRaisedBevelBorder(), 
						new EmptyBorder(2,4,2,4)));
	    button.addActionListener(buttonListener);
	    (buttonList.get(buttonRow)).add(button);
	    ButtonInfo buttonInfo = new ButtonInfo(commandType, serviceName, commandCode);
	    buttonInfoMap.put(buttonTitle, buttonInfo);
	  }
	} else {
	  commonInfo.showMessageLong("UpdateButtonStruct: $ format error $ " + buttonsInfo);
	}
      }
    }
  }

  class ActionHandler implements ActionListener { 

    public void actionPerformed(ActionEvent e) { 
      String buttonTitle = e.getActionCommand();
      ButtonInfo buttonInfo =  getButtonInfo(buttonTitle);
      String commandType = buttonInfo.commandType;
      if (commandType.equals("SIMPLE")) {
	String methodName = buttonInfo.methodToInvoke;
	String methodParam = buttonInfo.methodParam;
	simpleButtonPressed(methodName, methodParam);
      } else {
	String serviceName = buttonInfo.serviceName;
	String commandCode = buttonInfo.commandCode;
	if (commandType.equals("DELETE")) {
	  deleteButtonPressed(serviceName, commandCode);
	} else if (commandType.equals("UPDATE")) {
	  updateButtonPressed(serviceName, commandCode);
	} else if (commandType.equals("INSERT")) {
	  insertButtonPressed(serviceName, commandCode);
	} else if (commandType.equals("SPECIAL")) {
	  specialButtonPressed(serviceName, commandCode);
	}
      }
      commonInfo.timerRestart();
    }
  }

  //******** Delete Button Pressed ****************//

  private void deleteButtonPressed(String deleteServiceName, String deleteCode) {
    if (tableView.getSelectedRowCount() == 0) return;

    String deleteParams = commonInfo.getServerDeleteParams(deleteServiceName, deleteCode);
    String[] tokens =  deleteParams.split("\\|");
    String whereParams = tokens[0].trim();
    
    int count = 0;
    int[] rows = tableView.getSelectedRows();
    for (int row : rows) {
      count += confirmDelete(deleteServiceName, deleteCode, whereParams, row);
    }
    if (count > 0) {
      tableView.refreshTable();
    }
  }

  private int confirmDelete(String deleteServiceName, String deleteCode, 
			   String whereParams, int row) {
    ArrayList<Object> list = new ArrayList<Object>();
    list.add("次のデータを削除します。 UNDO 機能はないので了解する前に十分確認して下さい。");

    StringBuilder whereParamValues = new StringBuilder();
    String text;

    String[] columnCodes = whereParams.split("\\:");
    for (String columnCode : columnCodes) {
      String colValue = tableView.getCodeAt(row, columnCode);
      String colDisplay = tableView.getNameAt(row, columnCode);
      if (colValue == null) {
	if (parentTabbedPane.columnCodeMapContains(columnCode)) {
	  colValue = parentTabbedPane.getValueFromColumnCodeMap(columnCode);
	  colDisplay = parentTabbedPane.getDisplayFromColumnCodeMap(columnCode);
	} else if (commonInfo.commonCodeMapContains(columnCode)) {
	  colValue = commonInfo.getValueFromCommonCodeMap(columnCode);
	  colDisplay = commonInfo.getDisplayFromCommonCodeMap(columnCode);
	}
	if (colValue == null) {
	  if (!commonInfo.structControlDebugMode) {
	    commonInfo.showMessageLong("削除データを特定するパラメータが設定されていません。$ " + columnCode);
	    return 0;
	  } else {  //  structControl のデバッグ時の設定
	    String str = "debugMode: パラメータ " + columnCode + " の値を指定して下さい。";
	    String value = commonInfo.getDialogInput(str);
	    if (value == null) {
	      value = " ";
	    } else {
	      String val = value.trim();
	      if (val.equals("")) {
		value = " ";
	      } else {
		value = val;
	      }
	    }
	    colValue = value;
	    colDisplay = colValue;
	  }
	}
      }
      if (colValue.equals("")) colValue = " ";
      if (colValue.equals(colDisplay)) {
	text = " " + columnCode + " = " + colValue;
      } else {
	text = " " + columnCode + " = " + colValue + " ( " + colDisplay + " )";
      }
      whereParamValues.append(colValue).append("|");
      JTextField editor = new JTextField(text);
      editor.setEditable(false); 
      editor.setFont(new Font("DialogInput", Font.PLAIN, 12));  
      editor.setBackground(Color.yellow);
      list.add(editor);
    }

    int ans =  JOptionPane.showOptionDialog(commonInfo.getFrame(), 
					    list.toArray(), 
					    "削除の確認", 
					    JOptionPane.OK_CANCEL_OPTION,
					    JOptionPane.QUESTION_MESSAGE,
					    null,
					    null,
					    null);
    if (ans == JOptionPane.OK_OPTION) {
      int ret = commonInfo.getDeleteResult(deleteServiceName, deleteCode, whereParamValues.toString());
      if (ret == 0) {
	commonInfo.showMessageLong("データの削除に失敗しました $ " + whereParams + " = " + whereParamValues.toString());
      } 
      return ret;
    } else {
      return 0;
    }
  }

  //******** Update Button Pressed ****************//

  private void updateButtonPressed(String updateServiceName, String updateCode) {
    if (tableView.getSelectedRowCount() == 0) return;

    String updateParams = commonInfo.getServerUpdateParams(updateServiceName, updateCode);
    String[] tokens = updateParams.split("\\|");
    String setParams = tokens[0];
    String setEditors = tokens[1];
    String whereParams = tokens[2];

    int count = 0;
    int[] rows = tableView.getSelectedRows();
    for (int row : rows) {
      count += confirmUpdate(updateServiceName, updateCode, setParams, setEditors, whereParams, row);
    }
    if (count > 0) {
      tableView.refreshTable();
    }
  }

  private int confirmUpdate(String updateServiceName, String updateCode, 
			    String setParams, String setEditors,
			    String whereParams, int row) {
    ArrayList<Object> list = new ArrayList<Object>();
    list.add("更新データを設定して下さい。UNDO 機能はないので了解する前に十分確認して下さい。");

    StringBuilder whereParamValues = new StringBuilder();
    String text;
    String[] columnCodes = whereParams.split("\\:");
    for (String columnCode : columnCodes) {
      String colValue = tableView.getCodeAt(row, columnCode);
      String colDisplay = tableView.getNameAt(row, columnCode);
      if (colValue == null) {
	if (parentTabbedPane.columnCodeMapContains(columnCode)) {
	  colValue = parentTabbedPane.getValueFromColumnCodeMap(columnCode);
	  colDisplay = parentTabbedPane.getDisplayFromColumnCodeMap(columnCode);
	} else if (commonInfo.commonCodeMapContains(columnCode)) {
	  colValue = commonInfo.getValueFromCommonCodeMap(columnCode);
	  colDisplay = commonInfo.getDisplayFromCommonCodeMap(columnCode);
	}
	if (colValue == null) {
	  if (!commonInfo.structControlDebugMode) {
	    commonInfo.showMessageLong("更新データを特定するパラメータが設定されていません。$ " + columnCode);
	    return 0;
	  } else {  //  structControl のデバッグ時の設定
	    String str = "debugMode: パラメータ " + columnCode + " の値を指定して下さい。";
	    String value = commonInfo.getDialogInput(str);
	    if (value == null) {
	      value = " ";
	    } else {
	      String val = value.trim();
	      if (val.equals("")) {
		value = " ";
	      } else {
		value = val;
	      }
	    }
	    colValue = value;
	    colDisplay = colValue;
	  }
	}
      }
      if (colValue.equals("")) colValue = " ";
      if (colValue.equals(colDisplay)) {
	text = " " + columnCode + " = " + colValue;
      } else {
	text = " " + columnCode + " = " + colValue + " ( " + colDisplay + " )";
      }
      whereParamValues.append(colValue).append("|");
      JTextField editor = new JTextField(text);
      editor.setEditable(false); 
      editor.setFont(new Font("DialogInput", Font.PLAIN, 12));  
      editor.setBackground(Color.yellow);
      list.add(editor);
    }

    String[] params  = setParams.split("\\:");
    String[] editors = setEditors.split("\\:");
    int size = params.length;
    for (int i = 0; i < size; i++) {
      String columnCode = params[i];
      String selector = editors[i];
            
      if ((selector.equals(" ")) || (selector.equals("null"))) {
	String columnTitle = tableView.getColumnTitle(columnCode);
	String colValue = tableView.getCodeAt(row, columnCode);
	String colDisplay = tableView.getNameAt(row, columnCode);
	JComponent comp = commonInfo.getSelector(null, columnCode, columnTitle,
						 colValue, colDisplay, size);
	list.add(comp);
      } else {
	JComponent comp = commonInfo.getSelector(selector, null, null, null, null, size);
	list.add(comp);
      }
    }

    int ans =  JOptionPane.showOptionDialog(commonInfo.getFrame(),
					    list.toArray(), 
					    "更新の確認", 
					    JOptionPane.OK_CANCEL_OPTION,
					    JOptionPane.QUESTION_MESSAGE,
					    null,
					    null,
					    null);
    if (ans == JOptionPane.OK_OPTION) {
      StringBuilder setParamValues = new StringBuilder();

      String[] params2  = setParams.split("\\:");
      String[] editors2 = setEditors.split("\\:");
      int size2 = params2.length;
      for (int i = 0; i < size2; i++) {
	String columnCode = params2[i];
	String selector = editors2[i];
	if ((selector.equals(" ")) || (selector.equals("null"))) {
	  selector = null;
	}	
	String columnValue = commonInfo.getSelectorValue(selector, columnCode);
	setParamValues.append(columnValue);
      }
      int ret = commonInfo.getUpdateResult(updateServiceName, updateCode, 
					   setParamValues.toString()+whereParamValues.toString());
      if (ret != 1) {
	commonInfo.showMessage("データの更新に失敗しました");
      } 
      return ret;
    } else {
      return 0;
    }
  }  
  
  //******** Insert Button Pressed ****************//

  private void insertButtonPressed(String insertServiceName, String insertCode) {
    String insertParamData = commonInfo.getServerInsertParams(insertServiceName, insertCode);
    String[] tokens = insertParamData.split("\\|");
    String insertParams = tokens[0];
    String insertEditors = tokens[1];
    String whereParams = tokens[2];

    int count = confirmInsert(insertServiceName, insertCode, 
			      insertParams, insertEditors, 
			      whereParams);
    if (count > 0) {
      tableView.refreshTable();
    }
  }

  public int confirmInsert(String insertServiceName, String insertCode, 
			   String insertParams, String insertEditors,
			   String whereParams) {
    ArrayList<Object> list = new ArrayList<Object>();
    list.add("追加するデータを設定して下さい。UNDO 機能はないので了解する前に十分確認して下さい。");

    StringBuilder whereParamValues = new StringBuilder();
    String text;
    String colValue = null;
    String colDisplay = null;
    if (!whereParams.equals(" ")) {
      String[] columnCodes = whereParams.split("\\:");
      for (String columnCode : columnCodes) {
	if (parentTabbedPane.columnCodeMapContains(columnCode)) {
	  colValue = parentTabbedPane.getValueFromColumnCodeMap(columnCode);
	  colDisplay = parentTabbedPane.getDisplayFromColumnCodeMap(columnCode);
	} else if (commonInfo.commonCodeMapContains(columnCode)) {
	  colValue = commonInfo.getValueFromCommonCodeMap(columnCode);
	  colDisplay = commonInfo.getDisplayFromCommonCodeMap(columnCode);
	}
	if (colValue == null) {
	  if (!commonInfo.structControlDebugMode) {
	    commonInfo.showMessageLong("追加データを特定するパラメータが設定されていません。$ " + columnCode);
	    return 0;
	  } else {  //  structControl のデバッグ時の設定
	    String str = "debugMode: パラメータ " + columnCode + " の値を指定して下さい。";
	    String value = commonInfo.getDialogInput(str);
	    if (value == null) {
	      value = " ";
	    } else {
	      String val = value.trim();
	      if (val.equals("")) {
		value = " ";
	      } else {
		value = val;
	      }
	    }
	    colValue = value;
	    colDisplay = colValue;
	  }
	}
	if (colValue.equals("")) colValue = " ";
	if (colValue.equals(colDisplay)) {
	  text = " " + columnCode + " = " + colValue;
	} else {
	  text = " " + columnCode + " = " + colValue + " ( " + colDisplay + " )";
	}
	whereParamValues.append(colValue).append("|");
	JTextField editor = new JTextField(text);
	editor.setEditable(false); 
	editor.setFont(new Font("DialogInput", Font.PLAIN, 12));  
	editor.setBackground(Color.yellow);
	list.add(editor);
      }
    }

    String[] params  = insertParams.split("\\:");
    String[] editors = insertEditors.split("\\:");
    int size = params.length;
    for (int i = 0; i < size; i++) {
      String columnCode = params[i];
      String selector = editors[i];            
      if ((selector.equals(" ")) || (selector.equals("null"))) {
	String columnTitle = tableView.getColumnTitle(columnCode);   
	if ((columnTitle == null) || (columnTitle.trim().equals(""))) {
	  columnTitle = columnCode;
	}
	JComponent comp = commonInfo.getSelector(null, columnCode, columnTitle, "", "", size);
	list.add(comp);
      } else {
	JComponent comp = commonInfo.getSelector(selector, null, null, null, null, size);
	list.add(comp);
      }
    }

    int ans =  JOptionPane.showOptionDialog(commonInfo.getFrame(),
					    list.toArray(), 
					    "追加の確認", 
					    JOptionPane.OK_CANCEL_OPTION,
					    JOptionPane.QUESTION_MESSAGE,
					    null,
					    null,
					    null);

    if (ans == JOptionPane.OK_OPTION) {
      StringBuilder insertParamValues = new StringBuilder();
      String[] params2  = insertParams.split("\\:");
      String[] editors2 = insertEditors.split("\\:");
      int size2 = params2.length;
      for (int i = 0; i < size2; i++) {
	String columnCode = params2[i];
	String selector = editors2[i];
	if ((selector.equals(" ")) || (selector.equals("null"))) {
	  selector = null;
	}	
	String columnValue = commonInfo.getSelectorValue(selector, columnCode);
	insertParamValues.append(columnValue);
      }
      
      int ret = commonInfo.getInsertResult(insertServiceName, insertCode, 
					   insertParamValues.toString()+whereParamValues.toString());
      if (ret != 1) {
	commonInfo.showMessage("データの追加に失敗しました");
      } 
      return ret;
    } else {
      return 0;
    }
  }  

  //******** Special Button Pressed ****************//

  public void specialButtonPressed(String specialServiceName, String specialCode) {
    String specialParams = commonInfo.getServerSpecialParams(specialServiceName, specialCode);
    String[] tokens =  specialParams.split("\\|");
    String specialParam = tokens[0].trim();
    
    int count = confirmSpecial(specialServiceName, specialCode, specialParam);
    if (count > 0) {
      tableView.refreshTable();
    }
  }

  public int confirmSpecial(String specialServiceName, String specialCode, String specialParam) {
    StringBuilder specialParamValues = new StringBuilder();
    String text;
    String colValue = null;
    String colDisplay = null;

    String[] columnCodes = specialParam.split("\\:");
    for (String columnCode : columnCodes) {
      if (parentTabbedPane.columnCodeMapContains(columnCode)) {
	colValue = parentTabbedPane.getValueFromColumnCodeMap(columnCode);
	colDisplay = parentTabbedPane.getDisplayFromColumnCodeMap(columnCode);
      } else if (commonInfo.commonCodeMapContains(columnCode)) {
	colValue = commonInfo.getValueFromCommonCodeMap(columnCode);
	colDisplay = commonInfo.getDisplayFromCommonCodeMap(columnCode);
      }
      if (colValue == null) {
	if (!commonInfo.structControlDebugMode) {
	  commonInfo.showMessageLong("データを特定するパラメータが設定されていません。$ " + columnCode);
	  return 0;
	} else {  //  structControl のデバッグ時の設定
	  String str = "debugMode: パラメータ " + columnCode + " の値を指定して下さい。";
	  String value = commonInfo.getDialogInput(str);
	  if (value == null) {
	    value = " ";
	  } else {
	    String val = value.trim();
	    if (val.equals("")) {
	      value = " ";
	    } else {
	      value = val;
	    }
	  }
	  colValue = value;
	  colDisplay = colValue;
	}
      }
      if (colValue.equals("")) colValue = " ";
      if (colValue.equals(colDisplay)) {
	text = " " + columnCode + " = " + colValue;
      } else {
	text = " " + columnCode + " = " + colValue + " ( " + colDisplay + " )";
      }
      specialParamValues.append(colValue).append("|");
    }

    int ret = commonInfo.getSpecialResult(specialServiceName, specialCode, 
					  specialParamValues.toString());
    if (ret != 1) {
      commonInfo.showMessage("データの処理に失敗しました");
    } 
    return ret;
  }  

  //******** Simple Button Pressed ****************//
  private void simpleButtonPressed(String methodName, String methodParam) {
    try {
      dataPanelMethods.getClass().getMethod(methodName, new Class[] { String.class }).invoke(dataPanelMethods, new Object[] { methodParam });
    } catch (Exception ex) {
      commonInfo.showMessageLong("simpleButton: $ " + methodName + " : " + methodParam);  
    }
  }


  //******** checkQualification ****************//
  private boolean checkQualification(String qualificationMethod, String methodParam) {
    if (qualificationMethod == null) return true;
    try {
      RetBoolean ret = new RetBoolean();
      getClass().getMethod(qualificationMethod, new Class[] { RetBoolean.class }).invoke(this, new Object[] { ret } );
      return ret.getValue();
    } catch (Exception ex) {
      commonInfo.showMessageLong("qualificationMethod in ButtonFunctions is not invoked $ " + qualificationMethod);  
      return false;
    }
  }

  public void checkAdministrator(RetBoolean ret) {
    if (commonInfo.commonInfoMethods.checkAdministrator()) {
      ret.accept();
    } else {
      ret.reject();
    }
  }  

  public void checkGakumuStaff(RetBoolean ret) {
    if (commonInfo.commonInfoMethods.checkGakumuStaff()) {
      ret.accept();
    } else {
      ret.reject();
    }
  } 

  public void checkSomuStaff(RetBoolean ret) {
    if (commonInfo.commonInfoMethods.checkSomuStaff()) {
      ret.accept();
    } else {
      ret.reject();
    }
  }

  public void checkHigherThanTeacher(RetBoolean ret) {
    if (commonInfo.commonInfoMethods.checkHigherThanTeacher()) {
      ret.accept();
    } else {
      ret.reject();
    }
  }

  public void checkHigherThanEduCommittee(RetBoolean ret) {
    if (commonInfo.commonInfoMethods.checkHigherThanEduCommittee()) {
      ret.accept();
    } else {
      ret.reject();
    }
  }

  public void checkQualifiedForNintei(RetBoolean ret) {
    String faculty = parentTabbedPane.getValueFromColumnCodeMap("FACULTY");
    String department = parentTabbedPane.getValueFromColumnCodeMap("DEPARTMENT");
    String curriculumYear = parentTabbedPane.getValueFromColumnCodeMap("CURRICULUM_YEAR");
    if (commonInfo.commonInfoMethods.checkQualifiedForNintei(faculty, department, curriculumYear)) {
      ret.accept();
    } else {
      ret.reject();
    }    
  }

  public void checkHigherThanEduAssistant(RetBoolean ret) {
    String faculty = parentTabbedPane.getValueFromColumnCodeMap("FACULTY");
    String department = parentTabbedPane.getValueFromColumnCodeMap("DEPARTMENT");
    if (commonInfo.commonInfoMethods.checkHigherThanEduAssistant(faculty, department)) {
      ret.accept();
    } else {
      ret.reject();
    }    
  }

  public void checkHigherThanTeacherHimself(RetBoolean ret) {
    String faculty = parentTabbedPane.getValueFromColumnCodeMap("FACULTY");
    String department = parentTabbedPane.getValueFromColumnCodeMap("DEPARTMENT");
    String teacherCode = parentTabbedPane.getValueFromColumnCodeMap("TEACHER_CODE");
    if (commonInfo.commonInfoMethods.checkHigherThanTeacherHimself(faculty, department, teacherCode)) {
      ret.accept();
    } else {
      ret.reject();
    }    
  }
}
