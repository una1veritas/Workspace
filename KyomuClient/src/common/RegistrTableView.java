package common;

import clients.*;
//import java.util.*;
//import javax.swing.*;
//import javax.swing.table.*;
//import java.awt.*;
//import java.awt.event.*;
//import java.io.*;
//import java.io.File;

public class RegistrTableView extends TableViewBase {
  public RegistrInfo registrInfo;

  public RegistrTableView(String tableViewType,
			  String serviceName,
			  String nodePath,
			  String panelID,
			  CommonInfo commonInfo, 
			  TabbedPaneBase tabbedPane,
			  DataPanelBase dataPanel) {
    super(tableViewType, 
	  serviceName, nodePath, panelID,
	  commonInfo, tabbedPane, dataPanel,
	  "registr");

    this.registrInfo = commonInfo.commonInfoMethods.registrInfo;
    setSwitchCode();
    setQueryParams();
    tableModel = new RegistrTableModel(commonInfo, serviceName, panelID,
				       columnTitleList, columnCodeList,
				       columnDisplayList);
    setTableView();
  }

  public void pageOpened() {
    registrInfo.setRegistrInfo();
    setQueryParamValuesAndDisplays();
    if (queryParamValues != null) {
      tableModel.setTableData(serviceName, panelID, switchCode, queryParamValues); 
    }
    setColumnDisplay(); 
    commonInfo.timerRestart(); 
  }
  
  public void refreshRegistrInfo() {
    registrInfo.setRegistrInfo();
    refreshTableFlag = true;    
  }

  public void refreshRegistrView() {
    registrInfo.setRegistrInfo();
   
    setQueryParamValuesAndDisplays();    
    parentDataPanel.setTitleText(queryParamDisplays);
    if (queryParamValues != null) {
      tableModel.setTableData(serviceName, panelID, switchCode, queryParamValues); 
    }
    setColumnDisplay(); 	
    commonInfo.timerRestart();
  }

  public void cancelSelectedRegistr() {
    if (registrInfo.registrAllowedPeriod == false) {      
      commonInfo.showMessage("現在は「履修(修正)申告」できる期間ではありません。");
      return;
    }
    String schoolYear  = parentTabbedPane.getValueFromColumnCodeMap("SCHOOL_YEAR");
    String subjectCode = parentTabbedPane.getValueFromColumnCodeMap("SUBJECT_CODE");
    String subjectName = parentTabbedPane.getDisplayFromColumnCodeMap("SUBJECT_CODE");
    String teacherCode = parentTabbedPane.getValueFromColumnCodeMap("TEACHER_CODE");
    String teacherName = parentTabbedPane.getDisplayFromColumnCodeMap("TEACHER_CODE");
    String classCode   = parentTabbedPane.getValueFromColumnCodeMap("CLASS_CODE");
    
    int sig = registrInfo.getSubjectStatus(subjectCode, classCode);
    if (sig != 4) {   
      commonInfo.showMessage("「" +subjectName + "」は「仮履修登録」されていません。");
      return;
    }

    int ret = registrInfo.cancelRegistrSubject(schoolYear, subjectCode, classCode, 
					       subjectName, teacherName);
    if (ret > 0) {
      refreshRegistrView();
    }
  }

  public void registrSelectedSubject() {
    if (registrInfo.registrAllowedPeriod == false) {      
      commonInfo.showMessage("現在は「履修(修正)申告」できる期間ではありません。");
      return;
    }
    String schoolYear  = parentTabbedPane.getValueFromColumnCodeMap("SCHOOL_YEAR");
    String subjectCode = parentTabbedPane.getValueFromColumnCodeMap("SUBJECT_CODE");
    String subjectName = parentTabbedPane.getDisplayFromColumnCodeMap("SUBJECT_CODE");
    String teacherCode = parentTabbedPane.getValueFromColumnCodeMap("TEACHER_CODE");
    String teacherName = parentTabbedPane.getDisplayFromColumnCodeMap("TEACHER_CODE");
    String classCode   = parentTabbedPane.getValueFromColumnCodeMap("CLASS_CODE");
    String unit        = parentTabbedPane.getValueFromColumnCodeMap("UNIT");

    int sig = registrInfo.getSubjectStatus(subjectCode, classCode);
    if (sig == 1) {
      commonInfo.showMessage("「" +subjectName + "」は「修得済み科目」です。");
      return;
    } else if (sig == 2) {
      commonInfo.showMessage("「" +subjectName + "」は「単位認定」された科目です。");
      return;
    } else if (sig == 3) {
      commonInfo.showMessage("「" +subjectName + "」は既に「履修登録」されています。");
      return;
    } else if (sig == 4) {
      commonInfo.showMessage("「" +subjectName + "」は既に「仮履修登録」されています。");
      return;
    } else if (sig == 5) {
      commonInfo.showMessage("「" +subjectName + "」は既に「仮履修登録」されています。");
      return;
    }

    boolean ans = registrInfo.checkRegistrAllowed(subjectCode, classCode, unit);
    if (ans) {
      int ret = registrInfo.registrSubject(schoolYear, subjectCode, classCode, unit,
					   subjectName, teacherName);
      if (ret > 0) {
	refreshRegistrView();
      }
    }
  }
}
