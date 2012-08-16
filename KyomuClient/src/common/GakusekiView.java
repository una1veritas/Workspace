package common;
import java.util.*;
import javax.swing.*;
import javax.swing.border.*;
import java.awt.*;
//import java.io.*;
//import javax.swing.text.*;
//import javax.swing.event.*;
import clients.*;

public class GakusekiView extends JPanel {
  protected String serviceName;
  protected String nodePath;
  protected String panelID;
  private CommonInfo commonInfo;
  private TabbedPaneBase tabbedPane;
  private DataPanelBase  dataPanel;

  public TableViewBase tableView;
  private GakusekiTableModel tableModel;
  private PhotoViewBase photoView;

  private String STUDENT_CODE;
  private String STUDENT_NAME;
  private String DEPARTMENT;

//  private String paramValuesOld = "abdakabdarah";

  public GakusekiView(String serviceName,
		      String nodePath, 
		      String panelID,
		      CommonInfo commonInfo, 
		      TabbedPaneBase tabbedPane,
		      DataPanelBase dataPanel) {
    this.serviceName = serviceName;
    this.nodePath = nodePath;
    this.panelID = panelID;
    this.commonInfo = commonInfo;
    this.tabbedPane = tabbedPane;
    this.dataPanel = dataPanel;

    setGakusekiView();
  }

  public void setGakusekiView() { 
    dataPanel.setTitleText(STUDENT_NAME);
    setLayout(new BorderLayout());

    photoView = new PhotoViewBase(serviceName, nodePath, panelID,
				  commonInfo, 
				  tabbedPane, dataPanel); 
    photoView.desktop.setMaximumSize(new Dimension(300, 700)); 
    photoView.setMaximumSize(new Dimension(300, 700)); 
    add(photoView, BorderLayout.CENTER);

    tableView = new TableViewBase("SimpleTableView",
				  serviceName, nodePath, panelID,
				  commonInfo, 
				  tabbedPane, dataPanel, "GAKUSEKI");

    tableModel = new GakusekiTableModel(commonInfo, serviceName, panelID,
					tableView.columnTitleList,
					tableView.columnCodeList,
					tableView.columnDisplayList);
    tableView.setModel(tableModel);
    JScrollPane jsp = new JScrollPane(tableView);
    jsp.setMinimumSize(new Dimension(600, 600));
    add(jsp, BorderLayout.WEST);
    dataPanel.tableView = tableView;
  } 

  public void pageOpened() {
    if (!commonInfo.STUDENT_CODE.equals("")) {
      STUDENT_CODE = commonInfo.STUDENT_CODE;
      STUDENT_NAME = commonInfo.STUDENT_NAME;
      DEPARTMENT = commonInfo.getGakumuCodeShorterName("DEPARTMENT", 
						       commonInfo.STUDENT_DEPARTMENT);
    } else {
      STUDENT_CODE = tabbedPane. getValueFromColumnCodeMap("STUDENT_CODE");
      STUDENT_NAME = tabbedPane.getDisplayFromColumnCodeMap("STUDENT_CODE");
      DEPARTMENT = tabbedPane.getDisplayFromColumnCodeMap("DEPARTMENT");
      if ((STUDENT_CODE == null) || (STUDENT_CODE.equals(""))) {	
	if (commonInfo.structControlDebugMode) {
	  String str = "debugMode: パラメータ STUDENT_CODE の値を指定して下さい。";
	  String value = commonInfo.getDialogInput(str);
	  if ((value != null) && (!(value.trim()).equals(""))) {
	    STUDENT_CODE = value.trim();
	    STUDENT_NAME = " ";
	    DEPARTMENT = " ";
	  }
	}
      }
    }
    
    photoView.clearPhotoView();
    photoView.addStudentPhoto(0, STUDENT_CODE, STUDENT_NAME, DEPARTMENT);
    
    tableModel.setParamValues(STUDENT_CODE);
    tableView.setColumnDisplay();
    commonInfo.timerRestart(); 
  }

  public void setGakusekiData(String columnCode) {
    if (columnCode.equals("HOME_TYPE")) {
      TabbedPaneBase tab = new TabbedPaneBase("HomeTypeSelector", "root", commonInfo, null, null); 	 
      tab.setPreferredSize(new Dimension(500, 170));
      tab.pageOpened();      
      ArrayList<Object> list = new ArrayList<Object>();
      list.add(" 学籍表に設定するデータを記入して下さい。      ");
      list.add(" ");
      list.add(tab);
      
      int ans =  JOptionPane.showOptionDialog(commonInfo.getFrame(),
					      list.toArray(), 
					      "学籍の設定", 
					      JOptionPane.OK_CANCEL_OPTION,
					      JOptionPane.QUESTION_MESSAGE,
					      null, null, null);
      if (ans == JOptionPane.OK_OPTION) {
	int index = tab.getSelectedIndex();
	String setValue = tab.getValueFromColumnCodeMap(columnCode);
	if (setValue != null) {
	  int res = commonInfo.commonInfoMethods.updateGakusekiData(STUDENT_CODE, columnCode, setValue);
	  if (res > 0) {
	    pageOpened();
	  }
	}
      }  
    } else {
      String columnValue = (String)tableModel.titleValueMap.get(columnCode);      
      JTextField editor = new JTextField(100);
      editor.setFont(new Font("DialogInput", Font.PLAIN, 12));
      editor.setText(" " + columnValue);
      editor.setEditable(true);
      editor.setBorder(new TitledBorder(columnCode + " = " + columnValue));

      ArrayList<Object> list = new ArrayList<Object>();
      list.add(" 学籍表に設定するデータを記入して下さい。      ");
      list.add(" ");
      list.add(editor);

      int ans =  JOptionPane.showOptionDialog(commonInfo.getFrame(),
					      list.toArray(), 
					      "学籍の設定", 
					      JOptionPane.OK_CANCEL_OPTION,
					      JOptionPane.QUESTION_MESSAGE,
					      null, null, null);
      if (ans == JOptionPane.OK_OPTION) {
	String setValue = editor.getText().trim();	
	if (setValue.equals("")) {
	  setValue = " ";
	}
	int res = commonInfo.commonInfoMethods.updateGakusekiData(STUDENT_CODE, columnCode, setValue);
	if (res > 0) {
	  pageOpened();
	}
      }
    }
  }
}
