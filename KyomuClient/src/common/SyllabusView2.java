package common;

import clients.*;
import java.util.*;
import javax.swing.*;
//import javax.swing.border.*;
import java.awt.*;
import java.io.*;
import xml.*;
//import javax.swing.text.*;
import javax.swing.tree.*;
//import javax.swing.event.*;
import javax.xml.parsers.*;
import org.xml.sax.*;
//import org.w3c.dom.*;

public class SyllabusView2 extends JPanel {
  protected String tableViewType;
  protected String serviceName;
  protected String nodePath;
  protected String panelID;

  public CommonInfo commonInfo;
  public TabbedPaneBase parentTabbedPane;
  public DataPanelBase  dataPanel;

  //**** TableViewInfo ****//
  public ArrayList<Integer> columnNumberList = new ArrayList<Integer>();  
  public ArrayList<String> columnTitleList = new ArrayList<String>();   
  public ArrayList<String> columnCodeList = new ArrayList<String>();    
  public ArrayList<String> columnDisplayList = new ArrayList<String>(); 
  public ArrayList<Integer> columnWidthList = new ArrayList<Integer>();   
  public ArrayList<Color> columnFgColorList = new ArrayList<Color>(); 
  public ArrayList<Color> columnBgColorList = new ArrayList<Color>(); 
  public ArrayList<String> columnRendererList = new ArrayList<String>();

  protected String methodWhenOpened;
  protected int fontSize;
  protected int rowHeight;
  protected Color tableFgColor;
  protected Color tableBgColor;
  protected String sorterType;
  protected String selectionMode;
  protected String switchMethod;
  protected String switchMethodParam;
  protected String rowSelectMethod;
  protected String rowSelectMethodParam;

  protected String switchCode = "0";  // default switch code for table query
  protected String queryParams;
  protected String queryParamValues;
  protected String queryParamDisplays;
  protected boolean refreshTableFlag = false;
  protected int selectedRow = -1;

  public TableModelBase tableModel;

  public JEditorPane editPane;
  public JScrollPane jsp;
  public String schoolYear;
  public String subjectCode;
  public String teacherCode;
  public String classCode;
  public String titleText;
  public DefaultMutableTreeNode root = null;

  private String schoolYearName;
  private String semesterName;
  private String subjectName;
  private String englishName;
  private String className;
  private String teacherName;
  private String mailAddress;
  private String kubunName;
  private String reqName;
  private String unitName;
  private String deptList;
  private String gakuList; 
  private String weekHourList; 
  private String roomList; 
  private String englishSubjectName = "";
  private String englishTeacherName = "";

  PrintWriter fout;
  JFileChooser chooser = new JFileChooser(); 

  public SyllabusView2(String tableViewType,
		       String serviceName,
		       String nodePath,
		       String panelID,
		       CommonInfo commonInfo, 
		       TabbedPaneBase tabbedPane,
		       DataPanelBase dataPanel) {
    this.tableViewType = tableViewType;
    this.serviceName = serviceName;
    this.nodePath = nodePath;
    this.panelID = panelID;
    this.commonInfo = commonInfo;
    this.parentTabbedPane = tabbedPane;
    this.dataPanel = dataPanel;

    setTableViewInfo();
    setTableColumnInfo();
    tableModel = new TableModelBase(commonInfo, serviceName, panelID, 
				    columnTitleList, columnCodeList, columnDisplayList);
    setSyllabusView();
    commonInfo.syllabusView2 = this;
  }


  private void setTableViewInfo() {
    String tableViewStruct = commonInfo.getTableViewStruct(serviceName, panelID);
    if (tableViewStruct != null) {
      String[] lines = tableViewStruct.split("\\$");
      String[] tokens = lines[0].split("\\|");
      int size = tokens.length;
      if (size == 11) {
	for (int i = 0; i < 11; i++) {
	  String token = tokens[i].trim();
	  if (token.equals("")) token = null;
	  switch (i) {
	  case 0: 
	    methodWhenOpened = token; break;
	  case 1: 
	    fontSize = commonInfo.getFontSize(token); break;
	  case 2: 
	    rowHeight = commonInfo.getRowHeight(token); break;
	  case 3: 
	    tableFgColor = commonInfo.getTableFgColor(token); break;
	  case 4: 
	    tableBgColor = commonInfo.getTableBgColor(token); break;
	  case 5: 
	    sorterType = token; break;
	  case 6: 
	    selectionMode = token; break;
	  case 7: 
	    switchMethod = token; break;
	  case 8:
	    switchMethodParam = token; break;
	  case 9: 
	    rowSelectMethod = token; break;
	  case 10:
	    rowSelectMethodParam = token; break;
	  }
	}
      } else {
	commonInfo.showMessageLong("format error : TableViewStruct $ " + tableViewStruct); 
      }
    } else {
      commonInfo.showMessageLong("TableViewStruct is not found $ serviceName = " + serviceName + ", panelID = " + panelID); 
    }
  }

  private void setTableColumnInfo() {
    String tableColumnStruct = commonInfo.getTableColumnStruct(serviceName, panelID);
    if (tableColumnStruct != null) {
      String[] tableColumnsInfo = tableColumnStruct.split("\\$"); 
      for (String tableColumnInfo : tableColumnsInfo) {
	String[] tokens = tableColumnInfo.split("\\|"); 
	int size = tokens.length;
	if (size == 8) {
	  for (int i = 0; i < 8; i++) {
	    String token = tokens[i].trim();
	    if (token.equals("")) token = null;
	    switch (i) {
	    case 0: 
	      columnNumberList.add(new Integer(token)); break;
	    case 1:
	      columnTitleList.add(token); break;
	    case 2:
	      columnCodeList.add(token); break;
	    case 3:
	      columnDisplayList.add(token); break;
	    case 4:
	      columnWidthList.add(new Integer(token)); break;
	    case 5:
	      columnFgColorList.add(commonInfo.getTableFgColor(token)); break;
	    case 6:
	      columnBgColorList.add(commonInfo.getTableBgColor(token)); break;
	    case 7:
	      columnRendererList.add(token); break;
	    }
	  }
	} else {
	  commonInfo.showMessageLong("format error in TableColumnStruct $ " + tableColumnInfo);
	}
      }
    } else {  
      commonInfo.showMessageLong("TableColumnStruct is not found $ serviceName = " + serviceName + ", panelID = " + panelID); 
    }
  }

  public void setSyllabusView() { 
    setLayout(new BorderLayout()); 
    editPane = new JEditorPane();
    editPane.setContentType("text/html");
    editPane.setEditable(false);
    jsp = new JScrollPane( editPane );
    add(jsp, BorderLayout.CENTER);
  }
 
  public void pageOpened() { 
    String[] columnCodes = { "SCHOOL_YEAR", "SUBJECT_CODE", "TEACHER_CODE", "CLASS_CODE"};    
    HashMap<String, String> valueMap = new HashMap<String, String>();
    HashMap<String, String> displayMap = new HashMap<String, String>();
    
    String value = null;
    String display = null;
    for (String columnCode : columnCodes) {
      if (parentTabbedPane.columnCodeMapContains(columnCode)) {
	value = parentTabbedPane.getValueFromColumnCodeMap(columnCode);
	display = parentTabbedPane.getDisplayFromColumnCodeMap(columnCode);
	valueMap.put(columnCode, value);
	displayMap.put(columnCode, display);
      } else {
	if (commonInfo.structControlDebugMode) {
	  String str = "debugMode: パラメータ " + columnCode + " の値を指定して下さい。";
	  value = commonInfo.getDialogInput(str);
	  if ((value != null) && (!(value.trim()).equals(""))) {
	    valueMap.put(columnCode, value);
	    displayMap.put(columnCode, " ");
	  }
	}
      }
    }
    schoolYear  = valueMap.get("SCHOOL_YEAR");
    subjectCode = valueMap.get("SUBJECT_CODE");
    teacherCode = valueMap.get("TEACHER_CODE");
    classCode   = valueMap.get("CLASS_CODE");
    String subjectName = displayMap.get("SUBJECT_CODE");
    String teacherName = displayMap.get("TEACHER_CODE");
    String className   = displayMap.get("CLASS_CODE");

    if ((schoolYear == null) || (subjectCode == null) || (classCode == null)) {
      commonInfo.showMessageLong("SyllabusView: $ SCHOOL_YEAR or SUBJECT_CODE or CLASS_CODE $ の値が設定されていません。");
      return;
    }
    dataPanel.setTitleText(subjectName + " (" + className + ") " + teacherName);
    titleText = subjectName + " (" + className + ")  " + teacherName;

    String queryParamValues = schoolYear + "|" + subjectCode + "|" + classCode;    
    tableModel.setTableData(serviceName, panelID, switchCode, queryParamValues); 
    setEditPane_1();
  }

  public void setEditPane_1() {
    org.w3c.dom.Element element = null;
    String xmltext = commonInfo.commonInfoMethods.getSyllabusXml(schoolYear, subjectCode, teacherCode);
    if (xmltext == null) return;
    
    try {
      DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
      DocumentBuilder builder = factory.newDocumentBuilder();
      builder.setErrorHandler(new SimpleErrorHandler());
      StringReader reader = new StringReader(xmltext);
      InputSource source = new InputSource(reader);
      org.w3c.dom.Document doc = builder.parse(source);      
      element = doc.getDocumentElement();
    } catch (ParserConfigurationException e) {
      System.out.println("ParserConfigurationException");
      e.printStackTrace();
    } catch(SAXException e) {
      System.out.println("SAXException");
      e.printStackTrace();      
    } catch(IOException e) {
      System.out.println("IOException");
      e.printStackTrace();      
    }      
    if (element == null) {
      System.out.println("該当なし");
      return;
    }     
 
    root = UFile.makeJTTree(element);
    HTMLMakerJT2 maker = new HTMLMakerJT2();      
    UJTVisitor.traverse(root, maker);
    String text = maker.getText(); 
    editPane.setText(text);
    editPane.setCaretPosition(0);
  }
    
  public void setEditPane_2() {
    HTMLMakerJT2 maker = new HTMLMakerJT2();      
    UJTVisitor.traverse(root, maker);
    String text = maker.getText();  
    editPane.setText(text);
    editPane.setCaretPosition(0);
  }

  public String[] getListOfSubjectItems() {
    TextMakerJT2 maker = new TextMakerJT2(); 
    UJTVisitor.traverse(root, maker);
    String text = maker.getText();  
    String[] lines = text.split("\\|");
    return lines;
  }

  public String[] getListOfObjectives() {
    TextMakerJT3 maker = new TextMakerJT3(); 
    UJTVisitor.traverse(root, maker);
    String text = maker.getText();  
    String[] lines = text.split("\\|");
    return lines;
  }
}
