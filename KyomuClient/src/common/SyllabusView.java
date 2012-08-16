package common;

import clients.*;
import java.util.*;
import javax.swing.*;
import javax.swing.border.*;
import java.awt.*;
import java.io.*;
import xml.*;
//import javax.swing.text.*;
import javax.swing.tree.*;
//import javax.swing.event.*;
import javax.xml.parsers.*;
import org.xml.sax.*;
//import org.w3c.dom.*;

public class SyllabusView extends JPanel {
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

  public JTextArea textArea;
  public JEditorPane editPane;
  public JScrollPane jsp;
  public String schoolYear;
  public String subjectCode;
  public String teacherCode;
  public String classCode;
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

  public SyllabusView(String tableViewType,
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
				    columnTitleList, columnCodeList, 
				    columnDisplayList);
    setSyllabusView();
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
    textArea = new JTextArea();  
    textArea.setFont( new Font("DialogInput", Font.PLAIN, 12) );
    //    textArea.setFont( new Font("Serif", Font.PLAIN, 11) );
    textArea.setLineWrap(true);
    textArea.setEditable(false);
    textArea.setTabSize(8);
    textArea.setBorder(new EmptyBorder(2, 2, 2, 2));
    JScrollPane jsp2 = new JScrollPane( textArea );
    jsp2.setMinimumSize( new Dimension(180, 180) );
    add(jsp2, BorderLayout.NORTH);

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

    String queryParamValues = schoolYear + "|" + subjectCode + "|" + classCode;  
    tableModel.setTableData(serviceName, panelID, switchCode, queryParamValues); 
    setTextArea();    
    setEditPane_1();
  }

  public void setTextArea() {
    HashSet<String> deptSet = new HashSet<String>();
    HashSet<String> gakuSet = new HashSet<String>();
    HashSet<String> weekHourSet = new HashSet<String>();
    HashSet<String> roomSet = new HashSet<String>();
    deptList = "";
    gakuList = "";
    weekHourList = "";
    roomList = "";

    for (int i = 0; i < tableModel.getRowCount(); i++) {
      schoolYearName  = tableModel.getCellDisplayAt(i, "開講年度");
      semesterName    = tableModel.getCellDisplayAt(i, "学期");
      subjectName     = tableModel.getCellDisplayAt(i, "授業科目名");
      className       = tableModel.getCellDisplayAt(i, "クラス");
      teacherName     = tableModel.getCellDisplayAt(i, "担当教員");
      mailAddress     = tableModel.getCellDisplayAt(i, "メール");
      englishSubjectName     = tableModel.getCellDisplayAt(i, "英文科目表記");
      englishTeacherName     = tableModel.getCellDisplayAt(i, "英文教員表記");
      kubunName       = tableModel.getCellDisplayAt(i, "科目区分");
      reqName         = tableModel.getCellDisplayAt(i, "単位区分");
      unitName        = tableModel.getCellDisplayAt(i, "単位数") + "単位";

      String department  = tableModel.getCellDisplayAt(i, "学科/専攻");
      if (!deptSet.contains(department)) {
	deptSet.add(department);
	deptList = deptList + department + " ";
      }
      String gakunen     = tableModel.getCellDisplayAt(i, "学年");
      if (!gakuSet.contains(gakunen)) {
	gakuSet.add(gakunen);
	gakuList = gakuList + gakunen + " ";
      }
      String week        = tableModel.getCellDisplayAt(i, "週");
      String hour        = tableModel.getCellDisplayAt(i, "時限");
      String weekHour = week + " " + hour;
      if (!weekHourSet.contains(weekHour)) {
	weekHourSet.add(weekHour);
	if (!weekHourList.equals("")) {
	  weekHourList = weekHourList + ", " + weekHour;
	} else {
	  weekHourList = weekHour;
	}
      }
      String room = tableModel.getCellDisplayAt(i, "講義室");
      if (!roomSet.contains(room)) {
	roomSet.add(room);
	roomList = roomList + room + " ";
      }
    }
    
    StringBuffer sbuf = new StringBuffer();
    sbuf.append(" 開講年度:  " + schoolYearName);
    sbuf.append("\n   科目名: " + subjectName + " (クラス " + className + ")  " + englishSubjectName);
    sbuf.append("\n 担当教員: " + teacherName + " " + mailAddress );
    sbuf.append("\n 対象学科: " + deptList);
    sbuf.append("\n 対象学年: " + gakuList);
    sbuf.append("\n 科目区分: " + kubunName + " " + reqName + " " + unitName);
    sbuf.append("\n   時間割: " + semesterName + "  " + weekHourList);
    sbuf.append("\n   講義室: " + roomList);
    textArea.setText(sbuf.toString());  
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
    HTMLMakerJT maker = new HTMLMakerJT();      
    UJTVisitor.traverse(root, maker);
    String text = maker.getText(); 
    editPane.setText(text);
    editPane.setCaretPosition(0);
  }
    
  public void setEditPane_2() {
    HTMLMakerJT maker = new HTMLMakerJT();      
    UJTVisitor.traverse(root, maker);
    String text = maker.getText();  
    editPane.setText(text);
    editPane.setCaretPosition(0);
  }

  public void makeSyllabusLatex() {
    StringBuffer sbuf = new StringBuffer();
    makeSyllabusLatexText(sbuf);
    File f = null;
    int ret = chooser.showSaveDialog(commonInfo.getFrame());
    if (ret == JFileChooser.APPROVE_OPTION) {
      f = chooser.getSelectedFile(); 
    } else {
      return;
    }
    
    try {
      fout = new PrintWriter(new BufferedWriter(new FileWriter(f)));
      fout.println(sbuf.toString());
      fout.close();
    } catch (Exception e) {
      commonInfo.showMessage(e.toString());
      return;
    }
  }

  public void makeSyllabusLatexText(StringBuffer sbuf) {
    String lineSeparator = commonInfo.lineSeparator;
    sbuf.append("\\documentclass[11pt,twoside]{jarticle}\n");
    sbuf.append("\\def\\linesparpage#1{\\baselineskip=\\textheight\\divide\\baselineskip#1}");

    sbuf.append("\\setlength{\\oddsidemargin}{-8pt}").append(lineSeparator); 
    sbuf.append("\\setlength{\\evensidemargin}{-8pt}").append(lineSeparator); 
    sbuf.append("\\setlength{\\topmargin}{-30pt}").append(lineSeparator); 
    sbuf.append("\\setlength{\\textwidth}{450pt}").append(lineSeparator); 
    sbuf.append("\\setlength{\\textheight}{680pt}").append(lineSeparator); 
    sbuf.append("\\begin{document}").append(lineSeparator); 
    sbuf.append("\\linesparpage{60}").append(lineSeparator); 
    
    sbuf.append("\\begin{verbatim}").append(lineSeparator); 
    sbuf.append("   " + schoolYearName + "  " + semesterName).append(lineSeparator);
    sbuf.append("   科目名:　" + subjectName);
    if (englishSubjectName != null) {
      sbuf.append("  " + englishSubjectName).append(lineSeparator);
    } else {
      sbuf.append(lineSeparator);
    }
    sbuf.append("   クラス:　" + className).append(lineSeparator);
    sbuf.append(" 担当教員:　" + teacherName + "  " + mailAddress).append(lineSeparator);
    sbuf.append(" 対象学科:　" + deptList + " " + gakuList).append(lineSeparator);
    sbuf.append(" 科目区分:　" +kubunName +" " + reqName + " " + unitName).append(lineSeparator);
    sbuf.append(" 　時間割:　" +semesterName + "  " + weekHourList ).append(lineSeparator);
    sbuf.append(" 　講義室:　" +roomList).append(lineSeparator);              
    sbuf.append("\\end{verbatim}").append(lineSeparator); 

    JLatexMakerJT maker = new JLatexMakerJT();
    UJTVisitor.traverse(root, maker);
    String text = maker.getText();
    sbuf.append(text);
    sbuf.append("\\end{document}");
  }
 

  public void makeSyllabusText() {
    StringBuffer sbuf = new StringBuffer();
    makeSyllabusPlainText(sbuf);
    File f = null;
    int ret = chooser.showSaveDialog(commonInfo.getFrame());
    if (ret == JFileChooser.APPROVE_OPTION) {
      f = chooser.getSelectedFile(); 
    } else {
      return;
    }
    
    try {
      fout = new PrintWriter(new BufferedWriter(new FileWriter(f)));
      fout.println(sbuf.toString());
      fout.close();
    } catch (Exception e) {
      commonInfo.showMessage(e.toString());
      return;
    }
  }

  public void makeSyllabusPlainText(StringBuffer sbuf) {
    String lineSeparator = commonInfo.lineSeparator;
    
    sbuf.append("   " + schoolYearName + "  " + semesterName).append(lineSeparator);
    sbuf.append("   科目名:　" + subjectName);
    if (englishSubjectName != null) {
      sbuf.append("  " + englishSubjectName).append(lineSeparator);
    } else {
      sbuf.append(lineSeparator);
    }
    sbuf.append("   クラス:　" + className).append(lineSeparator);
    sbuf.append(" 担当教員:　" + teacherName + "  " + mailAddress).append(lineSeparator);
    sbuf.append(" 対象学科:　" + deptList + " " + gakuList).append(lineSeparator);
    sbuf.append(" 科目区分:　" +kubunName +" " + reqName + " " + unitName).append(lineSeparator);
    sbuf.append(" 　時間割:　" +semesterName + " " + weekHourList).append(lineSeparator);
    sbuf.append(" 　講義室:　" +roomList).append(lineSeparator).append(lineSeparator);  

    TextMakerJT maker = new TextMakerJT();
    UJTVisitor.traverse(root, maker);
    String text = maker.getText();
    sbuf.append(text);
  }
}
