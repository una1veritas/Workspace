package clients;

import common.*;
//import clients.*;

import javax.swing.*;
import javax.swing.border.*;
import java.awt.event.*;
import java.awt.*;
import java.net.URI;
import java.util.*;
//import java.io.*;
import javax.print.*;
import javax.print.attribute.*;
import javax.print.attribute.standard.*;
//import javax.print.event.*;

public class CommonInfo {
  private JFrame frame;
  private javax.swing.Timer timer;
  public String rootServiceName;
  public String serverHost;
  public int serverPort;
  public int timeoutMinute;
  public CommonInfoMethods commonInfoMethods;
  public ServerConnectionBase serverConn;
  
  public final String keyTrustURL = "http://kaoru.gakumu.jimui.kyutech.ac.jp/kyomu/nextGenerationClientTrust";
  public final String storePasswd = "NextGenerationKyomuInfo";
  public final String gakumuAddress = "jho-gakumu@jimu.kyutech.ac.jp";

  // ユーザー管理用に追加
  public String login_id;

  // 次の定数はプログラムによって管理されており、学務のルールが変更された場合
  // には、定数の値を変更する必要がある。
  // なにいってんの．．
  public double undergradRegistrUpperBound = 24.0;    // 学期毎に履修申告できる(集中を除く)単位数
  public double graduateRegistrUpperBound = 16.0;     // 学期毎に履修申告できる(集中を除く)単位数

  public boolean structControlDebugMode = false;      // システム構成情報管理ツールにおける
                                                      // デバッグモード
/*    
    public CommonInfo(JFrame frame, String rootServiceName) { 
      this(frame, rootServiceName, 30);
      serverHost = "131.206.103.7";
      serverPort = 3404;
    }
*/
    /*
    public void connectToServer() {
      serverConn = new KyomuServerConnection(serverHost, serverPort, this);
    }
    */
    public void connectToServer() {
        serverConn = new ServerConnection(serverHost, serverPort, this);
    }

/*  
  public CommonInfo(JFrame frame, String rootService, 
			int timeoutMinute) {
    this.frame = frame;
    this.rootServiceName = rootService.substring(1,rootService.length()-1);
    this.timeoutMinute = timeoutMinute;
    login_id = "";
    serverPort = 0;
    serverHost = "";
  }
  */
  public CommonInfo(JFrame myframe, URI tool) { 
//	    super(frame, tool.getPath(), 10);
	    frame = myframe;
	    String seviceName = tool.getPath();
	    rootServiceName = seviceName.substring(1,seviceName.length()-1);
	    login_id = "";
//
	        serverHost = tool.getHost();
	        serverPort = tool.getPort();
	        if ( tool.getScheme().equals("StudentTool")) {
	        	timeoutMinute = 10;
	        } else {
	        	timeoutMinute = 30;
	        }
	  }


  /*
  public CommonInfoBase(JFrame frame, String rootServiceName) {
    this(frame, rootServiceName, 10);
  }
  */
  /*
  public void connectToServer() {
  }
*/

  public void init() {
    lineSeparator = System.getProperty("line.separator");
    setTimer();
    setPresentDate();
    setColorMap();
    setGakumuCodeMap();
    setPsPrintService();
    commonInfoMethods = new CommonInfoMethods(this, serverConn);
    commonInfoMethods.setRegistrInfo();
  }

  private HashMap<String, Color> colorMap = new HashMap<String, Color>();
  private HashMap<String, String> gakumuCodeToNameMap = new HashMap<String, String>();
  private HashMap<String, String> gakumuCodeToShorterNameMap = new HashMap<String, String>();
  private HashSet<String> gakumuCategorySet = new HashSet<String>();
  private HashMap<String, String> commonCodeToValueMap = new HashMap<String, String>();
  private HashMap<String, String> commonCodeToDisplayMap = new HashMap<String, String>();
  
  public int thisYear;
  public int thisMonth;
  public int thisDay;
  public int thisSchoolYear;
  public int thisSemester;
  public String thisSemesterName;

  public int secondSemesterStartMonth = 0;
  public int secondSemesterStartDay;

  /*
  private int   defaultRowHeight; 
  private Color defaultTableBgColor;
  private Color defaultSimpleButtonColor;
  private Color defaultUpdateButtonColor;
  private Color defaultTitleFgColor; 
  private Color defaultTitleBgColor; 
*/
  public String lineSeparator;
  
//***** 現在開かれているタブに関する情報 **********//
  private String  presentServiceName = null;
  private String  presentNodePath = null;

  public String getPresentServiceName() {
    return presentServiceName;
  }

  public String getPresentNodePath() {
    return presentNodePath;
  }

  public void setPresentServiceName(String serviceName) {
    presentServiceName = serviceName;
  }

  public void setPresentNodePath(String nodePath) {
    presentNodePath = nodePath;
  }
  
//***** 職員ユーザに関する情報 **********//
  public  String  USER_ID = "";
  public  String  STAFF_CODE = "";
  public  String  STAFF_NAME = "";
  public  String  STAFF_ATTRIB = "";
  public  String  STAFF_QUALIFICATION = "";
  public  String  MAIL_ADDRESS = "";
  public  String  LOCAL_ATTRIB = "";
  public  String  STAFF_DEPARTMENT = ""; 
  public  String  STAFF_SENKO = "";

  public void addStaffAttribToCommonCodeMap() {
    addCommonCodeMap("STAFF_CODE", STAFF_CODE.trim(), STAFF_NAME);
    addCommonCodeMap("MY_STAFF_CODE", STAFF_CODE.trim(), STAFF_NAME);
    addCommonCodeMap("MY_STAFF_NAME", STAFF_NAME.trim(), STAFF_NAME);
    addCommonCodeMap("STAFF_QUALIFICATION", STAFF_QUALIFICATION.trim(), STAFF_QUALIFICATION);
    addCommonCodeMap("STAFF_DEPARTMENT", STAFF_DEPARTMENT.trim(), STAFF_DEPARTMENT);
    addCommonCodeMap("LOCAL_ATTRIB", LOCAL_ATTRIB.trim(), LOCAL_ATTRIB);

    int qual = Integer.parseInt(STAFF_QUALIFICATION);
    if (qual < 2) {
      rootServiceName = "StaffTool";
    }	
  }
  
//***** 職員ユーザに関する情報 (デバッグ用) **********//
  public  boolean staffDebugUserSelected = false;
  public  String  USER_ID_D = "";
  public  String  STAFF_CODE_D = "";
  public  String  STAFF_NAME_D = "";
  public  String  STAFF_ATTRIB_D = "";
  public  String  STAFF_QUALIFICATION_D = "";
  public  String  MAIL_ADDRESS_D = "";
  public  String  LOCAL_ATTRIB_D = "";
  public  String  STAFF_DEPARTMENT_D = ""; 
  public  String  STAFF_SENKO_D = "";

  public void addStaffAttribToCommonCodeMapD() {
    addCommonCodeMap("STAFF_CODE_D", STAFF_CODE_D.trim(), STAFF_NAME_D);
    addCommonCodeMap("MY_STAFF_CODE_D", STAFF_CODE_D.trim(), STAFF_NAME_D);
    addCommonCodeMap("MY_STAFF_NAME_D", STAFF_NAME_D.trim(), STAFF_NAME_D);
    addCommonCodeMap("STAFF_QUALIFICATION_D", STAFF_QUALIFICATION_D.trim(), STAFF_QUALIFICATION_D);
    addCommonCodeMap("STAFF_DEPARTMENT_D", STAFF_DEPARTMENT_D.trim(), STAFF_DEPARTMENT_D);
    addCommonCodeMap("LOCAL_ATTRIB_D", LOCAL_ATTRIB_D.trim(), LOCAL_ATTRIB_D);

    int qual = Integer.parseInt(STAFF_QUALIFICATION_D);
    if (qual < 2) {
      rootServiceName = "StaffTool";
    }	
  }


//***** 学生ユーザに関する情報 **********//
  public  String  STUDENT_CODE = "";
  public  String  STUDENT_NAME = "";
  public  String  STUDENT_STATUS = "";
  public  String  STUDENT_FACULTY = "";
  public  String  STUDENT_DEPARTMENT = "";
  public  String  STUDENT_COURSE = "";
  public  String  STUDENT_COURSE_2 = "";
  public  String  STUDENT_GAKUNEN = "";
  public  String  STUDENT_CURRICULUM_YEAR = "";
  public  String  STUDENT_SUPERVISOR = "";
  public  String  STUDENT_SUPERVISOR_NAME = "";

  public void addStudentAttribToCommonCodeMap() {
    STAFF_QUALIFICATION = "0";
    addCommonCodeMap("STUDENT_CODE", STUDENT_CODE, STUDENT_NAME);
    addCommonCodeMap("MY_STUDENT_CODE", STUDENT_CODE, STUDENT_NAME);
    addCommonCodeMap("STUDENT_FACULTY", STUDENT_FACULTY, STUDENT_FACULTY);
    addCommonCodeMap("MY_FACULTY", STUDENT_FACULTY, STUDENT_FACULTY);
    addCommonCodeMap("STUDENT_DEPARTMENT", 
		     STUDENT_DEPARTMENT, STUDENT_DEPARTMENT);
    addCommonCodeMap("MY_DEPARTMENT", 
		     STUDENT_DEPARTMENT, STUDENT_DEPARTMENT);
    addCommonCodeMap("STUDENT_COURSE", STUDENT_COURSE, STUDENT_COURSE);
    addCommonCodeMap("STUDENT_COURSE_2", STUDENT_COURSE_2, STUDENT_COURSE_2);
    addCommonCodeMap("MY_COURSE", STUDENT_COURSE, STUDENT_COURSE);
    addCommonCodeMap("MY_COURSE_2", STUDENT_COURSE_2, STUDENT_COURSE_2);
    addCommonCodeMap("STUDENT_GAKUNEN", STUDENT_GAKUNEN, STUDENT_GAKUNEN);
    addCommonCodeMap("MY_GAKUNEN", STUDENT_GAKUNEN, STUDENT_GAKUNEN);
    addCommonCodeMap("STUDENT_SUPERVISOR", 
		     STUDENT_SUPERVISOR, STUDENT_SUPERVISOR_NAME);
    addCommonCodeMap("MY_SUPERVISOR", 
		     STUDENT_SUPERVISOR, STUDENT_SUPERVISOR_NAME);
    addCommonCodeMap("STUDENT_CURRICULUM_YEAR", 
		     STUDENT_CURRICULUM_YEAR, STUDENT_CURRICULUM_YEAR);
    addCommonCodeMap("MY_CURRICULUM_YEAR", 
		     STUDENT_CURRICULUM_YEAR, STUDENT_CURRICULUM_YEAR);
    
    if (!STUDENT_FACULTY.equals("11")) {
      rootServiceName = "GradStudentTool";
    }
  }

//***** 学生ユーザに関する情報 (デバッグ用)  **********//
  public  boolean studentDebugUserSelected = false;
  public  String  STUDENT_CODE_D = "";
  public  String  STUDENT_NAME_D = "";
  public  String  STUDENT_STATUS_D = "";
  public  String  STUDENT_FACULTY_D = "";
  public  String  STUDENT_DEPARTMENT_D = "";
  public  String  STUDENT_COURSE_D = "";
  public  String  STUDENT_COURSE_2_D = "";
  public  String  STUDENT_GAKUNEN_D = "";
  public  String  STUDENT_CURRICULUM_YEAR_D = "";
  public  String  STUDENT_SUPERVISOR_D = "";
  public  String  STUDENT_SUPERVISOR_NAME_D = "";

  public void addStudentAttribToCommonCodeMapD() {
    STAFF_QUALIFICATION_D = "0";
    addCommonCodeMap("STUDENT_CODE_D", STUDENT_CODE_D, STUDENT_NAME_D);
    addCommonCodeMap("MY_STUDENT_CODE_D", STUDENT_CODE_D, STUDENT_NAME_D);
    addCommonCodeMap("STUDENT_FACULTY_D", STUDENT_FACULTY_D, STUDENT_FACULTY_D);
    addCommonCodeMap("MY_FACULTY_D", STUDENT_FACULTY_D, STUDENT_FACULTY_D);
    addCommonCodeMap("STUDENT_DEPARTMENT_D", 
		     STUDENT_DEPARTMENT_D, STUDENT_DEPARTMENT_D);
    addCommonCodeMap("MY_DEPARTMENT_D", 
		     STUDENT_DEPARTMENT_D, STUDENT_DEPARTMENT_D);
    addCommonCodeMap("STUDENT_COURSE_D", STUDENT_COURSE_D, STUDENT_COURSE_D);
    addCommonCodeMap("STUDENT_COURSE_2_D", STUDENT_COURSE_2_D, STUDENT_COURSE_2_D);
    addCommonCodeMap("MY_COURSE_D", STUDENT_COURSE_D, STUDENT_COURSE_D);
    addCommonCodeMap("MY_COURSE_2_D", STUDENT_COURSE_2_D, STUDENT_COURSE_2_D);
    addCommonCodeMap("STUDENT_GAKUNEN_D", STUDENT_GAKUNEN_D, STUDENT_GAKUNEN_D);
    addCommonCodeMap("MY_GAKUNEN_D", STUDENT_GAKUNEN_D, STUDENT_GAKUNEN_D);
    addCommonCodeMap("STUDENT_SUPERVISOR_D", 
		     STUDENT_SUPERVISOR_D, STUDENT_SUPERVISOR_NAME_D);
    addCommonCodeMap("MY_SUPERVISOR_D", 
		     STUDENT_SUPERVISOR_D, STUDENT_SUPERVISOR_NAME_D);
    addCommonCodeMap("STUDENT_CURRICULUM_YEAR_D", 
		     STUDENT_CURRICULUM_YEAR_D, STUDENT_CURRICULUM_YEAR_D);
    addCommonCodeMap("MY_CURRICULUM_YEAR_D", 
		     STUDENT_CURRICULUM_YEAR_D, STUDENT_CURRICULUM_YEAR_D);

    if (!STUDENT_FACULTY_D.equals("11")) {
      rootServiceName = "GradStudentTool";
    }
  }

  //***** デバッグモードの設定関係 **********//
  public void setDebugMode() {
    structControlDebugMode = true;
    if (studentDebugUserSelected) {
      serverConn.setStudentDebugMode(STUDENT_CODE_D);
    } else if (staffDebugUserSelected) {
      serverConn.setStaffDebugMode(USER_ID_D);
    }
  }

  public void resetDebugMode() {
    structControlDebugMode = false;
    serverConn.resetServerDebugMode();
  }

  //***** データ入力関連 **********//
  public boolean unsavedDataExist = false;

  public boolean unsavedDataExists() {
    return unsavedDataExist;
  }

  public void setUnsavedDataExistFlag(boolean flag) {
    unsavedDataExist = flag;
  }

  public SyllabusView2 syllabusView2 = null;  

  //*****  PrintService のデータ取得には   ******//  
  //*****  30秒程度の計算時間を必要とする  ******//   
  //*****  場合があるので、予め取得を済ま  ******//  
  //*****  せておく。                      ******//  

  public PrintService[] psPrintService = null;  
  public String[] psPrintServiceName = null;
  public PrintService selectedPsPrintService = null;
  public String selectedPsPrintServiceName = null;

  private void setPsPrintService() {    
    Thread thread = new PsPrinterThread();
    thread.setDaemon(true);
    thread.start();
  }

  class PsPrinterThread extends Thread {
    public void run() {        
      try{
	DocFlavor psFlavor = DocFlavor.INPUT_STREAM.POSTSCRIPT;
	PrintRequestAttributeSet aset = new HashPrintRequestAttributeSet();
	aset.add(MediaSizeName.ISO_A4);
	psPrintService = PrintServiceLookup.lookupPrintServices(psFlavor, aset);
	if (psPrintService != null) {
	  int size = psPrintService.length;	
	  if (size == 0) {
	    psPrintService = null;
	  }
	  psPrintServiceName = new String[size];
	  for (int i = 0; i < size; i++) {
	    psPrintServiceName[i] = psPrintService[i].getName();
	  }	
	}
      } catch (Exception e) {
	showMessage(e.toString());  
      }
    }
  }

  public JFrame getFrame() {
    return frame;
  }

  private void setTimer() {
    timer = new javax.swing.Timer(1000 * 60 * timeoutMinute, new ActionListener() {
      public void actionPerformed(ActionEvent e) {
	System.out.println(rootServiceName + ":  ...  time over ...");
	System.exit(0);
      }} );
    timer.setRepeats(false);
    timer.start();
  }

  public void timerRestart() {
    timer.restart();
  }

  //*** communication with server ***//

  public String getQueryResult(String serviceName,
			       String panelID,
			       String switchCode,
			       String paramValues) {
    String key = "QUERY|" + serviceName + "|" + panelID + "#" + switchCode;
    return serverConn.query(key, paramValues);
  }

  public String getSingleStructQueryResult(String serviceName,
					   String panelID,
					   String switchCode,
					   String paramValues) {
    String key = "QUERY_SINGLE_STRUCT|" + serviceName + "|" + panelID + "#" + switchCode;
    return serverConn.query(key, paramValues);
  }

  public int getDeleteResult(String serviceName,
			     String deleteCode,
			     String paramValues) {  
    String key = "DELETE|" + serviceName + "|" + deleteCode;
    return serverConn.update(key, paramValues);
  }

  public int getUpdateResult(String serviceName,
			     String updateCode,
			     String paramValues) {
    String key = "UPDATE|" + serviceName + "|" + updateCode;
    return serverConn.update(key, paramValues);
  }

  public int getInsertResult(String serviceName,
			     String insertCode,
			     String paramValues) {
    String key = "INSERT|" + serviceName + "|" + insertCode;
    return serverConn.update(key, paramValues);
  }

  public int getSpecialResult(String serviceName,
			      String specialCode,
			      String paramValues) {
    String key = "SPECIAL|" + serviceName + "|" + specialCode;
    return serverConn.update(key, paramValues);
  }


  //********** Selectors **********************//

  private HashMap<String, Object> selectorMap = new HashMap<String, Object>();
  private HashMap<String, Object> editorMap = new HashMap<String, Object>();

  public JComponent getSelector(String selector, 
				String columnCode, String columnTitle,
				String oldCode, String oldName, int size) {
    if (selector != null) {
      if (selectorMap.containsKey(selector)) {
	return (JComponent) selectorMap.get(selector);
      } else {
	int height = (int) (400 / size);
	TabbedPaneBase tab;
	if (selector.equals("DateSelector")) {
	  tab = new TabbedPaneBase(selector, "root", this, null, "setCalendarToOpen");
	} else {
	  tab = new TabbedPaneBase(selector, "root", this, null, null);
	}
	tab.setPreferredSize(new Dimension(600, height));
	tab.pageOpened();
	selectorMap.put(selector, tab);
	return (JComponent) tab;
      }
    } else {
      JTextField editor = new JTextField(100);
      editor.setFont(new Font("DialogInput", Font.PLAIN, 12));
      editor.setText("  " + oldCode);
      editor.setEditable(true);
      String text = columnTitle + " = " + oldName;
      TitledBorder border = new TitledBorder(null, text, TitledBorder.LEFT, TitledBorder.TOP,
					     new Font("DialogInput", Font.PLAIN, 10));
      editor.setBorder(border);
      editorMap.put(columnCode, editor);
      return (JComponent) editor;
    }      
  }	

  public String getSelectorValue(String selector, String columnCodes) {
    if (selector != null) {
      TabbedPaneBase tab = (TabbedPaneBase) selectorMap.get(selector);
      int index = tab.getSelectedIndex();
      DataPanelBase dpb = (DataPanelBase) tab.getComponentAt(index);
      TableViewBase tableView = dpb.getTableView();      
      if (tableView.getSelectedRowCount() == 0) return null;

      String[] tokens = columnCodes.split("\\#");
      StringBuilder sbuf = new StringBuilder();
      for (String token : tokens) {
	String value = tab.getValueFromColumnCodeMap(token);
	if ((value == null) || (value.trim().equals(""))) {
	  value = " ";
	}
	sbuf.append(value).append("|");
      }
      return sbuf.toString();
    } else {
      String[] tokens = columnCodes.split("\\#");
      StringBuilder sbuf = new StringBuilder();
      for (String token : tokens) {
	JTextField editor = (JTextField) editorMap.get(token);
	String value = editor.getText().trim();
	if (value.equals("")) {
	  value = " ";
	} 
	sbuf.append(value).append("|");
      }
      return sbuf.toString();
    }
  }

  //*** get struct data from server ***//

  private void setPresentDate() {
    String ans = serverConn.queryCommon("QUERY|COMMON_QUERY|queryPresentDate", "empty");
    String[] tokens = ans.split("\\|");
    String yr = tokens[0];
    String mn = tokens[1];
    String dy = tokens[2];

    thisYear = Integer.parseInt(yr);
    thisMonth = Integer.parseInt(mn);
    thisDay = Integer.parseInt(dy);
    
    if (thisMonth <= 3) {
      thisSchoolYear = thisYear - 1;
      thisSemester = 2;
      thisSemesterName = "後学期";
    } else {
      thisSchoolYear = thisYear;
      if (thisMonth >= 10) {
	thisSemester = 2;
	thisSemesterName = "後学期";
      } else {
	thisSemester = 1;
	thisSemesterName = "前学期";
      }
    }  
    addCommonCodeMap("THIS_SEMESTER_TRUE", ""+thisSemester, thisSemesterName);

    //**  正確な「学期の設定」を取得する  **//

    setSecondSemesterStartDate();
    if (thisMonth == 9) {  
      if (thisDay >= secondSemesterStartDay) {
	thisSemester = 2;
	thisSemesterName = "後学期";
      }	
    } 
    
    addCommonCodeMap("THIS_SCHOOL_YEAR", ""+thisSchoolYear, ""+thisSchoolYear+"年");
    addCommonCodeMap("THIS_SEMESTER", ""+thisSemester, thisSemesterName);
    if (thisMonth < 10) {
      addCommonCodeMap("THIS_MONTH", "0"+thisMonth, ""+thisMonth+"月");
    } else {
      addCommonCodeMap("THIS_MONTH", ""+thisMonth, ""+thisMonth+"月");
    }
    if (thisDay < 10) {
      addCommonCodeMap("THIS_DAY", "0"+thisDay, ""+thisDay+"日");
    } else {
      addCommonCodeMap("THIS_DAY", ""+thisDay, ""+thisDay+"日");
    }
    addCommonCodeMap("SCHOOL_YEAR", ""+thisSchoolYear, ""+thisSchoolYear+"年");
    addCommonCodeMap("SEMESTER", ""+thisSemester, thisSemesterName);
  }
/*
  private String getTodaySchoolEvent() {
    String schoolEvent = " ";
    String key = "QUERY|COMMON_QUERY|querySchoolEvent";
    String paramValues = "empty";
    String answer = serverConn.queryCommon(key, paramValues);
    if (answer != null) {
      String[] lines = answer.split("\\$");
      String[] tokens = lines[0].split("\\|");
      schoolEvent = tokens[0];
    }
    return schoolEvent;
  }
*/
  private void setSecondSemesterStartDate() {
    String key = "QUERY|COMMON_QUERY|querySchoolEventList";
    String paramValues = ""+thisSchoolYear;
    String answer = serverConn.queryCommon(key, paramValues);
    if (answer != null) {
      String[] lines = answer.split("\\$");
      for (String line : lines) {
	String[] tokens = line.split("\\|");
//	String year  = tokens[0];
	String month = tokens[1];
	String day   = tokens[2];
	String schoolEvent  = tokens[3];
	if (schoolEvent.indexOf("後期履修申告") >= 0) {
	  secondSemesterStartMonth = Integer.parseInt(month.trim());
	  secondSemesterStartDay  = Integer.parseInt(day.trim());
	  return;
	} else if (schoolEvent.indexOf("後期授業開始") >= 0) {
	  secondSemesterStartMonth = Integer.parseInt(month.trim());
	  secondSemesterStartDay  = Integer.parseInt(day.trim());
	  return;
	}  
      }
    } 
    secondSemesterStartMonth = 9;
    secondSemesterStartDay  = 25;
  }

  private void setColorMap() {
    String ans = serverConn.queryCommon("QUERY|COMMON_QUERY|queryColorNameDef", "empty");
    String[] lines = ans.split("\\$");
    for (String line : lines) {
      String[] tokens = line.split("\\|");
      Color c = new Color(Integer.parseInt(tokens[1]),
			  Integer.parseInt(tokens[2]),
			  Integer.parseInt(tokens[3]));
      colorMap.put(tokens[0], c);
    }
  } 

  private void setGakumuCodeMap() { 
    String ans = serverConn.queryCommon("QUERY|COMMON_QUERY|queryGakumuCodeDef", "empty");
    String[] lines = ans.split("\\$");
    for (String line : lines) {
      String[] tokens = line.split("\\|");
      String codeCategory = tokens[0];
      String code         = tokens[1];
      String name         = tokens[2];
      String shorterName  = tokens[3];
      String key = codeCategory + "|" + code;
      gakumuCodeToNameMap.put(key, name);
      gakumuCodeToShorterNameMap.put(key, shorterName);
      gakumuCategorySet.add(codeCategory);
    }
  } 

  public boolean gakumuCategorySetContains(String codeCategory) {
    return gakumuCategorySet.contains(codeCategory);
  }
 
  public boolean gakumuCodeMapContains(String codeCategory, String gakumuCode) {
    String key = codeCategory + "|" + gakumuCode;
    return gakumuCodeToShorterNameMap.containsKey(key);
  }

  public String getGakumuCodeName(String codeCategory, String gakumuCode) {
    String key = codeCategory + "|" + gakumuCode;
    return gakumuCodeToNameMap.get(key);
  }
 
  public String getGakumuCodeShorterName(String codeCategory, String gakumuCode) {
    String key = codeCategory + "|" + gakumuCode;
    return gakumuCodeToShorterNameMap.get(key);
  }
  
  public String getTabbedPaneStruct(String serviceName, String nodePath) {
    String param = serviceName + "|" + nodePath;
    return serverConn.queryCommon("QUERY|COMMON_QUERY|queryTabbedPaneStruct", param);
  }
  
  public String getSplitPaneStruct(String serviceName, String nodePath) {
    String param = serviceName + "|" + nodePath;
    return serverConn.queryCommon("QUERY|COMMON_QUERY|querySplitPaneStruct", param);
  }

  public String getDataPanelStruct(String serviceName, String panelID) {
    String param = serviceName + "|" + panelID;
    return serverConn.queryCommon("QUERY|COMMON_QUERY|queryDataPanelStruct", param);
  }

  public String getTableViewStruct(String serviceName, String panelID) {
    String param = serviceName + "|" + panelID;
    return serverConn.queryCommon("QUERY|COMMON_QUERY|queryTableViewStruct", param);
  }

  public String getVarTableViewStruct(String serviceName, String panelID) {
    String param = serviceName + "|" + panelID;
    return serverConn.queryCommon("QUERY|COMMON_QUERY|queryVarTableViewStruct", param);
  }

  public String getHtmlViewStruct(String serviceName, String panelID) {
    String param = serviceName + "|" + panelID;
    return serverConn.queryCommon("QUERY|COMMON_QUERY|queryHtmlViewStruct", param);
  }

  public String getJikanwariViewStruct(String serviceName, String panelID) {
    String param = serviceName + "|" + panelID;
    return serverConn.queryCommon("QUERY|COMMON_QUERY|queryJikanwariViewStruct", param);
  }

  public String getJikanwariColumnStruct(String serviceName, String panelID) {
    String param = serviceName + "|" + panelID;
    return serverConn.queryCommon("QUERY|COMMON_QUERY|queryJikanwariColumnStruct", param);
  }

  public String getTableColumnStruct(String serviceName, String panelID) {
    String param = serviceName + "|" + panelID;
    return serverConn.queryCommon("QUERY|COMMON_QUERY|queryTableColumnStruct", param);
  }

  public String getSimpleButtonStruct(String serviceName, String panelID) {
    String param = serviceName + "|" + panelID;
    return serverConn.queryCommon("QUERY|COMMON_QUERY|querySimpleButtonStruct", param);
  }

  public String getUpdateButtonStruct(String serviceName, String panelID) {
    String param = serviceName + "|" + panelID;
    return serverConn.queryCommon("QUERY|COMMON_QUERY|queryUpdateButtonStruct", param);
  }

  public String getServerQueryParams(String serviceName, String panelID, String switchCode) {
    String param = serviceName + "|" + panelID + "|" + switchCode;
    String ans = serverConn.queryCommon("QUERY|COMMON_QUERY|queryServerQueryParams", param);
    String[] lines = ans.split("\\$");
    String[] tokens = lines[0].split("\\|");
    return tokens[0];
  }

  public String getServerQuerySQL(String serviceName, String panelID, String switchCode) {
    String param = serviceName + "|" + panelID + "|" + switchCode;
    return serverConn.queryCommon("QUERY|COMMON_QUERY|queryServerQuerySQL", param);
  }

  public String getServerDeleteParams(String serviceName, String delCode) {
    String param = serviceName + "|" + delCode;
    String ans = serverConn.queryCommon("QUERY|COMMON_QUERY|queryServerDeleteParams", param);
    String[] lines = ans.split("\\$");
    return lines[0];
  }

  public String getServerDeleteSQL(String serviceName, String delCode) {
    String param = serviceName + "|" + delCode;
    return serverConn.queryCommon("QUERY|COMMON_QUERY|queryServerDeleteSQL", param);
  }

  public String getServerUpdateParams(String serviceName, String updCode) {
    String param = serviceName + "|" + updCode;
    String ans = serverConn.queryCommon("QUERY|COMMON_QUERY|queryServerUpdateParams", param);
    String[] lines = ans.split("\\$");
    return lines[0];
  }

  public String getServerUpdateSQL(String serviceName, String updCode) {
    String param = serviceName + "|" + updCode;
    return serverConn.queryCommon("QUERY|COMMON_QUERY|queryServerUpdateSQL", param);
  }

  public String getServerInsertParams(String serviceName, String insCode) {
    String param = serviceName + "|" + insCode;
    String ans = serverConn.queryCommon("QUERY|COMMON_QUERY|queryServerInsertParams", param);
    String[] lines = ans.split("\\$");
    return lines[0];
  }

  public String getServerInsertSQL(String serviceName, String insCode) {
    String param = serviceName + "|" + insCode;
    return serverConn.queryCommon("QUERY|COMMON_QUERY|queryServerInsertSQL", param);
  }

  public String getServerSpecialParams(String serviceName, String specialCode) {
    String param = serviceName + "|" + specialCode;
    String ans = serverConn.queryCommon("QUERY|COMMON_QUERY|queryServerSpecialParams", param);
    String[] lines = ans.split("\\$");
    return lines[0];
  }

  public String getCommonQueryParams(String queryName) {
    String ans = serverConn.queryCommon("QUERY|COMMON_QUERY|queryCommonQueryParams", queryName);
    String[] lines = ans.split("\\$");
    return lines[0];
  }

  public String getCommonQueryResult(String queryName, String paramValues) {
    String param = queryName + "#" + paramValues;
    return serverConn.queryCommon("QUERY|COMMON_QUERY|queryCommonQueryResult", param);
  }
    

  //*** get data from colorMap, gakumuCodeMap, commonCodeMap ***//

  public Color getColor(String colorName) {
    if (colorMap.containsKey(colorName)) {
      return colorMap.get(colorName);
    } else {
      return Color.red;
    }
  }

  public Color getTabFgColor(String colorName) {    
    if (colorMap.containsKey(colorName)) {
      return colorMap.get(colorName);
    } else {
      return colorMap.get("DarkBlue");
    }
  }

  public Color getTabBgColor(String colorName) {    
    if (colorMap.containsKey(colorName)) {
      return colorMap.get(colorName);
    } else {
      return colorMap.get("Khaki");
    }
  }

  public Color getHeadTitleFgColor(String colorName) {    
    if (colorMap.containsKey(colorName)) {
      return colorMap.get(colorName);
    } else {
      return colorMap.get("DarkGreen");
    }
  }

  public Color getHeadTitleBgColor(String colorName) {    
    if (colorMap.containsKey(colorName)) {
      return colorMap.get(colorName);
    } else {
      return colorMap.get("Honeydew");
    }
  }

  public Color getBottomTitleFgColor(String colorName) {    
    if (colorMap.containsKey(colorName)) {
      return colorMap.get(colorName);
    } else {
      return colorMap.get("DarkGreen");
    }
  }

  public Color getBottomTitleBgColor(String colorName) {    
    if (colorMap.containsKey(colorName)) {
      return colorMap.get(colorName);
    } else {
      return colorMap.get("Ivory");
    }
  }

  public Color getTableFgColor(String colorName) {    
    if (colorMap.containsKey(colorName)) {
      return colorMap.get(colorName);
    } else {
      return colorMap.get("Black");
    }
  }

  public Color getTableBgColor(String colorName) {    
    if (colorMap.containsKey(colorName)) {
      return colorMap.get(colorName);
    } else {
      return colorMap.get("MintCream");
    }
  }

  public Color getSimpleButtonFgColor(String colorName) {    
    if (colorMap.containsKey(colorName)) {
      return colorMap.get(colorName);
    } else {
      return colorMap.get("Black");
    }
  }

  public Color getSimpleButtonBgColor(String colorName) {    
    if (colorMap.containsKey(colorName)) {
      return colorMap.get(colorName);
    } else {
      return colorMap.get("Turquoise");
    }
  }

  public Color getUpdateButtonFgColor(String colorName) {    
    if (colorMap.containsKey(colorName)) {
      return colorMap.get(colorName);
    } else {
      return colorMap.get("Black");
    }
  }

  public Color getUpdateButtonBgColor(String colorName) {    
    if (colorMap.containsKey(colorName)) {
      return colorMap.get(colorName);
    } else {
      return colorMap.get("Aquamarine");
    }
  }

  public int getFontSize(String token) {
    if (token == null) return 12;
    else return Integer.parseInt(token);
  }

  public int getRowHeight(String token) {
    if (token == null) return 20;          // default row height
    else return Integer.parseInt(token);
  }

  public void addCommonCodeMap(String commonCode, String value, String display) {  
    if (commonCode != null) {
      if (value == null) {
	value = " ";
      } 
      if (display == null) {
	display = " ";
      } 
      commonCodeToValueMap.put(commonCode, value);
      commonCodeToDisplayMap.put(commonCode, display);
    }
  }

  public boolean commonCodeMapContains(String commonCode) {
    return commonCodeToValueMap.containsKey(commonCode);
  }

  public String getValueFromCommonCodeMap(String commonCode) {
    if (commonCodeMapContains(commonCode)) {
      return commonCodeToValueMap.get(commonCode);
    } else {
      return null;
    }
  }

  public String getDisplayFromCommonCodeMap(String commonCode) {
    if (commonCodeMapContains(commonCode)) {
      return commonCodeToDisplayMap.get(commonCode);
    } else {
      return null;
    }
  }

  //*** show Popup Message and Popup Notice ***//

  public void showMessage(String msg) {
    Object[] msgs = { msg };
    JOptionPane.showMessageDialog(frame, msgs, 
				  "Warning", JOptionPane.WARNING_MESSAGE);
  }

  public void showMessageLong(String msg) {
    String[] strArray = msg.split("\\$");
    JOptionPane.showMessageDialog(frame, strArray, 
                                  "Warning", JOptionPane.WARNING_MESSAGE);
  }

  public boolean showNotice(String msg) {
    int ans = JOptionPane.showConfirmDialog(frame, msg, "Comfirm", 
                                            JOptionPane.YES_NO_OPTION);
    return (ans == JOptionPane.OK_OPTION);
  }

  public boolean showNoticeLong(String msg) {
    String[] strArray = msg.split("\\$");
    int ans = JOptionPane.showConfirmDialog(frame, strArray,
					    "Comfirm", 
                                            JOptionPane.YES_NO_OPTION);
    return (ans == JOptionPane.OK_OPTION);
  }
    
  public String getDialogInput(String message) {
    return JOptionPane.showInputDialog(frame, message);
  }


  //*** get Info for Curriculum-Table-Print ***//

  public String getInfoForCurriculumTablePrint(String curriculumYear, 
					       String faculty,
					       String department,
					       String course) {

    String key = "QUERY|COMMON_QUERY|queryCurriculumInfoForPrint";
    String paramValues = curriculumYear + "|" + faculty + "|" + department + "|" + course;
    return serverConn.queryCommon(key, paramValues);
  }
} 
