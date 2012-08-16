package common;
import java.util.*;
//import java.net.*;
import javax.swing.*;
//import javax.swing.event.*;
import java.awt.*;
import java.awt.event.*;
//import javax.swing.border.*;
//import java.lang.reflect.*;
import clients.*;

public class JikanwariViewBase extends JPanel {
  private String serviceName;
  private String nodePath;
  private String panelID;

  private CommonInfo commonInfo;
  private TabbedPaneBase tabbedPane;
  private DataPanelBase dataPanel;
  private JikanwariModelBase jikanwariModel;
  private RegistrInfo registrInfo;

  //*** JIKANWARI_VIEW_INFO ***//
  public String methodWhenOpened = null;
  public String rowDisplay;
  public int    rowHeightUnit = 0;
  public int    rowHeight = 0;
  public String queryParamDisplays;

  //*** JIKANWARI_COLUMN_STRUCT ***//
  public ArrayList<Integer> columnNumberList = new ArrayList<Integer>();
  public ArrayList<String> columnTitleList = new ArrayList<String>();
  public ArrayList<String> columnKeyList = new ArrayList<String>();
  public ArrayList<Integer> columnWidthList = new ArrayList<Integer>();
  public ArrayList<String> columnRendererList = new ArrayList<String>();
  public HashMap<String, Integer> columnKeyMap = new HashMap<String, Integer>();
  public ArrayList<String> rowTitleList = new ArrayList<String>();
  public HashMap<String, Integer> rowKeyMap = new HashMap<String, Integer>();

  private String oldParamValues = "abdakabdarah";
  private boolean refreshTableFlag = false;

  private Dimension komaDim, komaDim2, unitDim;
  private Dimension upperLabelDim, leftLabelDim, upperleftLabelDim;
  private Dimension textLabelDim;

  private JPanel[][] koma;
  private JList[][] komaList;

  private MouseListener selectionListener = new SelectionHandler();;
  private Font font;
  private Color[] fgColor;
  private Color[] bgColor;
  private Color   selectedColor;
  private float CENTER_ALIGNMENT = javax.swing.JComponent.CENTER_ALIGNMENT;
 
  public JikanwariViewBase(String serviceName,
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
    this.registrInfo = commonInfo.commonInfoMethods.registrInfo;

    setJikanwariViewInfo();
    setJikanwariColumnInfo();
    setJikanwariRowInfo();

    jikanwariModel = new JikanwariModelBase(serviceName, panelID,
					    commonInfo, this);
    setLabelDimensionsAndColors();
    setEmptyJikanwari();
  }

  private void setJikanwariViewInfo() {
    String jikanwariViewStruct = commonInfo.getJikanwariViewStruct(serviceName, panelID);
    if (jikanwariViewStruct == null) {
      commonInfo.showMessageLong("JikanwariViewStruct is not found $ serviceName = " + serviceName + ", panelID = " + panelID);  
      return;
    } else {
      String[] lines = jikanwariViewStruct.split("\\$");
      String[] tokens = lines[0].split("\\|");
      int size = tokens.length;
      if (size == 3) {
	for (int i = 0; i < 3; i++) {
	  String token = tokens[i].trim();
	  if (token.equals("")) token = null;
	  switch (i) {
	  case 0: 
	    methodWhenOpened = token; break;
	  case 1: 
	    if (token == null) rowDisplay = "SIMPLE";
	    else rowDisplay = token; 
	    break;
	  case 2:  
	    if (token == null) rowHeightUnit = 25;
	    else rowHeightUnit = new Integer(token);
	    rowHeight = rowHeightUnit * 4;	    
	    break;
	  }
	}
      } else {
	commonInfo.showMessageLong("format error in JikanwariViewStruct $ " + jikanwariViewStruct);  
      }
    }
  }
        
  private void setJikanwariColumnInfo() { 
    String jikanwariColumnStruct = commonInfo.getJikanwariColumnStruct(serviceName, panelID);
    if (jikanwariColumnStruct == null) {
      commonInfo.showMessageLong(" JikanwariColumnStruct is not found $ serviceName = " + serviceName + ", panelID = " + panelID);  
      return;
    } else {
      int cnt = 0;
      String[] lines = jikanwariColumnStruct.split("\\$");
      for (String line : lines) {
	String[] tokens = line.split("\\|");
	if (tokens.length == 5) {
	  for (int i = 0; i < 5; i++) {
	    String token = tokens[i].trim();
	    if (token.equals("")) token = null;
	    switch (i) {
	    case 0: 
	      if (token == null) columnNumberList.add(null);
	      else columnNumberList.add(new Integer(token));
	      break;
	    case 1:
	      if (token == null) columnTitleList.add("");
	      else columnTitleList.add(token);
	      break;
	    case 2: 
	      if (token.equals("")) columnKeyList.add(null);
	      else {
		columnKeyList.add(token);
		columnKeyMap.put(token, new Integer(cnt));
	      } 
	      cnt++;
	      break;
	    case 3:
	      if (token == null) columnWidthList.add(120);
	      else columnWidthList.add(new Integer(token));
	      break;
	    case 4:
	      if (token == null) columnRendererList.add(null);
	      else columnRendererList.add(token);
	      break;
	    }
	  }
	} else {
	  commonInfo.showMessageLong(" format error in JikanwariColumnStruct $ " + line);  
	}
      }
    }
  }

  private void setJikanwariRowInfo() {
    String rowQueryParams = commonInfo.getServerQueryParams(serviceName, panelID, "7");
    String paramValues = setQueryParamValues(rowQueryParams);
    String ans = commonInfo.getQueryResult(serviceName, panelID, "7", paramValues);
    
    rowTitleList.add(" ");
    int cnt = 1;
    String[] lines = ans.split("\\$");
    for (String line : lines) {
      String[] tokens = line.split("\\|");
      String key = tokens[0];
      String title = commonInfo.getGakumuCodeShorterName(rowDisplay, key);
      rowTitleList.add(title);
      rowKeyMap.put(key, new Integer(cnt));
      cnt++;
    }
  }

  private String setQueryParamValues(String queryParams) {
    if (queryParams == null) {
      queryParamDisplays = "";
      return "empty";
    } else {
      StringBuilder sbuf1 = new StringBuilder();
      StringBuilder sbuf2 = new StringBuilder();
      String[] tokens = queryParams.split("\\:");
      for (String token : tokens) {
	String columnCode = token;
	String value = "";
	String display = "";
	if (tabbedPane.columnCodeMapContains(columnCode)) {
	  value = tabbedPane.getValueFromColumnCodeMap(columnCode);
	  display = tabbedPane.getDisplayFromColumnCodeMap(columnCode);
	  sbuf1.append(value).append("|");
	  sbuf2.append(display).append(" ");
	} else if (commonInfo.commonCodeMapContains(columnCode)) {
	  value = commonInfo.getValueFromCommonCodeMap(columnCode);
	  display = commonInfo.getDisplayFromCommonCodeMap(columnCode);
	  sbuf1.append(value).append("|");
	  sbuf2.append(display).append(" ");
	} else {
	  if (!commonInfo.structControlDebugMode) {
	    commonInfo.showMessageLong("JikwanrariViewBase: $ " + columnCode + " の値が設定されていません。");
	    return null;
	  } else {  //  structControl のデバッグ時の設定
	    String str = "debugMode: パラメータ " + columnCode + " の値を指定して下さい。";
	    value = commonInfo.getDialogInput(str);
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
	    sbuf1.append(value).append("|");
	    sbuf2.append(value).append(" ");
	  }
	}   
      }
      queryParamDisplays = sbuf2.toString();
      return sbuf1.toString();
    }
  }
 
  public void pageOpened() {
    if (methodWhenOpened != null) {
      invokeMethodWhenOpened(methodWhenOpened);
    }
    
    String dataQueryParams = commonInfo.getServerQueryParams(serviceName, panelID, "8");
    String paramValues = setQueryParamValues(dataQueryParams);    
    if (paramValues != null) {
      if ((!paramValues.equals(oldParamValues)) 
	  || (refreshTableFlag == true)) {
	dataPanel.setTitleText(queryParamDisplays);
	jikanwariModel.makeJikanwariData(paramValues);
	setKomaInfo();	
	oldParamValues = paramValues;
      }
    }
    commonInfo.timerRestart(); 
  } 

  public void refreshTable() { 
    String dataQueryParams = commonInfo.getServerQueryParams(serviceName, panelID, "8");
    String paramValues = setQueryParamValues(dataQueryParams); 
    if (paramValues != null) {
      dataPanel.setTitleText(queryParamDisplays);
      jikanwariModel.makeJikanwariData(paramValues);
      setKomaInfo();	
      oldParamValues = paramValues;
      commonInfo.timerRestart(); 
    }
  }  

  public JikanwariModelBase getJikanwariModel() {
    return jikanwariModel;
  }

  public void setLabelDimensionsAndColors() {
    int unit = rowHeightUnit;
    int columnWidth = columnWidthList.get(1);

    unitDim = new Dimension(unit-5, (int)((rowHeight - 20)/4));
    upperleftLabelDim = new Dimension(unit, unit); 
    upperLabelDim = new Dimension(columnWidth, unit);
    leftLabelDim = new Dimension(unit, rowHeight - 6);
    textLabelDim = new Dimension(columnWidth - 5, (int)(rowHeight/4) - 5);
    komaDim = new Dimension(columnWidth, rowHeight - 2);
    komaDim2 = new Dimension(columnWidth - 20, rowHeight - 6);

    font = new Font("DialogInput", Font.PLAIN, 12);

    fgColor = new Color[10];
    fgColor[0] = commonInfo.getColor("Black");
    fgColor[1] = commonInfo.getColor("IndianRed");
    fgColor[4] = commonInfo.getColor("MidnightBlue");
    fgColor[5] = new Color( 65, 105, 225);
    fgColor[9] = commonInfo.getColor("SeaGreen");  

    bgColor = new Color[5]; 
    bgColor[0] = commonInfo.getTableBgColor("null");
    bgColor[1] = commonInfo.getColor("Wheat");
    bgColor[2] = commonInfo.getColor("Lavender");
    bgColor[3] = commonInfo.getColor("LightCyan");
    bgColor[4] = commonInfo.getColor("DarkSeaGreen");
    selectedColor = Color.yellow;
  }

  public void setEmptyJikanwari() {
    removeAll();
    setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));

    int rowSize = rowTitleList.size();
    int colSize = columnTitleList.size();

    koma = new JPanel[rowSize][colSize];
    komaList = new JList[rowSize][colSize];

    for (int i = 0; i < rowSize; i++) {
      for (int j = 0; j < colSize; j++) {
        koma[i][j] = new JPanel();
        (koma[i][j]).setLayout(new BoxLayout(koma[i][j], BoxLayout.Y_AXIS));
        if ((i == 0) || (j == 0)) {
          (koma[i][j]).setBorder(BorderFactory.createRaisedBevelBorder());
        }
      }
    }

    JPanel panel = new JPanel();
    panel.setLayout(new BoxLayout(panel, BoxLayout.X_AXIS));
    (koma[0][0]).setMaximumSize(upperleftLabelDim);
    (koma[0][0]).setPreferredSize(upperleftLabelDim);
    panel.add(koma[0][0]);

    for (int j = 1; j < colSize; j++) {
      JLabel label = new JLabel();
      label.setFont(font);
      label.setText(jikanwariModel.komaInfo.get(0).get(j).get(0));
      label.setAlignmentX(CENTER_ALIGNMENT);
      label.setMinimumSize(upperLabelDim);
      label.setPreferredSize(upperLabelDim);
      (koma[0][j]).add(label);
      (koma[0][j]).setMaximumSize(upperLabelDim);
      (koma[0][j]).setPreferredSize(upperLabelDim);
      panel.add(koma[0][j]);
    }
    add(panel);

    for (int i = 1; i < rowSize; i++) {
      panel = new JPanel();
      panel.setLayout(new BoxLayout(panel, BoxLayout.X_AXIS));

      String str = jikanwariModel.komaInfo.get(i).get(0).get(0);
      if (str.length() < 4) {
	str = str + "    ";
      }
      char[] text = str.toCharArray();
      (koma[i][0]).add(Box.createRigidArea(new Dimension(4,4)));
      for (int k = 0; k < 4; k++) {
	JLabel label = new JLabel();
	label.setFont(font);
	label.setText("" + text[k]);
	label.setAlignmentX(CENTER_ALIGNMENT);
	label.setMinimumSize(unitDim);
	label.setPreferredSize(unitDim);
	(koma[i][0]).add(label);
      }
      (koma[i][0]).setMaximumSize(leftLabelDim);
      (koma[i][0]).setPreferredSize(leftLabelDim);
      panel.add(koma[i][0]);
 
      for (int j = 1; j < colSize; j++) {
        (koma[i][j]).setMaximumSize(komaDim);
        (koma[i][j]).setPreferredSize(komaDim);
	JList list = new JList();
	list.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
	list.addMouseListener(selectionListener);
	list.setCellRenderer(new KomaRenderer());
	komaList[i][j] = list;
	(koma[i][j]).add(new JScrollPane(list));
        panel.add(koma[i][j]);
      }
      add(panel);
    }
  }

  protected void setKomaInfo() {
    int rowSize = rowTitleList.size();
    int colSize = columnTitleList.size();

    for (int row = 1; row < rowSize; row++) {
      for (int col = 1; col < colSize; col++) {
	komaList[row][col].setListData(jikanwariModel.komaInfo.get(row).get(col).toArray());
      }
    }    
  }


  protected void invokeMethodWhenOpened(String methodName) {
    try {
      getClass().getMethod(methodName, (java.lang.Class[]) null).invoke(this, (java.lang.Object[]) null);
    } catch (Exception ex) {
      commonInfo.showMessageLong("MethodWhenOpened is not invoked $ " + methodName);  
    }
  } 
 
  public void setRefreshTableFlag() {
    refreshTableFlag = true;
  }

  private JPanel makeKomaCell(String subjectInfo, 
			      int count, int index, boolean selected) {
    JPanel panel = new JPanel();
    panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
    panel.setBorder(BorderFactory.createRaisedBevelBorder());
    panel.setMaximumSize(komaDim2);
    panel.setPreferredSize(komaDim2);
    panel.add(Box.createRigidArea(new Dimension(4,4)));

    if (subjectInfo.equals("empty")) {
      for (int k = 0; k < 4; k++) {
	JLabel label = new JLabel();
	label.setText(" ");
	label.setAlignmentX(CENTER_ALIGNMENT);
	label.setMinimumSize(textLabelDim);
	label.setPreferredSize(textLabelDim);
	panel.add(label);
      }
      panel.add(Box.createRigidArea(new Dimension(2,2)));

      if (selected) {
	panel.setBackground(selectedColor);
      } else {
	panel.setBackground(bgColor[0]);
      } 

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
      
      String schoolYear = SCHOOL_YEAR + "年";
      String semester = commonInfo.getGakumuCodeShorterName("SEMESTER", SEMESTER);
      String faculty = commonInfo.getGakumuCodeShorterName("FACULTY", FACULTY);
      String department = commonInfo.getGakumuCodeShorterName("DEPARTMENT", DEPARTMENT);
      String gakunen = commonInfo.getGakumuCodeShorterName("JIKAN_NENJI", GAKUNEN);
      String week = commonInfo.getGakumuCodeShorterName("JIKAN_WEEK", WEEK);
      String hour = HOUR + "時限目";
      String kubunCode = commonInfo.getGakumuCodeShorterName("KUBUN_CODE", KUBUN_CODE);
      String reqCode = commonInfo.getGakumuCodeShorterName("REQ_CODE", REQ_CODE);
      Double obj = new Double(UNIT);
      String unit = obj.doubleValue() + "単位";
      String room = commonInfo.getGakumuCodeShorterName("ROOM", ROOM);
      if (room == null) {
	room = "";
      }
      
      int ind = Integer.parseInt(REQ_CODE);
      Color fg = fgColor[ind];
      
      for (int k = 0; k < 4; k++) {
	JLabel label = new JLabel();
	label.setForeground(fg);
	label.setFont(font);
	if (k == 0) {
	  label.setText(SUBJECT_NAME + " (" + CLASS_CODE + ")");
	} else if (k == 1) {
	  label.setText(TEACHER_NAME);
	} else if (k == 2) {
	  label.setText(kubunCode + " " + unit + " " + reqCode);
	} else if (k == 3) {
	  String str;
	  if (count == 1) {
	    str = room;
	  } else {
	    str = "(" + (index+1) + "/" + count + ")  " + room;
	  }
	  label.setText(str);
	}
	label.setAlignmentX(CENTER_ALIGNMENT);
	label.setMinimumSize(textLabelDim);
	label.setPreferredSize(textLabelDim);
	panel.add(label);
      }
      panel.add(Box.createRigidArea(new Dimension(2,2)));
      if (selected) {
	panel.setBackground(selectedColor);
      } else {    
	panel.setBackground(bgColor[0]);
	if (FLAG.equals("1")) {
	  int sig = registrInfo.getSubjectStatus(SUBJECT_CODE, CLASS_CODE);
	  switch (sig) {
	  case 1:
	    panel.setBackground(bgColor[1]); break; // 修得済み
	  case 2:
	    panel.setBackground(bgColor[3]); break; // 認定済み
	  case 3:
	    panel.setBackground(bgColor[2]); break; // 履修登録
	  case 4:
	    panel.setBackground(bgColor[4]); break; // 仮履修登録
	  case 5:
	    panel.setBackground(bgColor[4]); break; // 仮履修登録
	  }
	}
      }
    }
    return panel;
  }
 
  private JPanel makeEmptyCell(boolean selected) {
    JPanel panel = new JPanel();
    panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
    panel.setBorder(BorderFactory.createRaisedBevelBorder());
    panel.setMaximumSize(komaDim2);
    panel.setPreferredSize(komaDim2);
    if (selected) {
      panel.setBackground(selectedColor);
    } else {
      panel.setBackground(bgColor[0]);
    }
    return panel;
  }

  public void clearSelection() {
    int rowSize = rowTitleList.size();
    int colSize = columnTitleList.size();

    for (int i = 1; i < rowSize; i++) {
      for (int j = 1; j < colSize; j++) {
	komaList[i][j].clearSelection();
      }
    }
  } 

  public void setSubjectInfo(String subjectInfo) {
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

    tabbedPane.addColumnCodeMap("SCHOOL_YEAR", SCHOOL_YEAR, SCHOOL_YEAR);
    tabbedPane.addColumnCodeMap("SUBJECT_CODE", SUBJECT_CODE, SUBJECT_NAME);
    tabbedPane.addColumnCodeMap("TEACHER_CODE", TEACHER_CODE, TEACHER_NAME);
    tabbedPane.addColumnCodeMap("CLASS_CODE", CLASS_CODE, CLASS_CODE);
    tabbedPane.addColumnCodeMap("UNIT", UNIT, UNIT);

    tabbedPane.addColumnCodeMap("EMPTY_KOMA_SELECTED", "false", "false");
  }

  class SelectionHandler extends MouseAdapter {

    public void mousePressed( MouseEvent e ) {
      JList list = (JList) e.getSource();
      int sel = list.getSelectedIndex();
      String subjectInfo = (String) list.getSelectedValue();
      if (!subjectInfo.equals("empty")) {
	setSubjectInfo(subjectInfo);
	clearSelection();
	list.setSelectedIndex(sel);
      } else {
	clearSelection();
	tabbedPane.addColumnCodeMap("EMPTY_KOMA_SELECTED", "true", "true");
	list.setSelectedIndex(sel);
      }

      int rowSize = rowTitleList.size();
      int colSize = columnTitleList.size();
      for (int i = 1; i < rowSize; i++) {
	for (int j = 1; j < colSize; j++) {
	  if (list == komaList[i][j]) {
	    tabbedPane.addColumnCodeMap("ROW_SELECTED", ""+i, ""+i);
	    tabbedPane.addColumnCodeMap("COL_SELECTED", ""+j, ""+j);
	  }
	}
      }
    }
  }

  class KomaRenderer extends JPanel implements ListCellRenderer {

    public Component getListCellRendererComponent(JList jlist, 
						  Object value, 
						  int index, 
						  boolean selected, 
						  boolean hasFocus) {
      String str = (String) value;
      int count = jlist.getModel().getSize();
      if (count == 0) {
	return makeEmptyCell(selected);
      } else {
	return makeKomaCell(str, count, index, selected);
      }
    }
  }

  public void refreshJikanwari() {
    refreshTableFlag = true;
  }

  public void refreshRegistrInfo() {
    registrInfo.setRegistrInfo();
    refreshTableFlag = true;
  }

  public void refreshRegistrJikanwari() {
    refreshRegistrInfo();
    refreshTable();
  }

  public void cancelSelectedRegistr() {
    if (registrInfo.registrAllowedPeriod == false) {      
      commonInfo.showMessage("現在は「履修(修正)申告」できる期間ではありません。");
      return;
    }
    String schoolYear  = tabbedPane.getValueFromColumnCodeMap("SCHOOL_YEAR");
    String subjectCode = tabbedPane.getValueFromColumnCodeMap("SUBJECT_CODE");
    String subjectName = tabbedPane.getDisplayFromColumnCodeMap("SUBJECT_CODE");
    String teacherCode = tabbedPane.getValueFromColumnCodeMap("TEACHER_CODE");
    String teacherName = tabbedPane.getDisplayFromColumnCodeMap("TEACHER_CODE");
    String classCode   = tabbedPane.getValueFromColumnCodeMap("CLASS_CODE");

    String result = tabbedPane.getValueFromColumnCodeMap("EMPTY_KOMA_SELECTED");
    if (result.equals("true")) return;

    int sig = registrInfo.getSubjectStatus(subjectCode, classCode);
    if (sig != 4) {   
      commonInfo.showMessage("「" +subjectName + "」は「仮履修登録」されていません。");
      return;
    }

    int ret = registrInfo.cancelRegistrSubject(schoolYear, subjectCode, classCode,
					       subjectName, teacherName);
    if (ret > 0) {
      refreshRegistrJikanwari();
    }
  }

  public void registrSelectedSubject() {
    if (registrInfo.registrAllowedPeriod == false) {      
      commonInfo.showMessage("現在は「履修(修正)申告」できる期間ではありません。");
      return;
    }
    String schoolYear  = tabbedPane.getValueFromColumnCodeMap("SCHOOL_YEAR");
    String subjectCode = tabbedPane.getValueFromColumnCodeMap("SUBJECT_CODE");
    String subjectName = tabbedPane.getDisplayFromColumnCodeMap("SUBJECT_CODE");
    String teacherCode = tabbedPane.getValueFromColumnCodeMap("TEACHER_CODE");
    String teacherName = tabbedPane.getDisplayFromColumnCodeMap("TEACHER_CODE");
    String classCode   = tabbedPane.getValueFromColumnCodeMap("CLASS_CODE");
    String unit        = tabbedPane.getValueFromColumnCodeMap("UNIT");
    
    String result = tabbedPane.getValueFromColumnCodeMap("EMPTY_KOMA_SELECTED");
    if (result.equals("true")) return;

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
	refreshRegistrJikanwari();
      }
    }
  }


}
