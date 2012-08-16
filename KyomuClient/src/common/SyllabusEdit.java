package common;

import clients.*;
import xml.*;
import java.util.*;
import java.io.*;
import java.net.*;
import javax.swing.*;
import javax.swing.border.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.text.*;
import javax.swing.tree.*;
import javax.swing.event.*;
import javax.xml.parsers.*;
import org.xml.sax.*;
//import org.w3c.dom.*;

public class SyllabusEdit extends JPanel {
  public CommonInfo commonInfo;
  public TabbedPaneBase parentTabbedPane;
  public DataPanelBase  dataPanel;
  public String serviceName;
  public String nodePath;
  public String panelID;

  public String schoolYear;
  public String subjectCode;
  public String teacherCode;
  public String classCode;
  public Font font;

  public DefaultMutableTreeNode root = null;
  public String lineSeparator;
  public boolean docChangeFlag = false;

  public String[][] buttonText = { { "HTML で表示", "showHTML" },
				   { "シラバスを他からコピー", "getOtherXML" },
				   { "編集を破棄/Revert", "readXML" },
				   { "編集内容を保存/Save", "saveXML" } };
/*
  public String[][] buttonText2 = { { "<P>  選択された項目の下に「パラグラフ (テキスト記入用)」を追加", "addParagraph" },
		    { "<OL> 選択された項目の下に空の「番号つきの列挙リスト」を追加", "addEnumerate" },
		    { "<UL> 選択された項目の下に空の「番号なしの列挙リスト」を追加", "addItemize" },
		    { "<LI> 選択された項目の下に「新しい列挙項目 (テキスト記入用)」を追加", "addItem" },
		    { "選択された項目を「削除」する", "removeElement" },
		    { "↓ 選択されている項目を一段下の項目と入替え", "childDown" },
		    { "↑ 選択されている項目を一段上の項目と入替え", "childUp" } };
*/
  
    public String[][] buttonText2 = { { "段落を追加", "addParagraph" },
		    { "番号つきリストを追加", "addEnumerate" },
		    { "番号なしリストを追加", "addItemize" },
		    { "リストに項目を追加", "addItem" },
		    { "選択項目を削除", "removeElement" },
		    { "項目を下へ", "childDown" },
		    { "項目を上へ", "childUp" } };

  public String[] itemText = { "授業の概要", 
			       "授業の位置付け", 
			       "授業項目", 
			       "授業の進め方", 
			       "授業の達成目標",
			       "成績評価の方法と基準",
			       "教科書",
			       "参考書",
			       "キーワード",
			       "備考" };
  
  public JTree     treeView;
  public JTextArea textArea;
  public JButton[] button;
  public JButton[] button2;
  public DefaultTreeModel model;


  public SyllabusEdit(String tableViewType,
		      String serviceName,
		      String nodePath,
		      String panelID,
		      CommonInfo commonInfo, 
		      TabbedPaneBase tabbedPane,
		      DataPanelBase dataPanel) {
    this.serviceName = serviceName;
    this.nodePath = nodePath;
    this.panelID = panelID;
    this.commonInfo = commonInfo;
    this.parentTabbedPane = tabbedPane;
    this.dataPanel = dataPanel;
    this.lineSeparator = commonInfo.lineSeparator;
    font = new Font("DialogInput", Font.PLAIN, 12);
    setSyllabusEdit();
  }

  public void setSyllabusEdit() { 
    setLayout(new BorderLayout());

    Color bgColor, fgColor;
    treeView = new JTree();
    treeView.addTreeSelectionListener(new TreeSelect());
    treeView.addMouseListener(new MyMouseListener());
    treeView.setFont(new Font("DialogInput", Font.PLAIN, 11));
    
    add(new JScrollPane(treeView), BorderLayout.CENTER);

    JPanel panel = new JPanel();
    panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS)); 
    ActionListener listener = new ActionModel();
    button = new JButton[buttonText.length];
    for (int i = 0; i < buttonText.length; i++) {
      panel.add(Box.createRigidArea(new Dimension(2,2)));
      button[i] = new JButton( buttonText[i][0] );
      button[i].addActionListener(listener); 
      if ((i == 0) || (i == 3)) {
	bgColor = Color.pink;
	fgColor = Color.black;
      } else if (i == 2) {
	bgColor = Color.yellow;
	fgColor = Color.red;
      } else {
	bgColor = Color.yellow;
	fgColor = Color.black;
      }
      button[i].setBackground(bgColor);
      button[i].setForeground(fgColor);   
      button[i].setFont(font);  
      button[i].setBorder(new CompoundBorder(BorderFactory.createRaisedBevelBorder(), 
					     new EmptyBorder(0,4,0,4)));
      panel.add(button[i]); 
    }

    panel.add(Box.createRigidArea(new Dimension(4,4)));
    textArea = new JTextArea(5, 36);
    textArea.getDocument().addDocumentListener(new DocChange());
    textArea.setLineWrap(true);   
    textArea.setEditable(true);   
    TitledBorder border1 = new TitledBorder(null, 
					    " 選択された項目のテキストの編集:  Return / Enter を入力しないで下さい ", 
					    TitledBorder.RIGHT, TitledBorder.TOP,
					    new Font("DialogInput", Font.PLAIN, 12) );
    EmptyBorder border2 = new EmptyBorder(4, 4, 4, 4);
    textArea.setBorder(new CompoundBorder(border1, border2));
    textArea.setFont( new Font("DialogInput", Font.PLAIN, 12) );
    JScrollPane jsp = new JScrollPane( textArea );
    panel.add( jsp );
    
    ActionListener listener2 = new ActionModel2();
    button2 = new JButton[buttonText2.length];
    for (int i = 0; i < buttonText2.length; i++) {
      panel.add(Box.createRigidArea(new Dimension(3,3)));
      button2[i] = new JButton( buttonText2[i][0] );
      button2[i].addActionListener(listener2);  
      if ((i == 0) || (i == 3)) {
	bgColor = Color.green;
	fgColor = Color.black;
      } else if (i == 4) {
	bgColor = Color.yellow;
	fgColor = Color.red;
      } else {
	bgColor = new Color(64, 224, 208);
	fgColor = Color.blue;
      }
      button2[i].setBackground(bgColor);
      button2[i].setForeground(fgColor);   
      button2[i].setFont(font);
      button2[i].setBorder(new CompoundBorder(BorderFactory.createRaisedBevelBorder(), 
					     new EmptyBorder(0,4,0,4)));
      panel.add(button2[i]);
    }

    add(panel, BorderLayout.EAST);

    JPanel panel2 = new JPanel();
    panel2.setLayout(new BoxLayout(panel2, BoxLayout.Y_AXIS)); 
    /*
    JTextField commentField = new JTextField(" <講義内容> <講義項目> <進め方> <達成目標> <評価方法> <教科書> 等の大項目は編集することができません。 ");
    commentField.setForeground(Color.blue);
    commentField.setFont(font);
    panel2.add(commentField);

    JTextField commentField2 = new JTextField(" Cut & Paste の方法:  「左ボタンで選択した項目」のテキストは「右ボタンで選択する項目」のテキストのあとに「追加」されます。");
    commentField2.setForeground(Color.blue);
    commentField2.setFont(font);
    panel2.add(commentField2);
*/
    JTextField commentField3 = new JTextField(" 全角のローマ数字 I 〜 X や「半角カタカナ」文字などの EUC に規定されていない文字や記号は、ファイルに格納されたり印刷されたりするとき、「文字化け」が");
    commentField3.setForeground(Color.red);
    commentField3.setFont(font);
    panel2.add(commentField3);

    JTextField commentField4 = new JTextField(" 起きます。半角の '「' と ' 」' は深刻な文字化けの原因となるので絶対に使わないで下さい。   Latex の制御に使われる &, %, <, \\ 等の記号も使用しないで下さい。");
    commentField4.setForeground(Color.red);
    commentField4.setFont(font);
    panel2.add(commentField4);

    add(panel2, BorderLayout.SOUTH);
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
    setTreeView(schoolYear, subjectCode, teacherCode);
    setTextArea();
  }

  public void setTextArea() {
    docChangeFlag = true;
    textArea.setText("");
    docChangeFlag = false;
  }

  public void setTreeView(String schoolYear, String subjectCode, String teacherCode) {
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
    model = new DefaultTreeModel(root);
    treeView.setModel(model);
    for (int i = 0; i < treeView.getRowCount(); i++) {
      treeView.expandRow(i);
    }
    treeView.setCellRenderer(new MyTreeCellRenderer());
  }

  public class ActionModel implements ActionListener {
    public void actionPerformed(ActionEvent e) {     
      Object obj = e.getSource();
      for (int i = 0; i < buttonText.length; i++) {
	if (obj == button[i]) {
	  invokeButtonMethod(buttonText[i][1]);
	  break;
	} 
      }
    }
  } 

  public class ActionModel2 implements ActionListener {
    public void actionPerformed(ActionEvent e) {     
      Object obj = e.getSource();
      for (int i = 0; i < buttonText2.length; i++) {
	if (obj == button2[i]) {
	  invokeButtonMethod(buttonText2[i][1]);
	  break;
	} 
      }
    }
  } 

  public void invokeButtonMethod(String methodName) {
    try {
      getClass().getMethod(methodName, (java.lang.Class[]) null).invoke(this, (java.lang.Object[]) null);
    } catch (Exception ex) {
      commonInfo.showMessageLong("SyllabusEdit: $ invokeButtonMethod is not invoked $ " + methodName);  
    }
  }
  
  public void showHTML() { 
    parentTabbedPane.openTab("syllabusView");
  }

  public void saveXML() {    
    if (!commonInfo.structControlDebugMode) {    
      if ((commonInfo.STAFF_QUALIFICATION.equals("3")) ||
	  (commonInfo.STAFF_QUALIFICATION.equals("4")) ||
	  (commonInfo.STAFF_QUALIFICATION.equals("8")) ||
	  (commonInfo.STAFF_QUALIFICATION.equals("9")) ||
	  (commonInfo.STAFF_CODE.equals(teacherCode))) {
	XMLMakerJT maker = new XMLMakerJT();
	UJTVisitor.traverse(root, maker);
	String text = maker.getText();
	String xmltext = convertTextToSave(text);
	commonInfo.commonInfoMethods.updateSyllabusXml(schoolYear, subjectCode, teacherCode, xmltext);
	commonInfo.unsavedDataExist = false;
	readXML();
      } else {
	commonInfo.showMessage("あなたにはこの教授要目を変更する権限が与えられていません。");
      }
    } else {  
      if ((commonInfo.STAFF_QUALIFICATION_D.equals("3")) ||
	  (commonInfo.STAFF_QUALIFICATION_D.equals("4")) ||
	  (commonInfo.STAFF_QUALIFICATION_D.equals("8")) ||
	  (commonInfo.STAFF_QUALIFICATION_D.equals("9")) ||
	  (commonInfo.STAFF_CODE_D.equals(teacherCode))) {
	XMLMakerJT maker = new XMLMakerJT();
	UJTVisitor.traverse(root, maker);
	String text = maker.getText();
	String xmltext = convertTextToSave(text);
	commonInfo.commonInfoMethods.updateSyllabusXml(schoolYear, subjectCode, teacherCode, xmltext);
	commonInfo.unsavedDataExist = false;
	readXML();
      } else {
	commonInfo.showMessage("あなたにはこの教授要目を変更する権限が与えられていません。");
      }
    }
  }

  public String convertTextToSave(String text) {
    String today = "" + commonInfo.thisYear + "-" + commonInfo.thisMonth + "-" + commonInfo.thisDay;
    String line, word;
    StringBuffer sbuf = new StringBuffer();
    StringTokenizer stk = new StringTokenizer(text, "\n");
    line = stk.nextToken();
    line = "<講義 科目コード=\""+subjectCode+"\" 教官コード=\""+teacherCode+"\" 変更日時=\"" + today + "\" 変更者=\"" + commonInfo.STAFF_NAME + "\">\n";
    sbuf.append(line);
    while (stk.hasMoreTokens()) {
      line = stk.nextToken();
      sbuf.append(line).append(lineSeparator);
    }
    return "<?xml version=\"1.0\" encoding=\"EUC-JP\"?>\n" + sbuf.toString().trim();
  }
   
  public void readXML() {
    setTreeView(schoolYear, subjectCode, teacherCode);
    setTextArea();
    commonInfo.unsavedDataExist = false;
  }
   
  public void getOtherXML() {
    ArrayList<Object> list = new ArrayList<Object>();
    list.add(" 読み込むべき教授要目を指定して下さい");
    list.add(" ");
    JComponent selector = commonInfo.getSelector("SchoolyearSubjectSelector", null, null, null, null, 1);
    list.add(selector);
    int ans =  JOptionPane.showOptionDialog(commonInfo.getFrame(), 
					    list.toArray(), 
					    "教授要目を指定", 
					    JOptionPane.OK_CANCEL_OPTION,
					    JOptionPane.QUESTION_MESSAGE,
					    null,
					    null,
					    null);
    if (ans == JOptionPane.OK_OPTION) {
      String paramValues = commonInfo.getSelectorValue("SchoolyearSubjectSelector",
						       "SCHOOL_YEAR#SUBJECT_CODE#TEACHER_CODE");
      if (paramValues != null) {
	StringTokenizer stk = new StringTokenizer(paramValues, "|");
	String schoolYear = stk.nextToken();
	String subjectCode = stk.nextToken();
	String teacherCode = stk.nextToken();
	setTreeView(schoolYear, subjectCode, teacherCode);
	setTextArea();
	commonInfo.unsavedDataExist = true;
      }
    }
  }

  public void addParagraph() {
    TreePath currentSelection = treeView.getSelectionPath();
    if (currentSelection != null) {
      DefaultMutableTreeNode currentNode = (DefaultMutableTreeNode)(currentSelection.getLastPathComponent());
      MutableTreeNode parent = (MutableTreeNode)(currentNode.getParent());
      if (parent != null) {
	JTNode node = (JTNode)currentNode.getUserObject();
	String tag = node.getName();          
	DefaultMutableTreeNode element = new DefaultMutableTreeNode(new JTElement("P", "", true));
	if (tag.equals(itemText[0]) ||
	    tag.equals(itemText[1]) ||
	    tag.equals(itemText[2]) ||
	    tag.equals(itemText[3]) ||
	    tag.equals(itemText[4]) ||
	    tag.equals(itemText[5]) ||
	    tag.equals(itemText[6]) ||
	    tag.equals(itemText[7]) ||
	    tag.equals(itemText[8]) ||
	    tag.equals(itemText[9])) {	  
	  model.insertNodeInto(element, currentNode, 
			       currentNode.getChildCount());
	  treeView.scrollPathToVisible(new TreePath(element.getPath()));
	  commonInfo.unsavedDataExist = true;
	} else if (tag.equals("P") ||
		   tag.equals("UL") ||
		   tag.equals("OL")) {	  
	  int index = parent.getIndex(currentNode);
	  model.insertNodeInto(element, parent, index + 1);
	  commonInfo.unsavedDataExist = true;
	}
      }
    }
  }

  public void addEnumerate() {
    TreePath currentSelection = treeView.getSelectionPath();
    if (currentSelection != null) {
      DefaultMutableTreeNode currentNode =
	(DefaultMutableTreeNode)(currentSelection.getLastPathComponent());
      MutableTreeNode parent = (MutableTreeNode)(currentNode.getParent());
      if (parent != null) {
	JTNode node = (JTNode)currentNode.getUserObject();
	String tag = node.getName();	
	DefaultMutableTreeNode element =
	  new DefaultMutableTreeNode(new JTElement("OL", "", false));
	if (tag.equals(itemText[0]) ||
	    tag.equals(itemText[1]) ||
	    tag.equals(itemText[2]) ||
	    tag.equals(itemText[3]) ||
	    tag.equals(itemText[4]) ||
	    tag.equals(itemText[5]) ||
	    tag.equals(itemText[6]) ||
	    tag.equals(itemText[7]) ||
	    tag.equals(itemText[9]) || 
	    tag.equals("UL") ||
	    tag.equals("OL")) {	  
	  model.insertNodeInto(element, currentNode, currentNode.getChildCount());
	  treeView.scrollPathToVisible(new TreePath(element.getPath()));  
	  commonInfo.unsavedDataExist = true; 
	} else if (tag.equals("LI") || tag.equals("P")) {
	  int index = parent.getIndex(currentNode);
	  model.insertNodeInto(element, parent, index + 1);
	  commonInfo.unsavedDataExist = true; 
	}
      }
    }
  }

  public void addItemize() {
    TreePath currentSelection = treeView.getSelectionPath();
    if (currentSelection != null) {
      DefaultMutableTreeNode currentNode =
	(DefaultMutableTreeNode)(currentSelection.getLastPathComponent());
      MutableTreeNode parent = (MutableTreeNode)(currentNode.getParent());
      if (parent != null) {
	JTNode node = (JTNode)currentNode.getUserObject();
	String tag = node.getName();          
	DefaultMutableTreeNode element =
	  new DefaultMutableTreeNode(new JTElement("UL", "", false));
          
	if (tag.equals(itemText[0]) ||
	    tag.equals(itemText[1]) ||
	    tag.equals(itemText[2]) ||
	    tag.equals(itemText[3]) ||
	    tag.equals(itemText[4]) ||
	    tag.equals(itemText[5]) ||
	    tag.equals(itemText[6]) ||
	    tag.equals(itemText[7]) ||
	    tag.equals(itemText[9]) || 
	    tag.equals("UL") ||
	    tag.equals("OL")) {	  
	  model.insertNodeInto(element, currentNode, currentNode.getChildCount());
	  treeView.scrollPathToVisible(new TreePath(element.getPath()));  
	  commonInfo.unsavedDataExist = true; 
	} else if (tag.equals("LI") || tag.equals("P")) {
	  int index = parent.getIndex(currentNode);
	  model.insertNodeInto(element, parent, index + 1);
	  commonInfo.unsavedDataExist = true; 
	}
      }
    }
  }

  public void addItem() {
    TreePath currentSelection = treeView.getSelectionPath();
    if (currentSelection != null) {
      DefaultMutableTreeNode currentNode =
	(DefaultMutableTreeNode)(currentSelection.getLastPathComponent());
      MutableTreeNode parent = (MutableTreeNode)(currentNode.getParent());
      if (parent != null) {
	JTNode node = (JTNode)currentNode.getUserObject();
	String tag = node.getName();          
	DefaultMutableTreeNode element =
	  new DefaultMutableTreeNode(new JTElement("LI", "", true));	
	if (tag.equals("UL") ||
	    tag.equals("OL")) {
	  model.insertNodeInto(element, currentNode, 
			       currentNode.getChildCount());
	  treeView.scrollPathToVisible(new TreePath(element.getPath()));
	  commonInfo.unsavedDataExist = true; 
	} else if (tag.equals("LI")) {            
	  int index = parent.getIndex(currentNode);
	  model.insertNodeInto(element, parent, index + 1);
	  commonInfo.unsavedDataExist = true; 
	}
      }
    }
  }

  public void removeElement() {
    TreePath currentSelection = treeView.getSelectionPath();
    if (currentSelection != null) {
      DefaultMutableTreeNode currentNode = (DefaultMutableTreeNode)
	(currentSelection.getLastPathComponent());
      DefaultMutableTreeNode parent =
	(DefaultMutableTreeNode)(currentNode.getParent());
      if (parent != null) {
	JTNode node = (JTNode)currentNode.getUserObject();
	String tag = node.getName();          
	if (node.isJTLeafNode()) {
	  int nchildren = parent.getChildCount();
	  if (nchildren == 1) {
	    JTNode parentnode = (JTNode)parent.getUserObject();
	    String parenttag = parentnode.getName();
	    if (parenttag.equals("OL") ||
		parenttag.equals("UL")) {                
	      model.removeNodeFromParent(parent);
	      commonInfo.unsavedDataExist = true; 
	      return;
	    }
	  }
	} 
	if (tag.equals("UL") || tag.equals("OL") ||
	    node.isJTLeafNode()) {            
	  model.removeNodeFromParent(currentNode);
	  commonInfo.unsavedDataExist = true; 
	}
	return;
      }
    }
  }

  public void childDown() {
    TreePath currentSelection = treeView.getSelectionPath();
    if (currentSelection != null) {
      DefaultMutableTreeNode currentNode =
	(DefaultMutableTreeNode)(currentSelection.getLastPathComponent());
      MutableTreeNode parent = (MutableTreeNode)(currentNode.getParent());
      if (parent != null) {
	JTNode node = (JTNode)currentNode.getUserObject();
	String tag = node.getName();          
	if (tag.equals("UL") || tag.equals("OL") ||
	    tag.equals("LI") || tag.equals("P")) {            
	  int index = parent.getIndex(currentNode);
	  int lastindex = parent.getChildCount();            
	  if (index + 1 < lastindex) {
	    model.removeNodeFromParent(currentNode);
	    model.insertNodeInto(currentNode, parent, index + 1);
	    commonInfo.unsavedDataExist = true; 
	  }
	}
      }
    }
  }

  public void childUp() {
    TreePath currentSelection = treeView.getSelectionPath();
    if (currentSelection != null) {
      DefaultMutableTreeNode currentNode =
	(DefaultMutableTreeNode)(currentSelection.getLastPathComponent());
      MutableTreeNode parent = (MutableTreeNode)(currentNode.getParent());
      if (parent != null) {
	JTNode node = (JTNode)currentNode.getUserObject();
	String tag = node.getName();          
	if (tag.equals("UL") || tag.equals("OL") ||
	    tag.equals("LI") || tag.equals("P")) {            
	  int index = parent.getIndex(currentNode);            
	  if (index != 0) {
	    model.removeNodeFromParent(currentNode);
	    model.insertNodeInto(currentNode, parent, index - 1);
	    commonInfo.unsavedDataExist = true; 
	  }
	}
      }
    }
  }  

  class MyMouseListener extends MouseAdapter {
    public void mouseClicked(MouseEvent e) {
      if (SwingUtilities.isRightMouseButton(e)) {
	TreePath path = treeView.getPathForLocation(e.getX(), e.getY());
	if (path != null) {
	  DefaultMutableTreeNode node = (DefaultMutableTreeNode) path.getLastPathComponent();
	  JTNode jtnode = (JTNode)node.getUserObject();
	  String text = jtnode.getValue();
	  String tag = jtnode.getName();
	  if (jtnode.isJTLeafNode() == false) {
	    return;
	  }
	  if (tag.equals("LI") || tag.equals("P")) {
	    String msg = makeNoticeText(text);
	    boolean ret = commonInfo.showNoticeLong(msg);
	    if (ret) {
	      String textToAdd = textArea.getText();
	      jtnode.setValue(text + textToAdd);
	      model.nodeChanged(node);
	    }
	  }      
	}
      }
    }
  }

  public String makeNoticeText(String text) {
    StringBuffer sbuf = new StringBuffer();
    sbuf.append("右ボタンで選択されたパラグラフ又は列挙項目のテキスト $   「");
    char[] ch = text.toCharArray();
    int cnt = 0;
    for (int i = 0; i < ch.length; i++) {
      sbuf.append(ch[i]);
      cnt++;
      if (!Character.isLetterOrDigit(ch[i])) {
	cnt++;
      } 
      if (cnt > 60) {
	sbuf.append("$   ");
	cnt = 0;
      }
    }
    sbuf.append("」$");    
    sbuf.append("のあとに、編集用エディタに表示されているテキスト $   「");
    ch = textArea.getText().toCharArray();
    cnt = 0;
    for (int i = 0; i < ch.length; i++) {
      sbuf.append(ch[i]);
      cnt++;
      if (!Character.isLetterOrDigit(ch[i])) {
	cnt++;
      } 
      if (cnt > 60) {
	sbuf.append("$   ");
	cnt = 0;
      }
    }
    sbuf.append("」$を追加してもいいですか？");  
    return sbuf.toString();
  }  
	

  class TreeSelect implements TreeSelectionListener {
    public void valueChanged(TreeSelectionEvent e) {
      JButton btn;
      TreePath path = e.getPath();
      DefaultMutableTreeNode node = (DefaultMutableTreeNode)path.getLastPathComponent();
      JTNode jtnode = (JTNode)node.getUserObject();
      docChangeFlag = true;
      textArea.setText(jtnode.getValue());
      docChangeFlag = false;

      String tag = jtnode.getName();
      if (jtnode.isJTLeafNode() == false) {
        textArea.setEnabled(false);
        textArea.setOpaque(false);
      } else {
        textArea.setEnabled(true);
        textArea.setOpaque(true);
      }
      
      // add <P> ボタン
      btn = button2[0];
      if (tag.equals(itemText[0]) ||
	  tag.equals(itemText[1]) ||
	  tag.equals(itemText[2]) ||
	  tag.equals(itemText[3]) ||
	  tag.equals(itemText[4]) ||
	  tag.equals(itemText[5]) ||
	  tag.equals(itemText[6]) ||
	  tag.equals(itemText[7]) ||
	  tag.equals(itemText[8]) ||
	  tag.equals(itemText[9]) || 
	  tag.equals("P") || tag.equals("UL") || tag.equals("OL")) {
        btn.setEnabled(true);
        btn.setOpaque(true);
      } else {
        btn.setEnabled(false);
        btn.setOpaque(false);
      }

      // add <LI> ボタン
      btn = button2[3];
      if (tag.equals("OL") || tag.equals("UL") || tag.equals("LI")) {
        btn.setEnabled(true);
        btn.setOpaque(true);
      } else {
        btn.setEnabled(false);
        btn.setOpaque(false);
      }

      // add <UL> ボタン
      btn = button2[2];
      if (tag.equals(itemText[0]) ||
	  tag.equals(itemText[1]) ||
	  tag.equals(itemText[2]) ||
	  tag.equals(itemText[3]) ||
	  tag.equals(itemText[4]) ||
	  tag.equals(itemText[5]) ||
	  tag.equals(itemText[6]) ||
	  tag.equals(itemText[7]) ||
	  tag.equals(itemText[8]) ||
	  tag.equals(itemText[9]) || 
          tag.equals("P") || tag.equals("UL") || tag.equals("OL") || tag.equals("LI")) {
        btn.setEnabled(true);
        btn.setOpaque(true);
      } else {
        btn.setEnabled(false);
        btn.setOpaque(false);
      }

      // add <OL> ボタン
      btn = button2[1];
      if (tag.equals(itemText[0]) ||
	  tag.equals(itemText[1]) ||
	  tag.equals(itemText[2]) ||
	  tag.equals(itemText[3]) ||
	  tag.equals(itemText[4]) ||
	  tag.equals(itemText[5]) ||
	  tag.equals(itemText[6]) ||
	  tag.equals(itemText[7]) ||
	  tag.equals(itemText[8]) ||
	  tag.equals(itemText[9]) || 
          tag.equals("P") || tag.equals("UL") || tag.equals("OL") || tag.equals("LI")) {
        btn.setEnabled(true);
        btn.setOpaque(true);
      } else {
        btn.setEnabled(false);
        btn.setOpaque(false);
      }

      // remove ボタン
      btn = button2[4];
      if (jtnode.isJTLeafNode() ||
          tag.equals("UL") || tag.equals("OL")) {
        btn.setEnabled(true);
        btn.setOpaque(true);
      } else {
        btn.setEnabled(false);
        btn.setOpaque(false);
      }

      // ↑ ボタン
      btn = button2[6];
      if (tag.equals("LI") || tag.equals("P") ||
          tag.equals("UL") || tag.equals("OL")) {
        btn.setEnabled(true);
        btn.setOpaque(true);
      } else {
        btn.setEnabled(false);
        btn.setOpaque(false);
      }

      // ↓ ボタン
      btn = button2[5];
      if (tag.equals("LI") || tag.equals("P") ||
          tag.equals("UL") || tag.equals("OL")) {
        btn.setEnabled(true);
        btn.setOpaque(true);
      } else {
        btn.setEnabled(false);
        btn.setOpaque(false);
      }
    }
  }

  class MyTreeCellRenderer extends JLabel implements TreeCellRenderer {    
    Icon nodeIcon;
    Icon leafIcon;

    public MyTreeCellRenderer() {
      try {
	URL url = new URL("http://kaoru.gakumu.jimui.kyutech.ac.jp/kyomu/ICON/node.gif");
	nodeIcon = new ImageIcon(url);
	url = new URL("http://kaoru.gakumu.jimui.kyutech.ac.jp/kyomu/ICON/leaf.gif");
	leafIcon = new ImageIcon(url);
      } catch (Exception e) {
	nodeIcon = null; 
	leafIcon = null;
      }
    }

    public Component getTreeCellRendererComponent(JTree tree,
                                                  Object value,
                                                  boolean selected,
                                                  boolean expanded,
                                                  boolean leaf,
                                                  int row,
                                                  boolean hasFocus) {
      if (selected) {
        setOpaque(true);
        setBackground(Color.lightGray);
      }
      if (hasFocus) {
        setOpaque(true);
        setBackground(Color.cyan);
      } else {
        setOpaque(false);
      }

      DefaultMutableTreeNode node = (DefaultMutableTreeNode)value;
      JTNode jtnode = (JTNode)node.getUserObject();      
      if (jtnode.isJTLeafNode()) {
	this.setIcon(leafIcon);
      } else {
        this.setIcon(nodeIcon);
      }
      this.setText(node.toString());
      return this;
    }
  }
        
  class DocChange implements DocumentListener {
    public void insertUpdate(DocumentEvent e) {
      changetext(e);
    }

    public void removeUpdate(DocumentEvent e) {
      changetext(e);
    }

    public void changedUpdate(DocumentEvent e) {
      changetext(e);
    }

    private void changetext(DocumentEvent e) {
      if (!docChangeFlag) {
	commonInfo.unsavedDataExist = true;    
      } 

      TreePath currentSelection = treeView.getSelectionPath();
      if (currentSelection != null) {
        DefaultMutableTreeNode currentNode = (DefaultMutableTreeNode)
          (currentSelection.getLastPathComponent());
        
        JTNode node = (JTNode)currentNode.getUserObject();
        if (node.isJTLeafNode()) {
          javax.swing.text.Document doc = e.getDocument();
          try {
            String str = doc.getText(0, doc.getLength());
            node.setValue(str);
            model.nodeChanged(currentNode);
          } catch (BadLocationException ee) {
          }
        }
      }
    }
  }     
}

