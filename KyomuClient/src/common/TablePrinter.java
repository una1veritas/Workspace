package common;

import clients.*;
import java.util.*;
import java.awt.*;
//import java.awt.event.*;
import javax.swing.*;
//import javax.swing.text.*;
//import javax.swing.border.*;
//import javax.swing.filechooser.*;
//import javax.swing.table.*;
import java.io.*;
import javax.swing.tree.*;
import xml.*;
import javax.xml.parsers.*;
import org.xml.sax.*;
//import org.w3c.dom.*;
import javax.print.*;
import javax.print.attribute.*;
import javax.print.attribute.standard.*;
//import javax.print.event.*;

public class TablePrinter {
  private TableViewBase tableView;
  private TabbedPaneBase tabbedPane;
  private DataPanelBase dataPanel;
  private CommonInfo commonInfo;
  private CommonInfoMethods commonMethods;
  private ServerConnectionBase serverConn;
  private ServerConnectionMethods serverMethods;

  private PrintWriter syllabusErrorLog = null;
  public static JFileChooser syllabusLogChooser;  
  public static JFileChooser chooser;

  private static JCheckBox[] checkBox;
  private static HashMap<String, Boolean> checkBoxMap = new HashMap<String, Boolean>();
  private static String titleTextOriginal = "";

  private JCheckBox counterCheckBox = null;
  private boolean counterSelected = true;

  public static Runtime runtime = null;

  public static boolean warningFlag = false;
  public static boolean latexFlag = false;
  public static boolean pdfFlag = false;
  public static boolean psPrinterFlag = false;

  public static String lineSeparator;
  public static String fileSeparator;
  public static String userDir;
  public static String osName;    

  public static String[] latexType = { "platex (latex2e) ", "jlatex " }; 
  public static String[] printerFont = { "標準フォント ", "小型フォント ", "極小フォント " };  
  public static String[] emptyColumn = { "空白欄(小)", "空白欄(中)", "空白欄(大)" } ;
  public static String[] delimType = { "comma ", "tab ", "colon " }; 

  public static ArrayList<Object> topList = new ArrayList<Object>();
  public static ArrayList<Object> bottomList = new ArrayList<Object>();
  public static ArrayList<Object> delimList = new ArrayList<Object>();

  public static ButtonGroup latexGroup = new ButtonGroup();
  public static JRadioButton[] latexButton = new JRadioButton[latexType.length];
  public static JPanel latexPanel = new JPanel();

  public static ButtonGroup fontGroup = new ButtonGroup();
  public static JRadioButton[] fontButton = new JRadioButton[printerFont.length];
  public static JPanel fontPanel = new JPanel();

  public static ButtonGroup emptyColGroup = new ButtonGroup();
  public static JRadioButton[] emptyColButton = new JRadioButton[emptyColumn.length];
  public static JPanel emptyColPanel = new JPanel();

  public static JTextField emptyColNumTF = new JTextField("0");

  public static ButtonGroup delimGroup = new ButtonGroup();
  public static JRadioButton[] delimButton = new JRadioButton[delimType.length];
  public static JPanel delimPanel = new JPanel();

  public static int paperDir = 0;
  public static String xoffset = null;
  public static String yoffset = null;
  public static boolean printerParamFixed = false;

  public static JTextField titleTextTF;


  public TablePrinter(TableViewBase tableView) { 
    this.tableView = tableView;
    this.commonInfo = tableView.commonInfo;
    this.tabbedPane = tableView.parentTabbedPane;
    this.dataPanel = tableView.parentDataPanel;
    this.serverConn = commonInfo.serverConn;
    this.serverMethods = serverConn.serverConnectionMethods;
    this.commonMethods = commonInfo.commonInfoMethods;

    if (chooser == null) {
      initializeStaticFields();
    }

    if (runtime == null) {
      runtime = Runtime.getRuntime();
    }

    if (counterCheckBox == null) {
      counterCheckBox = new JCheckBox("通し番号");
      counterCheckBox.setFont(new Font("Serif", Font.PLAIN, 11));  
      counterCheckBox.setSelected(true);
    }
      
    if (titleTextTF == null) {
      titleTextTF = new JTextField(64);  
      titleTextTF.setFont(new Font("DialogInput", Font.PLAIN, 11)); 
    }

    initColumnSelection();
  }

  public TablePrinter(CommonInfo commonInfo) { 
    this.commonInfo = commonInfo;
    if (runtime == null) {
      runtime = Runtime.getRuntime();
    }
    if (userDir == null) {
      initializeStaticFields2();
    }
  }

  public void initColumnSelection() {
    if (checkBox == null) {
      checkBox = new JCheckBox[tableView.getColumnCount()];
      for (int i = 0; i < tableView.getColumnCount(); i++) {
	String title = tableView.getColumnName(i);
	checkBoxMap.put(title, new Boolean(true));
      }
      titleTextOriginal = dataPanel.getTitleText().trim();
      titleTextTF.setText(titleTextOriginal);
    }
  }

  public JPanel getColumnSelectionPanel() {
    JPanel panel = new JPanel();
    panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
    panel.setAlignmentX(JPanel.LEFT_ALIGNMENT);   

    JPanel panel2 = new JPanel();
    panel2.setLayout(new BoxLayout(panel2, BoxLayout.X_AXIS));       
    JLabel label = new JLabel("表題 (編集可):");
    label.setFont(new Font("Serif", Font.PLAIN, 11));  
    panel2.add(label);
    String titleText = dataPanel.getTitleText().trim();
    if (!titleText.equals(titleTextOriginal)) {
      titleTextTF.setText(titleText);  
    }
    panel2.add(titleTextTF);
    panel.add(panel2);
    label = new JLabel("出力すべき「欄名」をチェックして下さい。 「データ表の欄を並べる順序」は、データ表の「欄」をマウスでドラッグする");
    label.setFont(new Font("Serif", Font.PLAIN, 11));      
    panel.add(label);
    label = new JLabel("ことにより変更することができ、それによって「印刷されるデータ表」における「欄を並び方」も変更されます。");
    label.setFont(new Font("Serif", Font.PLAIN, 11));      
    panel.add(label);

    panel.add(counterCheckBox); 

    if (columnIsChanged()) {
      checkBox = new JCheckBox[tableView.getColumnCount()];    
      for (int i = 0; i < tableView.getColumnCount(); i++) {
	String title = tableView.getColumnName(i);   
	Boolean obj = checkBoxMap.get(title);
	boolean check;
	if (obj != null) {
	  check = obj.booleanValue();
	} else {
	  check = true;    //
	}
	checkBox[i] = new JCheckBox(title);
	checkBox[i].setFont(new Font("Serif", Font.PLAIN, 11));  
	checkBox[i].setSelected(check);
	panel.add(checkBox[i]);
      }
    } else {  
      for (int i = 0; i < tableView.getColumnCount(); i++) {
	String title = tableView.getColumnName(i);   
	Boolean obj = checkBoxMap.get(title);
	boolean check;
	if (obj != null) {
	  check = obj.booleanValue();
	} else {
	  check = true;    //
	}
	checkBox[i] = new JCheckBox(title);
	checkBox[i].setFont(new Font("Serif", Font.PLAIN, 11));  
	checkBox[i].setSelected(check);
	panel.add(checkBox[i]);
      }
    }
    return panel;
  }

  private boolean columnIsChanged() {
    int c1 = checkBox.length;
    int c2 = tableView.getColumnCount();
    if (c1 != c2) return true; 

    for (int i = 0; i < tableView.getColumnCount(); i++) {
      String title = tableView.getColumnName(i);   
      if (!checkBoxMap.containsKey(title)) return true;
    }
    return false;
  }

  private void initializeStaticFields() {
    String homeDir = System.getProperty("user.home");
    chooser = new JFileChooser(homeDir);

    syllabusLogChooser = new JFileChooser(homeDir);
    syllabusLogChooser.setDialogTitle("シラバス印刷関連のエラーログを格納する");

    lineSeparator = System.getProperty("line.separator");
    fileSeparator = System.getProperty("file.separator");
    osName  = System.getProperty("os.name");
    userDir = System.getProperty("user.dir");

//    topList.add("LaTeX の種類を指定して下さい。");
    latexPanel.setLayout(new BoxLayout(latexPanel, BoxLayout.X_AXIS)); 
    JLabel label = new JLabel("LaTeX の種類を指定:   ");
    label.setFont(new Font("Serif", Font.PLAIN, 11));  
    latexPanel.add(label);
    for (int i = 0; i < latexType.length; i++) {
      latexButton[i] = new JRadioButton(latexType[i]);
      latexButton[i].setFont(new Font("Serif", Font.PLAIN, 11));  
      latexGroup.add(latexButton[i]);
      latexPanel.add(latexButton[i]);
    }
    latexButton[0].setSelected(true);
    topList.add(latexPanel);
      
//    topList.add("「印刷フォント」のサイズを指定して下さい。");
    fontPanel.setLayout(new BoxLayout(fontPanel, BoxLayout.X_AXIS));   
    label = new JLabel("印刷フォントのサイズを指定:   ");
    label.setFont(new Font("Serif", Font.PLAIN, 11));      
    fontPanel.add(label);
    for (int i = 0; i < printerFont.length; i++) {
      fontButton[i] = new JRadioButton(printerFont[i]);
      fontButton[i].setFont(new Font("Serif", Font.PLAIN, 11));  
      fontGroup.add(fontButton[i]);
      fontPanel.add(fontButton[i]);
    }
    fontButton[0].setSelected(true);
    topList.add(fontPanel);

//    bottomList.add("追加する「空白欄」のサイズを指定して下さい。");
    emptyColPanel.setLayout(new BoxLayout(emptyColPanel, BoxLayout.X_AXIS));    
    label = new JLabel("追加する「空白欄」のサイズを指定:  ");
    label.setFont(new Font("Serif", Font.PLAIN, 11));      
    emptyColPanel.add(label);
    for (int i = 0; i < emptyColumn.length; i++) {
      emptyColButton[i] = new JRadioButton(emptyColumn[i]);
      emptyColButton[i].setFont(new Font("Serif", Font.PLAIN, 11));  
      emptyColGroup.add(emptyColButton[i]);
      emptyColPanel.add(emptyColButton[i]);
    }   
    bottomList.add(emptyColPanel);  
    emptyColButton[0].setSelected(true);
    
//    JPanel emptyColNumPanel = new JPanel();
//    emptyColNumPanel.setLayout(new BoxLayout(emptyColPanel, BoxLayout.X_AXIS)); 
//    emptyColNumPanel.add(new JLabel("追加する「空白欄」の個数を指定: "));
//    emptyColNumPanel.add(emptyColNumTF);
//    bottomList.add(emptyColNumPanel);
    
    JPanel emptyColNumPanel = new JPanel();
    emptyColNumPanel.setLayout(new BoxLayout(emptyColNumPanel, BoxLayout.X_AXIS));    
    label = new JLabel("追加する「空白欄」の個数 : ");
    label.setFont(new Font("Serif", Font.PLAIN, 11));    
    emptyColNumPanel.add(label);  
    emptyColNumTF.setFont(new Font("DialogInput", Font.PLAIN, 11)); 
    emptyColNumPanel.add(emptyColNumTF);
    bottomList.add(emptyColNumPanel);

    delimList.add("delimiter (区切り文字) を選択して下さい。");
    delimPanel.setLayout(new BoxLayout(delimPanel, BoxLayout.X_AXIS));  
    for (int i = 0; i < delimType.length; i++) {
      delimButton[i] = new JRadioButton(delimType[i]);
      delimButton[i].setFont(new Font("Serif", Font.PLAIN, 11));  
      delimGroup.add(delimButton[i]);
      delimPanel.add(delimButton[i]);
    }    
    delimButton[0].setSelected(true);
    delimList.add(delimPanel);
  }


  private void initializeStaticFields2() {
    String homeDir = System.getProperty("user.home");
    lineSeparator = System.getProperty("line.separator");
    fileSeparator = System.getProperty("file.separator");
    osName  = System.getProperty("os.name");
    userDir = System.getProperty("user.dir");
  }

  private boolean platexIsSelected() {
    return latexButton[0].isSelected();
  }

  private int getEmptyColumnCount() {
    String text = emptyColNumTF.getText().trim();
    try {
      return Integer.parseInt(text);
    } catch (Exception e) {
      return 0;
    }
  }

  private double getEmptyColumnWidth() {
    if (emptyColButton[0].isSelected()) {
      return 6.0;
    } else if (emptyColButton[1].isSelected()) {
      return 12.0;
    } else {
      return 30.0;
    } 
  }

  private int getFontSelection() {
    if (fontButton[0].isSelected()) {
      return 0;
    } else if (fontButton[1].isSelected()) {
      return 1;
    } else {
      return 2;
    }
  }

  private String getDelimSelection() {
    if (delimButton[0].isSelected()) {
      return ",";
    } else if (delimButton[1].isSelected()) {
      return "\t";
    } else {
      return ":";
    }
  }
  
  public boolean makeLatexSource(StringBuffer sbuf) {     
    String fontText;
    int lineNumber;
    double scale;
    boolean endFlag = false;
    int pageCount = 1;

    ArrayList<Object> list = new ArrayList<Object>(topList);
    JPanel panel = getColumnSelectionPanel();
    list.add(panel);
   
    for (int i = 0; i < bottomList.size(); i++) {
      list.add(bottomList.get(i));
    }
    
    int ans = JOptionPane.showOptionDialog(commonInfo.getFrame(),
					   list.toArray(), 
					   "様々な設定", 
					   JOptionPane.OK_CANCEL_OPTION,
					   JOptionPane.QUESTION_MESSAGE,
					   null,
					   null,
					   null);

    if (ans != JOptionPane.OK_OPTION) {
      return false;
    }
    
    boolean usePlatex = platexIsSelected();
    int fontSel = getFontSelection();
    switch (fontSel) {
    case 0:
      fontText = "\\normalsize";
      lineNumber = 40;
      scale = 1.0;
      break;
    case 1:
      fontText = "\\small";
      lineNumber = 50;
      scale = 0.8;
      break;
    case 2:
      fontText = "\\scriptsize";
      lineNumber = 65;
      scale = 0.6;
      break;
    default:
      fontText = "\\normalsize";
      lineNumber = 40;
      scale = 1.0;
      break;
    }       

    int emptyColCount = getEmptyColumnCount();
    double emptyColWidth = scale * getEmptyColumnWidth();

    if (counterCheckBox.isSelected()) {
      counterSelected = true;
    } else {
      counterSelected = false;
    }

    if (usePlatex) {
      sbuf.append("\\documentclass[11pt]{jarticle}").append(lineSeparator); 
    } else {
      sbuf.append("\\documentstyle[11pt]{jarticle}").append(lineSeparator); 
    }
    sbuf.append("\\setlength{\\oddsidemargin}{-10pt}").append(lineSeparator); 
    sbuf.append("\\setlength{\\topmargin}{-30pt}").append(lineSeparator); 
    sbuf.append("\\setlength{\\textwidth}{500pt}").append(lineSeparator); 
    sbuf.append("\\setlength{\\textheight}{720pt}").append(lineSeparator); 
    sbuf.append("\\begin{document}").append(lineSeparator); 
    sbuf.append("\\pagestyle{empty}").append(lineSeparator); 
    sbuf.append(fontText).append(lineSeparator); 

    int lineCount = 0;
    int colCount;

    while (!endFlag) {
      sbuf.append("\\begin{table}[t]").append(lineSeparator); 
      sbuf.append(fontText).append(lineSeparator);  //
      StringBuffer sbuf2 = new StringBuffer();

      if (counterSelected) {
	sbuf.append("\\begin{tabular}[t]{|r|");
	sbuf2.append("\\hspace{3mm}");
	colCount = 1;
      } else {
	sbuf.append("\\begin{tabular}[t]{|");
	colCount = 0;
      }
      
      for (int i = 0; i < tableView.getColumnCount(); i++) { 

	String title = tableView.getColumnName(i);     

	if (checkBox[i].isSelected()) {
	  checkBoxMap.put(title, new Boolean(true));  //
	  sbuf.append("r|");
	  if (colCount == 0) {
	    sbuf2.append(trimColumnName(tableView.getColumnName(i)));
	  } else {
	    sbuf2.append(" & ").append(trimColumnName(tableView.getColumnName(i)));
	  }
	  colCount++;
	} else {
	  checkBoxMap.put(title, new Boolean(false));  //
	}
      }

      for (int i = 0; i < emptyColCount; i++) {
	sbuf.append("r|");
	sbuf2.append(" & \\hspace{"+emptyColWidth+"mm}");
	colCount++;
      }	
      sbuf.append("} \\hline").append(lineSeparator); 
      sbuf2.append(" \\\\ \\hline\\hline").append(lineSeparator); 
      
      sbuf.append("\\multicolumn{").append(""+colCount).append("}{|c|}{ ");
      String titleText = titleTextTF.getText().trim();
      sbuf.append(titleText);
      String dateText = ""+commonInfo.thisYear+":"+commonInfo.thisMonth+":"+commonInfo.thisDay;
      sbuf.append(" \\hfill ").append(dateText);
      sbuf.append(" (").append(""+pageCount).append(") } \\\\ \\hline\\hline").append(lineSeparator);
      pageCount++;
      sbuf.append(sbuf2.toString());

      for (int j = 0; j < lineNumber; j++) { 
	if (lineCount < tableView.getRowCount()) {
	  int localColCount = 0;
	  if (counterSelected) {
	    sbuf.append(""+(lineCount+1));
	    localColCount++;
	  }
	  for (int i = 0; i < tableView.getColumnCount(); i++) {
	    if (checkBox[i].isSelected()) {
	      CellObject cobj = (CellObject)tableView.getValueAt(lineCount, i);
	      if (localColCount == 0) {
		sbuf.append(cobj.getDisplay());
	      } else {
		sbuf.append(" & ").append(cobj.getDisplay());
	      }
	      localColCount++;
	    }
	  }
	  for (int i = 0; i < emptyColCount; i++) {
	    sbuf.append(" & ");
	  }
	} else {
	  endFlag = true;
	  for (int i = 0; i < (colCount-1); i++) {
	    sbuf.append(" & ");
	  }	  
	}
	lineCount++;
	if ((lineCount % 10) == 0) {
	  sbuf.append("\\\\ \\hline\\hline ").append(lineSeparator); 
	} else {
	  sbuf.append("\\\\ \\hline ").append(lineSeparator); 
	}
      }
      sbuf.append("\\end{tabular}").append(lineSeparator);
      sbuf.append("\\end{table}").append(lineSeparator);
    }
    sbuf.append("\\end{document}").append(lineSeparator); 
    return true;
  }

  
  public boolean makeAttendLatexSource(StringBuffer sbuf) {    
    boolean endFlag = false;
    int pageCount = 1;    
    boolean usePlatex = true;
    String fontText = "\\scriptsize";
    int lineNumber = 65;
    double scale = 0.6;

    if (usePlatex) {
      sbuf.append("\\documentclass[11pt]{jarticle}").append(lineSeparator); 
    } else {
      sbuf.append("\\documentstyle[11pt]{jarticle}").append(lineSeparator); 
    }
    sbuf.append("\\setlength{\\oddsidemargin}{-25pt}").append(lineSeparator); 
    sbuf.append("\\setlength{\\topmargin}{-40pt}").append(lineSeparator); 
    sbuf.append("\\setlength{\\textwidth}{500pt}").append(lineSeparator); 
    sbuf.append("\\setlength{\\textheight}{720pt}").append(lineSeparator); 
    sbuf.append("\\begin{document}").append(lineSeparator); 
    sbuf.append("\\pagestyle{empty}").append(lineSeparator); 
    sbuf.append(fontText).append(lineSeparator); 

    int lineCount = 0;
    int colCount;

    boolean counterSelected = false;


    checkBox = new JCheckBox[tableView.getColumnCount()];
    for (int i = 0; i < tableView.getColumnCount(); i++) {
      String title = tableView.getColumnName(i);
      checkBoxMap.put(title, new Boolean(true));
    }
    titleTextOriginal = dataPanel.getTitleText().trim();
    titleTextTF.setText(titleTextOriginal);
    
    
    checkBox = new JCheckBox[tableView.getColumnCount()];  
    checkBox[0] = new JCheckBox(tableView.getColumnName(0), true);
    checkBox[1] = new JCheckBox(tableView.getColumnName(1), true);
    checkBox[2] = new JCheckBox(tableView.getColumnName(2), false);
    checkBox[3] = new JCheckBox(tableView.getColumnName(3), false);
    checkBox[4] = new JCheckBox(tableView.getColumnName(4), false);
    checkBox[5] = new JCheckBox(tableView.getColumnName(5), false);
    checkBox[6] = new JCheckBox(tableView.getColumnName(6), false);
    for (int i = 0; i < 14; i++) {
      checkBox[7+i] = new JCheckBox(tableView.getColumnName(7+i), true);
    }

    while (!endFlag) {
      sbuf.append("\\begin{table}[t]").append(lineSeparator); 
      sbuf.append(fontText).append(lineSeparator);  //
      StringBuffer sbuf2 = new StringBuffer();
      
      sbuf.append("\\begin{tabular}[t]{|");
      colCount = 0;

      for (int i = 0; i < tableView.getColumnCount(); i++) {
	if (checkBox[i].isSelected()) {
	  sbuf.append("r|");
	  if (colCount == 0) {
	    sbuf2.append(trimColumnName(tableView.getColumnName(i)));
	  } else {
	    sbuf2.append("&").append(trimColumnName(tableView.getColumnName(i)));
	  }
	  colCount++;
	}
      }

      sbuf.append("} \\hline").append(lineSeparator); 
      sbuf2.append(" \\\\ \\hline\\hline").append(lineSeparator); 
      
      sbuf.append("\\multicolumn{").append(""+colCount).append("}{|c|}{ ");
      String titleText = titleTextTF.getText().trim();
      sbuf.append(titleText);
      String dateText = ""+commonInfo.thisYear+":"+commonInfo.thisMonth+":"+commonInfo.thisDay;
      sbuf.append(" \\hfill ").append(dateText);
      sbuf.append(" (").append(""+pageCount).append(") } \\\\ \\hline\\hline").append(lineSeparator);
      pageCount++;
      sbuf.append(sbuf2.toString());

      for (int j = 0; j < lineNumber; j++) { 
	if (lineCount < tableView.getRowCount()) {
	  int localColCount = 0;
	  if (counterSelected) {
	    sbuf.append(""+(lineCount+1));
	    localColCount++;
	  }
	  for (int i = 0; i < tableView.getColumnCount(); i++) {
	    if (checkBox[i].isSelected()) {
	      CellObject cobj = (CellObject)tableView.getValueAt(lineCount, i);
	      if (localColCount == 0) {
		if (i == 1) {
		  String studentName = cobj.getDisplay();
		  if (studentName.length() > 5) {
		    studentName = studentName.substring(0, 5);
		  }
		  sbuf.append(studentName);
		} else {
		  sbuf.append(cobj.getDisplay());
		}
	      } else {
		if (i == 1) {
		  String studentName = cobj.getDisplay();
		  if (studentName.length() > 5) {
		    studentName = studentName.substring(0, 5);
		  }
		  sbuf.append("&").append(studentName);
		} else {
		  sbuf.append("&").append(cobj.getDisplay());
		}
	      }
	      localColCount++;
	    }
	  }
	} else {
	  endFlag = true;
	  for (int i = 0; i < (colCount-1); i++) {
	    sbuf.append(" & ");
	  }	  
	}
	lineCount++;
	if ((lineCount % 10) == 0) {
	  sbuf.append("\\\\ \\hline\\hline ").append(lineSeparator); 
	} else {
	  sbuf.append("\\\\ \\hline ").append(lineSeparator); 
	}
      }
      sbuf.append("\\end{tabular}").append(lineSeparator);
      sbuf.append("\\end{table}").append(lineSeparator);
    }
    sbuf.append("\\end{document}").append(lineSeparator); 
    return true;
  }


  public void printPsFile(String psfname) {
    if (commonInfo.selectedPsPrintService != null) {
      try {
	FileInputStream stream = new FileInputStream(psfname);
	DocFlavor flavor = DocFlavor.INPUT_STREAM.POSTSCRIPT;
	DocAttributeSet das = new HashDocAttributeSet();
	das.add(MediaSizeName.ISO_A4);
	Doc doc = new SimpleDoc(stream, flavor, das);
	DocPrintJob docPrintJob = commonInfo.selectedPsPrintService.createPrintJob();       
	docPrintJob.print(doc, null);
	
	File file = new File(psfname);
	if (file.exists()) {
	  file.delete();
	}    
      } catch (PrintException e) {
	e.printStackTrace();
      } catch (FileNotFoundException fe) {
	fe.printStackTrace();
      }
    }
  }

  public void setPsPrinter() { 
    if (commonInfo.selectedPsPrintService == null) {
      JList jlist = new JList(commonInfo.psPrintServiceName);
      jlist.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
      int ind = commonInfo.psPrintServiceName.length;
      jlist.setSelectedIndex(ind - 1);
      
      ArrayList<Object> list = new ArrayList<Object>();
      list.add("下記のリストの中から出力用の ps プリンタを ");
      list.add("選択して下さい。 ");
      list.add(jlist);
            
      int ans = JOptionPane.showOptionDialog(commonInfo.getFrame(), 
					     list.toArray(), 
					     "ps プリンタの指定", 
					     JOptionPane.OK_CANCEL_OPTION,
					     JOptionPane.QUESTION_MESSAGE,
					     null,
					     null,
					     null);
      
      if (ans == JOptionPane.OK_OPTION) {
	int index = jlist.getSelectedIndex();
	if (index != -1) {
	  commonInfo.selectedPsPrintService = commonInfo.psPrintService[index];
	  commonInfo.selectedPsPrintServiceName = commonInfo.psPrintServiceName[index];
	}
      }
    }
  }


  public boolean getPsPrinterParam() {     
    String printerParamFile = userDir + fileSeparator + "PRINTER-PARAM.TXT";
    try {
      BufferedReader fin = new BufferedReader(new FileReader(printerParamFile));
      String line = fin.readLine();
      StringTokenizer stk = new StringTokenizer(line, "|");
      String dir = stk.nextToken();
      paperDir = Integer.parseInt(dir);
      xoffset = stk.nextToken();
      yoffset = stk.nextToken();      
      fin.close();
      return true;
    } catch (Exception e) { }
    return false;
  }
   
  public void setPrinterParam() {
    String printerParamFile = userDir + fileSeparator + "PRINTER-PARAM.TXT";
    try {
      PrintWriter fout = new PrintWriter(new BufferedWriter(new FileWriter(printerParamFile)));
      String param = "" + paperDir + "|" + xoffset + "|" + yoffset;
      fout.println(param);
      fout.close();
    } catch (Exception e) {
      commonInfo.showMessage(e.toString());
    }
  }
 
  public boolean showPsPrinterParamDialog() {
    setPsPrinter();
    if (commonInfo.selectedPsPrintService == null) {
      return false;
    }

    boolean res = getPsPrinterParam();
    if (!res) {
      paperDir = 0;
      xoffset = "0mm";
      yoffset = "0mm";
      setPrinterParam();
    }
    
    if (!printerParamFixed) {
      String printerName = commonInfo.selectedPsPrintServiceName;

      JPanel paperDirPanel = new JPanel();
      paperDirPanel.setLayout(new BoxLayout(paperDirPanel, BoxLayout.X_AXIS));       
      String[] paperDirText = { "縦置き ", "横置き " }; 
      ButtonGroup paperDirGroup = new ButtonGroup();
      JRadioButton[] paperDirButton = new JRadioButton[2];  
      paperDirPanel.add(new JLabel("  用紙の方向： ", SwingConstants.LEFT));
      for (int i = 0; i < 2; i++) {
	paperDirButton[i] = new JRadioButton(paperDirText[i]);
	paperDirGroup.add(paperDirButton[i]);
	paperDirPanel.add(paperDirButton[i]);
      }
      paperDirButton[paperDir].setSelected(true);

      ArrayList<Object> list = new ArrayList<Object>();
      list.add("プリンタの設定を確認し、必要に応じて設定を変更 ");
      list.add("して下さい。");
      list.add("  プリンタ名： " + printerName );
      list.add(paperDirPanel);
      JPanel xoffsetPanel = new JPanel();
      xoffsetPanel.setLayout(new BoxLayout(xoffsetPanel, BoxLayout.X_AXIS));   
      xoffsetPanel.add(new JLabel("  x方向のオフセット： ", SwingConstants.LEFT));
      JTextField xoffsetTF = new JTextField(16);
      xoffsetTF.setText(xoffset);
      xoffsetPanel.add(xoffsetTF);
      list.add(xoffsetPanel);
      JPanel yoffsetPanel = new JPanel();
      yoffsetPanel.setLayout(new BoxLayout(yoffsetPanel, BoxLayout.X_AXIS));   
      yoffsetPanel.add(new JLabel("  y方向のオフセット： ", SwingConstants.LEFT));     
      JTextField yoffsetTF = new JTextField(16);
      yoffsetTF.setText(yoffset);
      yoffsetPanel.add(yoffsetTF);
      list.add(yoffsetPanel);

      Object[] options = { "この設定でOK", "試し印刷", "CANCEL"};
      int ans = JOptionPane.showOptionDialog(commonInfo.getFrame(), 
					     list.toArray(), 
                                             "認証画面", 
                                             JOptionPane.YES_NO_CANCEL_OPTION,
                                             JOptionPane.QUESTION_MESSAGE,
                                             null,
                                             options,
                                             options[0]);
      if ((ans == 0) || (ans == 1)) {
	if (paperDirButton[0].isSelected()) {
	  paperDir = 0;
	} else {
	  paperDir = 1;
	}
	xoffset = xoffsetTF.getText().trim();
	yoffset = yoffsetTF.getText().trim();
	setPrinterParam();
	
	if (ans == 0) {
	  printerParamFixed = true;
	} else {
	  printerParamFixed = false;
	}	
	return true;
      } else {
	return false;
      }
    }
    return true;
  }   

  public void latexToPrinter(StringBuffer sbuf) {
    commonInfo.timerRestart();
    showInitialPrinterWarning();

    if (psPrinterFlag) {

      boolean res = showPsPrinterParamDialog();
      if (!res) return;

      String fname = "KYOMU_TABLE_PS";
      String texName = fname + ".tex";  
      String dviName = fname + ".dvi";
      String auxName = fname + ".aux";
      String logName = fname + ".log";
      String psName  = fname + ".ps";
      
      try {
	File f = new File(texName);
	PrintWriter fout = new PrintWriter(new BufferedWriter(new FileWriter(f)));
	fout.println(sbuf.toString());
	fout.close();	 

	ArrayList<String> opts = new ArrayList<String>();
	opts.add("platex");
	opts.add(fname);
	Process process = new ProcessBuilder(opts).redirectErrorStream(true).start();
	InputStream in = process.getInputStream();
	int ch;
	while ((ch = in.read()) != -1) { }  
	process.waitFor();
	process.destroy();

	opts = new ArrayList<String>();
	opts.add("dvipsk");
	if (paperDir == 1) {
	  opts.add("-t");
	  opts.add("landscape");
	}
	opts.add("-O");
	opts.add(xoffset + "," + yoffset);
	opts.add(fname);
	Process process2 = new ProcessBuilder(opts).redirectErrorStream(true).start();
	in = process2.getInputStream();
	while ((ch = in.read()) != -1) { }
	process.waitFor();
	process.destroy();
	
	File file = new File( texName );
	if (file.exists()) {
	  file.delete();
	} 
	file = new File( dviName );
	if (file.exists()) {
	  file.delete();
	} 
	file = new File( auxName );
	if (file.exists()) {
	  file.delete();
	} 
	file = new File( logName );
	if (file.exists()) {
	  file.delete();
	} 
	
	printPsFile(psName);

      } catch (Exception e) {
	commonInfo.showMessage(e.toString());
	return;
      }	 
    } else if (pdfFlag) {

      ArrayList<Object> list = new ArrayList<Object>();
      list.add(" 出力用の pdf ファイル名を設定して下さい。 ");
      JPanel pdfPanel = new JPanel();
      pdfPanel.setLayout(new BoxLayout(pdfPanel, BoxLayout.X_AXIS)); 
      pdfPanel.add(new JLabel("pdfファイル名: ", SwingConstants.LEFT));
      JTextField pdfFileNameTF = new JTextField("  ");
      pdfPanel.add( pdfFileNameTF );
      list.add( pdfFileNameTF );
      list.add(" ファイル名が XXX と指定された場合、ユーザディレクトリ " );
      list.add(" (教務情報システムを起動したディレクトリ) にファイル名  " );
      list.add("  XXX.pdf のPDFファイルが作成されます。 " );
      list.add(" ファイル名が指定されないと KYOMU_TABLE.pdf が " );
      list.add(" 作成されます。 ");
      
      int ans = JOptionPane.showOptionDialog(commonInfo.getFrame(), 
					     list.toArray(), 
					     "pdfファイルの指定", 
					     JOptionPane.OK_CANCEL_OPTION,
					     JOptionPane.QUESTION_MESSAGE,
					     null,
					     null,
					     null);
      
      if (ans == JOptionPane.OK_OPTION) {
	String fname = pdfFileNameTF.getText().trim();
	if (fname.equals("")) {
	  fname = "KYOMU_TABLE";
	}
	String texName = fname + ".tex";  
	String dviName = fname + ".dvi";
	String auxName = fname + ".aux";
	String logName = fname + ".log";

	try {
	  PrintWriter fout = new PrintWriter(new BufferedWriter(new FileWriter(texName)));
	  fout.println(sbuf.toString());
	  fout.close();	 
	  
	  Process process = runtime.exec( "platex " + fname );
	  InputStream in = process.getInputStream();
	  int ch;
	  while ((ch = in.read()) != -1) { }  
	  process.waitFor();
	  process.destroy();

	  process = runtime.exec( "dvipdfm " + fname );
	  in = process.getInputStream();
	  while ((ch = in.read()) != -1) { }
	  process.waitFor();
	  process.destroy();
	  
	  File file = new File( texName );
	  if (file.exists()) {
	    file.delete();
	  } 
	  file = new File( dviName );
	  if (file.exists()) {
	    file.delete();
	  } 
	  file = new File( auxName );
	  if (file.exists()) {
	    file.delete();
	  } 
	  file = new File( logName );
	  if (file.exists()) {
	    file.delete();
	  } 

	} catch (Exception e) {
	  commonInfo.showMessage(e.toString());
	  return;
	}	
      } else {
	return;
      }
    } else {

      File f = null;
      String path;      
      try {
	f = File.createTempFile("KYOMU", ".tex", new File("/tmp"));
      } catch (Exception e) {
	commonInfo.showMessage(e.toString());
	return;
      }
      path = f.getPath();
      String fname = path.substring(0, path.indexOf(".tex"));
      String command = "latex-to-lpr " + fname;  
      
      try {
	PrintWriter fout = new PrintWriter(new BufferedWriter(new FileWriter(f)));
	fout.println(sbuf.toString());
	fout.close();

	Process process = runtime.exec(command);
	InputStream in = process.getInputStream();
	int ch;
	while ((ch = in.read()) != -1) { }  
	process.waitFor();
	process.destroy();
 
      } catch (Exception e) {
	commonInfo.showMessage(e.toString());
	return;
      }
    }
  }

  public String trimColumnName(String str) {
    int ind = str.indexOf("#");
    if (ind > 0) {
      return str.substring(0, ind);
    } else {
      return str;
    }
  }
  
  public void showLatexTypeSelector() {
    if (!latexFlag) {
      commonInfo.timerRestart();
      ArrayList<Object> list = new ArrayList<Object>();
      list.add(" 使用する latex の種類を指定して下さい。 ");
      list.add( latexPanel );     
      list.add(" "); 
      int ans = JOptionPane.showOptionDialog(commonInfo.getFrame(), 
					     list.toArray(), 
					     "Latex設定", 
					     JOptionPane.OK_CANCEL_OPTION,
					     JOptionPane.QUESTION_MESSAGE,
					     null,
					     null,
					     null);
      
      if (ans == JOptionPane.OK_OPTION) { 	
	latexFlag = true;
      }
    }
  }

  public void showPDFWarning() {
    Object[] warning = { 
      " データ表の内容を印刷するための PDF ファイルを作成します。",
      "  ",
      "  前提条件：",
      "   0.  latex のタイプとして platex (LaTeX2e) が選択されていること。 ",
      "   1.  platex  がインストールされていること。 ",
      "   2.  dvipdfm がインストールされていること。",
      "  ",
      "  出力：  ",
      "    ユーザが指定するファイル名を持つ pdf ファイルがユーザディレクトリに ",
      "    生成されます。",
      "    ユーザディレクトリは、このツールを起動したディレクトリです。",
      "  ",
      "  pdf ファイルをプリンタへに出力するには Acrobat 等を利用して下さい。",
      "  ",
      "  システムが実行する作業の手順: ",
      "    (1) ユーザが「出力用のファイル名 XXX 」を入力する。 ",
      "    (2) latex のソースファイル XXX.tex を作成する。 ",
      "    (3) platex XXX  ( XXX.tex ==> XXX.dvi, XXX.aux, XXX.log  )",
      "    (4) dvipdfm XXX ( XXX.dvi ==> XXX.pdf )",
      "    (5) XXX.tex, XXX.dvi, XXX.aux, XXX.log を削除する。 ",
      "  ",
      "  platex と dvipdfm をインストールする方法が分からない場合には、学務係から ",
      "  インストール用の CD を借りてインストールして下さい。 ",
      "  "
      };
    
    int ans = JOptionPane.showOptionDialog(commonInfo.getFrame(), 
					   warning, 
					   "印刷上の注意", 
					   JOptionPane.OK_CANCEL_OPTION,
					   JOptionPane.QUESTION_MESSAGE,
					   null,
					   null,
					   null);
    if (ans == JOptionPane.OK_OPTION) {
      warningFlag = true;
    } else {
      return;
    }
  }

  public void printTable() {
    showInitialPrinterWarning();
    StringBuffer sbuf = new StringBuffer();
    boolean res = makeLatexSource(sbuf);
    if (res) {
      latexToPrinter(sbuf);
    }
  }

  public void printAttendTable() {   
    showInitialPrinterWarning();
    StringBuffer sbuf = new StringBuffer();
    boolean res = makeAttendLatexSource(sbuf);
    if (res) {
      latexToPrinter(sbuf);
    }
  }


  public void showInitialPrinterWarning() {    
    if (!warningFlag) {
      String osName = System.getProperty("os.name");
      if (osName.toLowerCase().startsWith("window")) {     
	showWindowsPrinterWarning();
	if (pdfFlag) {
	  showPDFWarning();
	}
      } else {
	showUnixPrinterWarning();
	if (pdfFlag) {
	  showPDFWarning();
	}
      }
    }
  }

  public void showUnixPrinterWarning() {

    Object[] warning = { 
      "  このプリント機能は UNIX 系 (Solaris or Linux) の標準的プリント環境を  ",
      "  前提として設計されています。",
      "  ",
      "  利用しているマシンに「データ表の Latex のソースファイルをコンパイルして得ら ",
      "  れる ps ファイル等を直接的にプリンタに出力するための機能」が備わっていない ",
      "  場合には「取消しl」ボタンを押して下さい。       ",
      "  ",
      "  その場合には、「印刷ボタン」が押されると「データ表のPDFファイル」が作成される ",
      "  ように設定されます。",
      "  ",
      "  ",
      "  「了解」ボタンを押した場合、システムは次の作業手順を実行します：",
      "1.  作業用ディレクトリ /tmp の下に (ランダムなファイル名) XXXXXX.tex の  ",
      "    Latex ソースファイルを作成する。 ",
      "2.  フロントエンド計算機上で次のコマンドを実行する。 ",
      "       latex-to-lpr  XXXXXX  ",
      "  ",
      "  latex-to-lpr は、フロントエンドマシンのプリンタ関連の環境に応じて各々のユーザが ",
      "  用意すべき「実行スクリプト」であり、概ね、次のような内容を含む必要があります。 ",
      "  ",
      "      #!/bin/sh ",
      "      cd  /tmp ",
      "      jlatex  $1.tex ",
      "      dvi2ps  $1.dvi  |  lpr  -Pjlp ",
      "      rm  $1.* ",
      "  ",
      "  platex (LaTeX2e) を選択した場合、スクリプトの jlatex は platex で置き換えること。 ",
    };
      
    int ans = JOptionPane.showOptionDialog(commonInfo.getFrame(), 
					   warning, 
					   "印刷上の注意", 
					   JOptionPane.OK_CANCEL_OPTION,
					   JOptionPane.QUESTION_MESSAGE,
					   null,
					   null,
					   null);
    if (ans == JOptionPane.OK_OPTION) {
      warningFlag = true;
      pdfFlag = false;
    } else {
      pdfFlag = true;
    }
  }

  public void showWindowsPrinterWarning() {

    Object[] warning = { 
      "  このプリント機能は、Windows 系のマシンの標準的なプリント環境を前提として、  ",
      "  データ表の内容を直接的に ps プリンタに出力するように設計されています。",
      "  ",
      "  利用しているマシンに下記の「データ表の Latex のソースファイルをコンパイル ",
      "  して得られる ps ファイル等を直接的に ps プリンタに出力するための機能」が  ",
      "  備わっていない場合には、「取消し」ボタンを押して下さい。       ",
      "  ",
      " 直接的なプリントのための前提条件： ",
      "   0.  latex のタイプとして platex (LaTeX2e) が選択されていること。 ",
      "   1.  platex  がインストールされていること。 ",
      "   2.  dvipsk  がインストールされていること。",
      "   3.  マシンが ps プリンタに接続されていること。",
      "  ",
      "  システムが実行する作業の手順: ",
      "    (1) latex のソースファイル XXX.tex が作成される。 ",
      "    (2) platex XXX  ( XXX.tex ==> XXX.dvi, XXX.aux, XXX.log )",
      "    (3) dvipsk XXX  ( XXX.dvi ==> XXX.ps )",
      "    (4) XXX.ps を「プリントジョブ」に追加する ",
      "        これにより ps ファイル XXX.ps の内容は「そのマシンに設定されている ",
      "        デフォルトの ps プリンタ」に出力される。 ",
      "    (5) XXX.tex, XXX.dvi, XXX.aux, XXX.log, XXX.ps を削除する ",
      "  ",
      "  platex と dvipsk をインストールする方法が分からない場合には、学務係から ",
      "  インストール用の CD を借りて、それらをインストールして下さい。 ",
      "  "
      };
      
    int ans = JOptionPane.showOptionDialog(commonInfo.getFrame(), 
					   warning, 
					   "PSプリンタの利用に関する注意", 
					   JOptionPane.OK_CANCEL_OPTION,
					   JOptionPane.QUESTION_MESSAGE,
					   null,
					   null,
					   null);
    if (ans == JOptionPane.OK_OPTION) {
      warningFlag = true;
      pdfFlag = false;
      psPrinterFlag = true;
    } else {
      pdfFlag = true;
      psPrinterFlag = false;
    }

    if (commonInfo.psPrintService == null) {
      commonInfo.showMessage("このマシンには PS プリンタが接続されていません。");
      pdfFlag = true;
      psPrinterFlag = false;  
    }
  }

  public void makeLatexSource() {
    StringBuffer sbuf = new StringBuffer();  
    boolean res = makeLatexSource(sbuf); 
    if (!res) return;
  
    File f = null;
    int ret = chooser.showSaveDialog(commonInfo.getFrame());
    if (ret == JFileChooser.APPROVE_OPTION) {
      f = chooser.getSelectedFile(); 
    } else {
      return;
    } 
    try {
      PrintWriter fout = new PrintWriter(new BufferedWriter(new FileWriter(f)));
      fout.println(sbuf.toString());
      fout.close();
    } catch (Exception e) {
      commonInfo.showMessage(e.toString());
      return;
    }
  }

  public void printNinteiTable() {
    showInitialPrinterWarning();
    StringBuffer sbuf = new StringBuffer();
    boolean res = makeNinteiLatexSource(sbuf);
    if (res) {
      latexToPrinter(sbuf);
    }
  }

  private String[][] curriculumInfo1 = { { "31", "10", "知能情報工学科", "１年次入学者用", "AI-1-CURRICULUM" },
					 { "32", "10", "電子情報工学科", "１年次入学者用", "ELEC-1-CURRICULUM" },
					 { "33", "10", "システム創成情報工学科", "１年次入学者用", "CONT-1-CURRICULUM" },
					 { "34", "10", "機械情報工学科", "１年次入学者用", "MECH-1-CURRICULUM" },
					 { "35", "10", "生命情報工学科", "１年次入学者用", "BIO-1-CURRICULUM" },
					 { "30", "10", "情報工学部共通", "１年次入学者用", "COMMON-1-CURRICULUM" } };

  private String[][] curriculumInfo3 = { { "31", "30", "知能情報工学科", "３年次編入学者用", "AI-3-CURRICULUM" },
					 { "32", "30", "電子情報工学科", "３年次編入学者用", "ELEC-3-CURRICULUM" },
					 { "33", "30", "システム創成情報工学科", "３年次編入学者用", "CONT-3-CURRICULUM" },
					 { "34", "30", "機械情報工学科", "３年次編入学者用", "MECH-3-CURRICULUM" },
					 { "35", "30", "生命情報工学科", "３年次編入学者用", "BIO-3-CURRICULUM" },
					 { "30", "30", "情報工学部共通", "３年次編入学者用", "COMMON-3-CURRICULUM" } };

  private String[][] ninteiFormatInfo = { { "31", "30", "知能情報工学科", "３年次編入学者用", "AI-FORMAT" },
					  { "32", "30", "電子情報工学科", "３年次編入学者用", "ELEC-FORMAT" },
					  { "33", "30", "システム創成情報工学科", "３年次編入学者用", "CONT-FORMAT" },
					  { "34", "30", "機械情報工学科", "３年次編入学者用", "MECH-FORMAT" },
					  { "35", "30", "生命情報工学科", "３年次編入学者用", "BIO-FORMAT" } };


  public void makeNinteiFormTableLatex() { 
    int lineNumber = 36;

    String title1 = "3年次編入生用　単位認定科目　希望調査書";
    String curriculumYear = tabbedPane.getValueFromColumnCodeMap("CURRICULUM_YEAR");
    String curriculumYearName = tabbedPane.getDisplayFromColumnCodeMap("CURRICULUM_YEAR");
    String faculty = "11";
    String facultyName = "情報工学部";

    String homeDir = System.getProperty("user.home");
    File dir = new File(homeDir, "デスクトップ");  
    File dir1 = new File(dir, "WEB-HENNYU");  
    File dir2 = new File(dir1, "CurriculumPDF");  

    for (String[] deptInfo : ninteiFormatInfo) {
      String department = deptInfo[0];
      String course = deptInfo[1];
      String departmentName = deptInfo[2];
      String courseName = deptInfo[3];
      String fname = deptInfo[4] + ".tex";
      File f = new File(dir2, fname);
      PrintWriter fout;
      try {
	fout = new PrintWriter(new BufferedWriter(new FileWriter(f)));
      } catch (Exception e) {
	commonInfo.showMessage(e.toString());
	return;
      }

      String title2 = "" + curriculumYearName + " : " + facultyName + " : " + departmentName;
      String dateText = ""+commonInfo.thisYear+":"+commonInfo.thisMonth+":"+commonInfo.thisDay;
 
      if (platexIsSelected()) {
	fout.println("\\documentclass[11pt]{jarticle}");
      } else {
	fout.println("\\documentstyle[11pt]{jarticle}");
      }
      fout.println("\\setlength{\\oddsidemargin}{-30pt}");
      fout.println("\\setlength{\\evensidemargin}{-30pt}");
      fout.println("\\setlength{\\topmargin}{-30pt}");
      fout.println("\\setlength{\\textwidth}{500pt}");
      fout.println("\\setlength{\\textheight}{720pt}");
      fout.println("\\begin{document}");
      fout.println("\\pagestyle{empty}");
      fout.println("\\renewcommand{\\arraystretch}{1.06}");

      boolean endFlag = false;
      int pageCount = 1;
      int lineCount = 0;
      int colCount = 7;

      String ans = commonInfo.getInfoForCurriculumTablePrint(curriculumYear, faculty, department, course);
      String[] rowInfo = ans.split("\\$");
      int rowCount = rowInfo.length;
      int cnt = 0;

      while (!endFlag) {
	fout.println("\\begin{table}[t]\n \\begin{tabular}[t]{|r|r|r|r|r||r|r|} \\hline");
	fout.println("\\multicolumn{"+colCount+"}{|c|}{  "+title1+"  \\hfill ("+dateText+")}  \\\\ \\hline");
	fout.println("\\multicolumn{"+colCount+"}{|c|}{  "+title2+"  \\hfill ("+pageCount+")}  \\\\ \\hline\\hline");
	pageCount++;
	fout.println("\\multicolumn{"+colCount+"}{|l|}{  "+"単位認定を希望する科目については「認定希望」欄に ◯ 印を記入し、「認定の根拠となる修得科目」欄には }  \\\\ \\hline");
	fout.println("\\multicolumn{"+colCount+"}{|l|}{  "+"その単位認定の根拠となる高専等での修得科目 (シラバスの頁を附記) を記入して下さい。 }  \\\\ \\hline\\hline");
	fout.println("年次 & 科目名 & 科目区分 & 単位区分 & 単位数 & 認定希望 & 認定の根拠となる修得科目\\\\ \\hline\\hline");

	for (int j = 0; j < lineNumber; j++) {
	  if (cnt < rowCount) {
	    String[] item = rowInfo[cnt].split("\\|");
	    cnt++;
	    String gakunen = item[0];
	    String subject = item[1];
	    String kubun   = item[2];
	    String req     = item[3];
	    String unit    = item[4];
	    if (subject.length() > 15) {
	      subject = subject.substring(0,15);
	    }
	    StringBuffer sbuf = new StringBuffer();
	    sbuf.append(gakunen).append(" & ");
	    sbuf.append(subject).append(" & ");
	    sbuf.append(kubun).append(" & ");
	    sbuf.append(req).append(" & ");
	    sbuf.append(unit).append(" & & ");
	    fout.println(sbuf.toString());	  
	  } else {
	    StringBuffer sbuf = new StringBuffer();
	    for (int k = 0; k < (colCount - 1); k++) {
	      sbuf.append(" & ");
	    }	
	    fout.println(sbuf.toString());	
	    endFlag = true;  
	  }
	  lineCount++;
	  if ((lineCount % 10) == 0) {
	    fout.println("\\\\ \\hline\\hline ");
	  } else {
	    fout.println("\\\\ \\hline ");
	  }
	}
	fout.println("\\end{tabular}");
	fout.println("\\end{table}");
      }
      fout.println("\\end{document}");
      fout.close();
    }
  }

  public void makeCurriculum1TableLatex() {
    int lineNumber = 40;
    String curriculumYear = tabbedPane.getValueFromColumnCodeMap("CURRICULUM_YEAR");
    String curriculumYearName = tabbedPane.getDisplayFromColumnCodeMap("CURRICULUM_YEAR");
    String faculty = "11";
    String facultyName = "情報工学部";

    String homeDir = System.getProperty("user.home");
    File dir = new File(homeDir, "デスクトップ");  
    File dir1 = new File(dir, "WEB-HENNYU");  
    File dir2 = new File(dir1, "CurriculumPDF");  

    for (String[] deptInfo : curriculumInfo1) {
      String department = deptInfo[0];
      String course = deptInfo[1];
      String departmentName = deptInfo[2];
      String courseName = deptInfo[3];
      String fname = deptInfo[4] + ".tex";
      File f = new File(dir2, fname);
      PrintWriter fout;
      try {
	fout = new PrintWriter(new BufferedWriter(new FileWriter(f)));
      } catch (Exception e) {
	commonInfo.showMessage(e.toString());
	return;
      }

      String title = "履修課程表: " + curriculumYearName + " : " + facultyName + " : " + departmentName + " : " + courseName;
      String dateText = ""+commonInfo.thisYear+":"+commonInfo.thisMonth+":"+commonInfo.thisDay;
 
      if (platexIsSelected()) {
	fout.println("\\documentclass[11pt]{jarticle}");
      } else {
	fout.println("\\documentstyle[11pt]{jarticle}");
      }
      fout.println("\\setlength{\\oddsidemargin}{-30pt}");
      fout.println("\\setlength{\\evensidemargin}{-30pt}");
      fout.println("\\setlength{\\topmargin}{-30pt}");
      fout.println("\\setlength{\\textwidth}{500pt}");
      fout.println("\\setlength{\\textheight}{720pt}");
      fout.println("\\begin{document}");
      fout.println("\\pagestyle{empty}");
      fout.println("\\renewcommand{\\arraystretch}{1.06}");

      boolean endFlag = false;
      int pageCount = 1;
      int lineCount = 0;
      int colCount = 7;

      String ans = commonInfo.getInfoForCurriculumTablePrint(curriculumYear, faculty, department, course);
      String[] rowInfo = ans.split("\\$");
      int rowCount = rowInfo.length;
      int cnt = 0;

      while (!endFlag) {
	fout.println("\\begin{table}[t]\n \\begin{tabular}[t]{|r|r|r|r|r||r|r|} \\hline");
	fout.println("\\multicolumn{"+colCount+"}{|c|}{  "+title+"  \\hfill "+dateText+" 　("+pageCount+")}  \\\\ \\hline\\hline");
	pageCount++; 
	fout.println("年次 & 科目名 & 科目区分 & 単位区分 & 単位数 & 　　　　 & 　　　　 \\\\ \\hline\\hline");

	for (int j = 0; j < lineNumber; j++) {
	  if (cnt < rowCount) {
	    String[] item = rowInfo[cnt].split("\\|");
	    cnt++;
	    String gakunen = item[0];
	    String subject = item[1];
	    String kubun   = item[2];
	    String req     = item[3];
	    String unit    = item[4];
	    StringBuffer sbuf = new StringBuffer();
	    sbuf.append(gakunen).append(" & ");
	    sbuf.append(subject).append(" & ");
	    sbuf.append(kubun).append(" & ");
	    sbuf.append(req).append(" & ");
	    sbuf.append(unit).append(" & & ");
	    fout.println(sbuf.toString());	  
	  } else {
	    StringBuffer sbuf = new StringBuffer();
	    for (int k = 0; k < (colCount - 1); k++) {
	      sbuf.append(" & ");
	    }	
	    fout.println(sbuf.toString());
	    endFlag = true;  
	  }
	  lineCount++;
	  if ((lineCount % 10) == 0) {
	    fout.println("\\\\ \\hline\\hline ");
	  } else {
	    fout.println("\\\\ \\hline ");
	  }
	}
	fout.println("\\end{tabular}");
	fout.println("\\end{table}");
      }
      fout.println("\\end{document}");
      fout.close();
    }
  }

  public void makeCurriculum3TableLatex() {
    int lineNumber = 40;
    String curriculumYear = tabbedPane.getValueFromColumnCodeMap("CURRICULUM_YEAR");
    String curriculumYearName = tabbedPane.getDisplayFromColumnCodeMap("CURRICULUM_YEAR");
    String faculty = "11";
    String facultyName = "情報工学部";

    String homeDir = System.getProperty("user.home");
    File dir = new File(homeDir, "デスクトップ");  
    File dir1 = new File(dir, "WEB-HENNYU");  
    File dir2 = new File(dir1, "CurriculumPDF");  

    for (String[] deptInfo : curriculumInfo3) {
      String department = deptInfo[0];
      String course = deptInfo[1];
      String departmentName = deptInfo[2];
      String courseName = deptInfo[3];
      String fname = deptInfo[4] + ".tex";
      File f = new File(dir2, fname);
      PrintWriter fout;
      try {
	fout = new PrintWriter(new BufferedWriter(new FileWriter(f)));
      } catch (Exception e) {
	commonInfo.showMessage(e.toString());
	return;
      }

      String title = "履修課程表: " + curriculumYearName + " : " + facultyName + " : " + departmentName + " : " + courseName;
      String dateText = ""+commonInfo.thisYear+":"+commonInfo.thisMonth+":"+commonInfo.thisDay;
 
      if (platexIsSelected()) {
	fout.println("\\documentclass[11pt]{jarticle}");
      } else {
	fout.println("\\documentstyle[11pt]{jarticle}");
      }
      fout.println("\\setlength{\\oddsidemargin}{-30pt}");
      fout.println("\\setlength{\\evensidemargin}{-30pt}");
      fout.println("\\setlength{\\topmargin}{-30pt}");
      fout.println("\\setlength{\\textwidth}{500pt}");
      fout.println("\\setlength{\\textheight}{720pt}");
      fout.println("\\begin{document}");
      fout.println("\\pagestyle{empty}");
      fout.println("\\renewcommand{\\arraystretch}{1.06}");

      boolean endFlag = false;
      int pageCount = 1;
      int lineCount = 0;
      int colCount = 7;

      String ans = commonInfo.getInfoForCurriculumTablePrint(curriculumYear, faculty, department, course);
      String[] rowInfo = ans.split("\\$");
      int rowCount = rowInfo.length;
      int cnt = 0;

      while (!endFlag) {
	fout.println("\\begin{table}[t]\n \\begin{tabular}[t]{|r|r|r|r|r||r|r|} \\hline");
	fout.println("\\multicolumn{"+colCount+"}{|c|}{  "+title+"  \\hfill "+dateText+" 　("+pageCount+")}  \\\\ \\hline\\hline");
	pageCount++; 
	fout.println("年次 & 科目名 & 科目区分 & 単位区分 & 単位数 & 　　　　 & 　　　　 \\\\ \\hline\\hline");

	for (int j = 0; j < lineNumber; j++) {
	  if (cnt < rowCount) {
	    String[] item = rowInfo[cnt].split("\\|");
	    cnt++;
	    String gakunen = item[0];
	    String subject = item[1];
	    String kubun   = item[2];
	    String req     = item[3];
	    String unit    = item[4];
	    StringBuffer sbuf = new StringBuffer();
	    sbuf.append(gakunen).append(" & ");
	    sbuf.append(subject).append(" & ");
	    sbuf.append(kubun).append(" & ");
	    sbuf.append(req).append(" & ");
	    sbuf.append(unit).append(" & & ");
	    fout.println(sbuf.toString());	  
	  } else {
	    StringBuffer sbuf = new StringBuffer();
	    for (int k = 0; k < (colCount - 1); k++) {
	      sbuf.append(" & ");
	    }	
	    fout.println(sbuf.toString());
	    endFlag = true;  
	  }
	  lineCount++;
	  if ((lineCount % 10) == 0) {
	    fout.println("\\\\ \\hline\\hline ");
	  } else {
	    fout.println("\\\\ \\hline ");
	  }
	}
	fout.println("\\end{tabular}");
	fout.println("\\end{table}");
      }
      fout.println("\\end{document}");
      fout.close();
    }
  }


  public void makeNinteiTableLatex() {
    StringBuffer sbuf = new StringBuffer();  
    boolean res = makeNinteiLatexSource(sbuf); 
    if (res == false) return;
    
    File f = null;
    int ret = chooser.showSaveDialog(commonInfo.getFrame());
    if (ret == JFileChooser.APPROVE_OPTION) {
      f = chooser.getSelectedFile(); 
    } else {
      return;
    } 
    try {
      PrintWriter fout = new PrintWriter(new BufferedWriter(new FileWriter(f)));
      fout.println(sbuf.toString());
      fout.close();
    } catch (Exception e) {
      commonInfo.showMessage(e.toString());
      return;
    }
  }


  public boolean makeNinteiLatexSource(StringBuffer sbuf) {
    int ans = JOptionPane.showOptionDialog(commonInfo.getFrame(), 
					   "「３年次編入学生用の履修課程表」が表示されていることを確認", 
					   "確認", 
					   JOptionPane.OK_CANCEL_OPTION,
					   JOptionPane.QUESTION_MESSAGE,
					   null,
					   null,
					   null);
    if (ans != JOptionPane.OK_OPTION) {
      return false;
    }

    int lineNumber = 36;
//    int cnt = 0;
    String title1 = "3年次編入生用　単位認定科目　希望調査書";
    String curriculumYear = tabbedPane.getDisplayFromColumnCodeMap("CURRICULUM_YEAR");
    String faculty = tabbedPane.getDisplayFromColumnCodeMap("FACULTY");
    String department = tabbedPane.getDisplayFromColumnCodeMap("DEPARTMENT");

    String title2 = "" + curriculumYear + " : " + faculty + " : " + department;
    String dateText = ""+commonInfo.thisYear+":"+commonInfo.thisMonth+":"+commonInfo.thisDay;

    if (platexIsSelected()) {
      sbuf.append("\\documentclass[11pt]{jarticle}\n");
    } else {
      sbuf.append("\\documentstyle[11pt]{jarticle}\n");
    }
    sbuf.append("\\setlength{\\oddsidemargin}{-30pt}\n");
    sbuf.append("\\setlength{\\evensidemargin}{-30pt}\n");
    sbuf.append("\\setlength{\\topmargin}{-30pt}\n");
    sbuf.append("\\setlength{\\textwidth}{500pt}\n");
    sbuf.append("\\setlength{\\textheight}{720pt}\n");
    sbuf.append("\\begin{document}\n");
    sbuf.append("\\pagestyle{empty}\n");
    sbuf.append("\\renewcommand{\\arraystretch}{1.06}\n");

    boolean endFlag = false;
    int pageCount = 1;
    int lineCount = 0;
    int colCount = 7;

    while (!endFlag) {
      String str1, str2, str3, str4;

      str1 = "\\begin{table}[t]\n \\begin{tabular}[t]{|r|r|r|r|r||r|r|} \\hline \n";
      str2 = "\\multicolumn{"+colCount+"}{|c|}{  "+title1+"  \\hfill ("+dateText+")}  \\\\ \\hline \n";
      str3 = "\\multicolumn{"+colCount+"}{|c|}{  "+title2+"  \\hfill ("+pageCount+")}  \\\\ \\hline\\hline \n";
      pageCount++;
      String str51 = "\\multicolumn{"+colCount+"}{|l|}{  "+"単位認定を希望する科目については「認定希望」欄に ◯ 印を記入し、「認定の根拠となる修得科目」欄には }  \\\\ \\hline \n";
      String str52 = "\\multicolumn{"+colCount+"}{|l|}{  "+"その単位認定の根拠となる高専等での修得科目 (シラバスの頁を附記) を記入して下さい。 }  \\\\ \\hline\\hline \n";
      String str6 = "年次 & 科目名 & 科目区分 & 単位区分 & 単位数 & 認定希望 & 認定の根拠となる修得科目\\\\ \\hline\\hline \n";

      sbuf.append(str1);
      sbuf.append(str2);
      sbuf.append(str3);
      sbuf.append(str51);
      sbuf.append(str52);
      sbuf.append(str6);  

      for (int j = 0; j < lineNumber; j++) {
	if (lineCount < tableView.getRowCount()) {
	  for (int i = 0; i < tableView.getColumnCount(); i++) {
	    String colName = tableView.getColumnName(i);
	    if ((colName.equals("学年")) 
		|| (colName.equals("授業科目名")) 
		|| (colName.equals("科目区分")) 
		|| (colName.equals("単位区分")) 
		|| (colName.equals("単位数"))) {
	      CellObject cobj = (CellObject)tableView.getValueAt(lineCount, i);
	      sbuf.append(cobj.getDisplay()).append(" & ");
	    }
	  }
	  sbuf.append(" & ");
	} else {
	  endFlag = true;
	  for (int i = 0; i < (colCount - 1); i++) {
	    sbuf.append(" & ");
	  }	  
	}
	lineCount++;
	if ((lineCount % 10) == 0) {
	  sbuf.append("\\\\ \\hline\\hline ").append(lineSeparator); 
	} else {
	  sbuf.append("\\\\ \\hline ").append(lineSeparator); 
	}
      }
      sbuf.append("\\end{tabular}").append(lineSeparator);
      sbuf.append("\\end{table}").append(lineSeparator);
    }
    sbuf.append("\\end{document}").append(lineSeparator); 
    return true;
  }



  public void printStudentInitialPassword(String code, String name, 
					  String passwd, String status) {
    StringBuffer sbuf = new StringBuffer();
    String faculty = tabbedPane.getDisplayFromColumnCodeMap("FACULTY");
    String department = tabbedPane.getDisplayFromColumnCodeMap("DEPARTMENT");
    String course = tabbedPane.getDisplayFromColumnCodeMap("COURSE");
    String gakunen = tabbedPane.getDisplayFromColumnCodeMap("GAKUNEN");
    String attrib = faculty + " " + department + " " + course + " " + gakunen;
    String today = "" + commonInfo.thisYear + "年" + commonInfo.thisMonth + "月"
      + commonInfo.thisDay + "日";

    if (!latexFlag) {
      showLatexTypeSelector();
    }

    if (platexIsSelected()) {
      sbuf.append("\\documentclass[11pt]{jarticle}\n");
    } else {
      sbuf.append("\\documentstyle[11pt]{jarticle}\n");
    }
    sbuf.append("\\setlength{\\oddsidemargin}{0pt}\n");
    sbuf.append("\\setlength{\\topmargin}{0pt}\n");
    sbuf.append("\\setlength{\\textwidth}{500pt}\n");
    sbuf.append("\\setlength{\\textheight}{720pt}\n");
    sbuf.append("\\begin{document}\n");
    sbuf.append("\\pagestyle{empty}\n");
    sbuf.append("\\large \n");    
    sbuf.append("\\begin{verbatim}\n");
    sbuf.append("初期パスワード\n\n");
    sbuf.append("初期パスワード印刷日時 :  " + today + "\n");
    sbuf.append("　　　　　　所属学科等 :  " + attrib + "\n");
    sbuf.append("　　　　　　　学生氏名 :  " + name + "\n");
    sbuf.append("　　　　　　　学生番号 :  " + code + "\n");
    sbuf.append("　　　　初期パスワード :  " + passwd + "\n");
    sbuf.append("　　　　　　　在籍状況 :  " + status + "\n\n\n");
    sbuf.append("初期パスワードのままで放置するとあなたの教務情報に\n");
    sbuf.append("他人がアクセスする危険が生じます。 \n");
    sbuf.append("今日中にパスワードを設定して下さい。\n");
    sbuf.append("\\end{verbatim}\n");
    sbuf.append("\\end{document}\n");
    latexToPrinter(sbuf); 
  }


  public void printStaffInitialPassword(String code, String name, 
					String id, String passwd) {
    StringBuffer sbuf = new StringBuffer();
    String today = "" + commonInfo.thisYear + "年" + commonInfo.thisMonth + "月"
      + commonInfo.thisDay + "日";

    if (!latexFlag) {
      showLatexTypeSelector();
    }

    if (platexIsSelected()) {
      sbuf.append("\\documentclass[11pt]{jarticle}\n");
    } else {
      sbuf.append("\\documentstyle[11pt]{jarticle}\n");
    }
    sbuf.append("\\setlength{\\oddsidemargin}{0pt}\n");
    sbuf.append("\\setlength{\\topmargin}{0pt}\n");
    sbuf.append("\\setlength{\\textwidth}{500pt}\n");
    sbuf.append("\\setlength{\\textheight}{720pt}\n");
    sbuf.append("\\begin{document}\n");
    sbuf.append("\\pagestyle{empty}\n");
    sbuf.append("\\large \n");    
    sbuf.append("\\begin{verbatim}\n");
    sbuf.append("初期パスワード\n\n");
    sbuf.append("パスワード初期化の日時 :  " + today + "\n");
    sbuf.append("　　　　　　　職員氏名 :  " + name + "\n");
    sbuf.append("　　　　　　　職員 ID  :  " + id + "\n");
    sbuf.append("　　　　　　職員コード :  " + code + "\n");
    sbuf.append("　　　　初期パスワード :  " + passwd + "\n\n\n");
    sbuf.append("この初期パスワードは数日で無効になります。\n");
    sbuf.append("今日中にパスワードを設定して下さい。\n");
    sbuf.append("\\end{verbatim}\n");
    sbuf.append("\\end{document}\n");
    latexToPrinter(sbuf);
  }

  public void makeTextFile() {
    File f = null;
    int ret = chooser.showSaveDialog(commonInfo.getFrame());
    if (ret == JFileChooser.APPROVE_OPTION) {
      f = chooser.getSelectedFile(); 
    } else {
      return;
    }

    ArrayList<Object> list = new ArrayList<Object>(delimList);

    int ans = JOptionPane.showOptionDialog(commonInfo.getFrame(), 
					   list.toArray(), 
					   "印刷設定", 
					   JOptionPane.OK_CANCEL_OPTION,
					   JOptionPane.QUESTION_MESSAGE,
					   null,
					   null,
					   null);

    if (ans != JOptionPane.OK_OPTION) {
      return;
    }

    StringBuffer sbuf = new StringBuffer();    
    makeTextSource(sbuf);    
    try {
      PrintWriter fout = new PrintWriter(new BufferedWriter(new FileWriter(f)));
      fout.print(sbuf.toString());
      fout.close();
    } catch (Exception e) {
      commonInfo.showMessage(e.toString());
      return;
    }
  }

  public void makeTextSource(StringBuffer sbuf) { 
    String delim = getDelimSelection();

    ArrayList<Object> list = new ArrayList<Object>();
    JPanel panel = getColumnSelectionPanel();
    list.add(panel);

    int ans = JOptionPane.showOptionDialog(commonInfo.getFrame(), 
					   list.toArray(), 
					   "出力するカラムの設定", 
					   JOptionPane.OK_CANCEL_OPTION,
					   JOptionPane.QUESTION_MESSAGE,
					   null,
					   null,
					   null);
    if (ans != JOptionPane.OK_OPTION) {
      return;
    }

    if (counterCheckBox.isSelected()) {
      counterSelected = true;
    } else {
      counterSelected = false;
    }
  
    String titleText = titleTextTF.getText().trim();
    sbuf.append(titleText).append("   ( ");
    String dateText = ""+commonInfo.thisYear+":"+commonInfo.thisMonth+":"+commonInfo.thisDay;
    sbuf.append(dateText).append(" )").append(lineSeparator); 
    sbuf.append("").append(lineSeparator); 

    
    if (counterSelected) {
	sbuf.append("通し番号").append(delim);
    }       

    for (int i = 0; i < tableView.getColumnCount(); i++) { 
      if (checkBox[i].isSelected()) {
	String columnName = tableView.getColumnName(i);
	if (columnName.equals("")) {
	  columnName = " ";
	}
	sbuf.append(columnName).append(delim); 
      }
    }
    sbuf.append(lineSeparator);
    
    int lineCount = 1;
    for (int j = 0; j < tableView.getRowCount(); j++) {
      if (counterSelected) {
	sbuf.append("" + lineCount).append(delim);
	lineCount++;
      }      
      for (int i = 0; i < tableView.getColumnCount(); i++) {
	if (checkBox[i].isSelected()) {
	  CellObject cobj = (CellObject)tableView.getValueAt(j, i);
	  String text = cobj.getDisplay().trim();
	  if (text.equals("")) {
	    text = " ";
	  }
	  sbuf.append(text).append(delim); 
	}
      }
      sbuf.append(lineSeparator);
    }
  }
  
  public void appendHead(StringBuffer sbuf, String deptName) {    
    sbuf.append("\\documentclass[11pt]{jbook}\n");
    sbuf.append("\\def\\linesparpage#1{\\baselineskip=\\textheight\\divide\\baselineskip#1}\n");
    sbuf.append("\\setlength{\\oddsidemargin}{-10pt}\n");
    sbuf.append("\\setlength{\\evensidemargin}{-10pt}\n");
    sbuf.append("\\setlength{\\topmargin}{-30pt}\n");
    sbuf.append("\\setlength{\\textwidth}{480pt}\n");
    sbuf.append("\\setlength{\\textheight}{700pt}\n");
    sbuf.append("\\begin{document}\n");
    sbuf.append("\\setcounter{secnumdepth}{-1}\n");
    sbuf.append("\\linesparpage{55}\n");
    sbuf.append("\\tableofcontents\n");
    sbuf.append("\\chapter{ " + deptName + " シラバス}\n" );
  }

  public void appendSection(StringBuffer sbuf, String schoolYear, String deptName, int gakunen) {
    sbuf.append("\\section{ " + schoolYear + "年度　" + deptName + "　" + gakunen + "年次 }\n");
    sbuf.append("\\vspace{3mm}\n");
  }

  public void appendSection2(StringBuffer sbuf, String schoolYear, String kubunName) {
    sbuf.append("\\section{ " + schoolYear + "年度　学部共通科目　" + kubunName + "}\n");
    sbuf.append("\\vspace{3mm}\n");
  }

  public void appendSection3(StringBuffer sbuf, String schoolYear, String kubunName) {
    sbuf.append("\\section{ " + schoolYear + "年度　大学院科目　" + kubunName + " }\n");
    sbuf.append("\\vspace{3mm}\n");
  }

  public void appendSection4(StringBuffer sbuf, String schoolYear, String deptName, String kubunName) {
    sbuf.append("\\section{ " + schoolYear + "年度　" + deptName + " " + kubunName + " }\n");
    sbuf.append("\\vspace{0.3cm}\n");
  }

  public void appendSubsection(StringBuffer sbuf, ArrayList<Object> list, 
			       String param, HashMap<String, String> map) {
    try {
      String code;
      String subjectCode = (String)list.get(0);
      String classCode   = (String)list.get(1);
      String teacherCode = (String)list.get(2);
      String gakunen   = (String)list.get(3);
      
      code = (String)list.get(4);
      String semester;
      if (code.trim().equals("")) {
	gakunen = " ";
	semester = "(本年度休講)";
      } else {
	gakunen = gakunen + "年";
	switch (Integer.parseInt(code)) {
	case 1:
	  semester = "前期"; break;
	case 2:
	  semester = "後期"; break;
	case 3:
	  semester = "通年"; break;
	default:
	  semester = "(休講)"; break;
	}
      }
      code = (String)list.get(5);
      String kubun = commonInfo.getGakumuCodeName("KUBUN_CODE", code);
      code = (String)list.get(6);
      String req = commonInfo.getGakumuCodeName("REQ_CODE", code);
      code = (String)list.get(7);
      String unit = code + "単位";
      code = (String)list.get(8);
      String room = commonInfo.getGakumuCodeName("ROOM", code);
      String subjectName = (String)list.get(9);
      String englishName = (String)list.get(10);
      String teacherName = (String)list.get(11);
      
      code  = (String)list.get(12);
      String staffAttrib = commonInfo.getGakumuCodeName("STAFF_ATTRIB", code);
      code  = (String)list.get(13);
      if (code.equals("3")) {
	staffAttrib = "非常勤教員";
      }
      String mailAddress = (String)list.get(14);
      String weekHour = "";
      for (int i = 15; i < list.size(); i++) {
	String wh = (String)list.get(i);
	weekHour = weekHour + convertWeekHour(wh);
      }    
      
      int len = subjectName.length();
      int snum = 0;
      int vnum = 0;
      int inum = 0;
      int num = 0;
      int zlen = 0;
      double addspace;

      for (int i = 0; i < len; i++) {
	char ch = subjectName.charAt(i);
	if (ch == 'I') inum++; 
	if (ch == 'V') vnum++;
	if (ch == ' ') snum++;
      }
      num = inum + vnum + snum;
      double x = inum * 0.4 + snum * 0.3 + vnum * 0.8;
      zlen = len - num + (int)x;
      double rem = x - ((int)x);
      if (rem > 0) {
	zlen++;
	addspace = 1.0 - rem;
      } else {
	addspace = 0.0;
      }     
      
      StringBuffer sbuf5 = new StringBuffer();
      sbuf5.append(subjectName);
      if (zlen < 18) {
	for (int i = zlen; i < 18; i++) {
	  sbuf5.append("　");
	}
      }
      String subjectName5 = sbuf5.toString();

      int klen = kubun.length();
      int kadd = 6 - klen;

      sbuf.append("\\subsection{ "+subjectName5+"\\hspace{"+addspace+"zw} \\dotfill　"+teacherName+"　(" + kubun+ ")\\hspace{"+kadd+"zw} }\n");
      sbuf.append("{\\small \\begin{quotation}\n");
      sbuf.append("\\noindent 科目名:　" + subjectName + "　(" + classCode + ") \\hspace{0.3cm} " + englishName + " \\\\ \n");
      sbuf.append(" 担当教員:　" + teacherName + "  (" + staffAttrib + ")  \\verb+  " + mailAddress + " +\\\\ \n");
      sbuf.append(" "+kubun+" \\hspace{0.1cm} " + req + " " + unit + "  \\\\ \n");
      sbuf.append(" " + gakunen + " " + semester + "　" + weekHour + "　" + room + " \n");
      
      if (!map.containsKey(param)) {
	map.put(param, "1");
	appendMainText(sbuf, param);
      }
      
      sbuf.append("\\end{quotation} } \n");
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  public void appendSubsection2(StringBuffer sbuf, ArrayList<Object> list, 
				String param, HashMap<String, String> map) {
    try {
      String code;
      String subjectCode = (String)list.get(0);
      String classCode   = (String)list.get(1);
      String teacherCode = (String)list.get(2);
      String gakunen   = (String)list.get(3);
      
      code = (String)list.get(4);
      String semester;
      if (code.trim().equals("")) {
	semester = "(本年度休講)";
      } else {
	switch (Integer.parseInt(code)) {
	case 1:
	  semester = "前期"; break;
	case 2:
	  semester = "後期"; break;
	case 3:
	  semester = "通年"; break;
	default:
	  semester = "(休講)"; break;
	}
      }
      code = (String)list.get(5);
      String kubun = commonInfo.getGakumuCodeShorterName("KUBUN_CODE", code);
      code = (String)list.get(6);
      String req = commonInfo.getGakumuCodeName("REQ_CODE", code);
      code = (String)list.get(7);
      String unit = code + "単位";
      code = (String)list.get(8);
      String room = commonInfo.getGakumuCodeName("ROOM", code);
      String subjectName = (String)list.get(9);
      String englishName = (String)list.get(10);
      String teacherName = (String)list.get(11);
      
      code  = (String)list.get(12);
      String staffAttrib = commonInfo.getGakumuCodeName("STAFF_ATTRIB", code);
      code  = (String)list.get(13);
      if (code.equals("3")) {
	staffAttrib = "非常勤教員";
      }
      String mailAddress = (String)list.get(14);

      String deptGakunen = "";
      int deptCount = 0;
      String weekHour = "";
      
      for (int i = 15; i < list.size(); i++) {
	String tmp = (String)list.get(i);
	if (tmp.length() >= 5) {
	  deptGakunen = deptGakunen + convertDeptGakunen(tmp);
	  deptCount++;
	} else {
	  weekHour = weekHour + convertWeekHour(tmp);
	}  
      }
      if (deptCount >= 5) {
	deptCount = 1;
	deptGakunen = "共通科目 " + gakunen;
      }   


      int len = subjectName.length();
      int snum = 0;
      int vnum = 0;
      int inum = 0;
      int num = 0;
      int zlen = 0;
      double addspace;

      for (int i = 0; i < len; i++) {
	char ch = subjectName.charAt(i);
	if (ch == 'I') inum++; 
	if (ch == 'V') vnum++;
	if (ch == ' ') snum++;
      }
      num = inum + vnum + snum;
      double x = inum * 0.4 + snum * 0.3 + vnum * 0.8;
      zlen = len - num + (int)x;
      double rem = x - ((int)x);
      if (rem > 0) {
	zlen++;
	addspace = 1.0 - rem;
      } else {
	addspace = 0.0;
      }     
      
      StringBuffer sbuf5 = new StringBuffer();
      sbuf5.append(subjectName);
      if (zlen < 18) {
	for (int i = zlen; i < 18; i++) {
	  sbuf5.append("　");
	}
      }
      String subjectName5 = sbuf5.toString();

      int klen = kubun.length();
      int kadd = 6 - klen;

      sbuf.append("\\subsection{ "+subjectName5+"\\hspace{"+addspace+"zw} \\dotfill　"+teacherName+"　(" + kubun+ ")\\hspace{"+kadd+"zw} }\n");
      sbuf.append("\\begin{quotation}\n");
      sbuf.append("\\noindent {\\small 科目名:　" + subjectName + "　(" + classCode + ") \\hspace{0.3cm} " + englishName + " } \\\\ \n");
      sbuf.append("{\\small 担当教員:　" + teacherName + "  (" + staffAttrib + ")  \\verb+  " + mailAddress + " +}\\\\ \n");
      sbuf.append("{\\small "+kubun+" \\hspace{0.1cm} " + req + " " + unit + " \\hspace{0.1cm} }\\\\ \n");
      
      if (deptCount == 1) {
	sbuf.append("{\\small " + deptGakunen + "　" + semester + "  " + weekHour + " " + room + " } \n");
      } else {
	sbuf.append("{\\small " + deptGakunen + "} \\\\ \n");
	sbuf.append("{\\small " + semester + "  " + weekHour + " " + room + " } \n");
      }
    
      if (!map.containsKey(param)) {
	map.put(param, "1");
	appendMainText(sbuf, param);
      }
      
      sbuf.append("\\end{quotation} \n");
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  public void appendSubsection3(StringBuffer sbuf, ArrayList<Object> list, 
				String param, HashMap<String, String> map) {
    try {
      String code;
      String subjectCode = (String)list.get(0);
      String teacherCode = (String)list.get(1);
      String kubunCode = (String)list.get(2);
      String reqCode   = (String)list.get(3);     
      String unit = (String)list.get(4) + "単位";
      String subjectName = (String)list.get(5);
      String englishName = (String)list.get(6);
      String teacherName = (String)list.get(7);      
      code  = (String)list.get(8);
      String staffAttrib = commonInfo.getGakumuCodeName("STAFF_ATTRIB", code);
      code  = (String)list.get(9);
      if (code.equals("3")) {
	staffAttrib = "非常勤教員";
      }
      String mailAddress = (String)list.get(10); 
      String kubun = commonInfo.getGakumuCodeName("KUBUN_CODE", kubunCode);
      String req = commonInfo.getGakumuCodeName("REQ_CODE", reqCode);

      sbuf.append("\\subsection{ "+subjectName+" \\hspace{2zw} \\dotfill \\hspace{2zw} "+kubun+" \\hspace{1zw}$\\ldots\\ldots$\\hspace{1zw} "+teacherName+" }\n");
      sbuf.append("\\begin{quotation}\n");
      sbuf.append("\\noindent {\\small 科目名:　" + subjectName + " \\hspace{0.3cm} " + englishName + " } \\\\ \n");
      sbuf.append("{\\small 担当教員:　" + teacherName + "  (" + staffAttrib + ")  \\verb+  " + mailAddress + " +}\\\\ \n");
      sbuf.append("{\\small "+kubun+" \\hspace{0.1cm} " + req + " " + unit + " \\hspace{0.1cm} }\\\\ \n");
      sbuf.append("{\\small ( 本年度は休講です ) } \n");
    
      if (!map.containsKey(param)) {
	map.put(param, "1");
	appendMainText(sbuf, param);
      }
      
      sbuf.append("\\end{quotation} \n");
    } catch (Exception e) {
      e.printStackTrace();
    }
  }


  public String convertDeptGakunen(String dg) {
    StringTokenizer stk = new StringTokenizer(dg, "|");
    String dept = stk.nextToken();
    String gaku = stk.nextToken();
    int d = Integer.parseInt(dept);
    int g = Integer.parseInt(gaku);
    switch (d) { 
    case 31:
      return "知能情報" + g + "年 ";
    case 32:
      return "電子情報" + g + "年 ";
    case 33:
      return "システム創成" + g + "年 ";
    case 34:
      return "機械情報" + g + "年 ";
    case 35:
      return "生命情報" + g + "年 ";
    default:
      return dept + "学科" + g + "年 ";
    }
  }      

  public String convertWeekHour(String wh) {
    StringTokenizer stk = new StringTokenizer(wh, "|");
    String week = stk.nextToken();
    String hour = stk.nextToken();
    int w = Integer.parseInt(week);
    int h = Integer.parseInt(hour);
    switch (w) {     
    case 0:
      return "集中講義等 ";
    case 1:
      return "月曜" + h + "限目 ";
    case 2:
      return "火曜" + h + "限目 ";
    case 3:
      return "水曜" + h + "限目 ";
    case 4:
      return "木曜" + h + "限目 ";
    case 5:
      return "金曜" + h + "限目 ";
    case 6:
      return "土曜" + h + "限目 ";
    default:
      return "？曜" + h + "限目 ";
    }
  }      


  public void appendMainText(StringBuffer sbuf, String param) {  
    StringTokenizer stk = new StringTokenizer(param, "|");
    String schoolYear = stk.nextToken();
    String subjectCode = stk.nextToken();
    String teacherCode = stk.nextToken();
  
    org.w3c.dom.Element element = null;
    String xmltext = commonMethods.getSyllabusXml(schoolYear, subjectCode, teacherCode);
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
      if (syllabusErrorLog != null) {
	syllabusErrorLog.println("ParserConfigurationException:");
	syllabusErrorLog.println("  " + param);
      } else {
	System.out.println("ParserConfigurationException");
      }
    } catch(SAXException e) {
      if (syllabusErrorLog != null) {
	syllabusErrorLog.println("SAXException:");
	syllabusErrorLog.println("  " + param);
      } else {
	System.out.println("SAXException");
      }      
    } catch(IOException e) {
      if (syllabusErrorLog != null) {
	syllabusErrorLog.println("IOException:");
	syllabusErrorLog.println("  " + param);
      } else {
	System.out.println("IOException");
      }  
    }      
    if (element == null) {
      if (syllabusErrorLog != null) {
	syllabusErrorLog.println("該当なし:");
	syllabusErrorLog.println("  " + param);
      } else {
	System.out.println("該当なし");
	return;
      }
    }      
    DefaultMutableTreeNode root = UFile.makeJTTree(element);

    JLatexMakerJT2 maker = new JLatexMakerJT2();      
    UJTVisitor.traverse(root, maker);
    String text = maker.getText(); 
    sbuf.append(text);
  }


  public void makeHtmlSyllabusBody(ArrayList<Object> list, 
				   String param, HashMap<String, String> map, File ofile) {
    StringBuffer sbuf = new StringBuffer();
    try {
      PrintWriter fout;    
      fout = new PrintWriter(new BufferedWriter(new FileWriter(ofile)));

      String code;
      String subjectCode = (String)list.get(0);
      String classCode   = (String)list.get(1);
      String teacherCode = (String)list.get(2);
      String gakunen   = (String)list.get(3);
      
      code = (String)list.get(4);
      String semester;
      if (code.trim().equals("")) {
	semester = "(本年度休講)";
      } else {
	switch (Integer.parseInt(code)) {
	case 1:
	  semester = "前期"; break;
	case 2:
	  semester = "後期"; break;
	case 3:
	  semester = "通年"; break;
	default:
	  semester = "(休講)"; break;
	}
      }
      code = (String)list.get(5);
      String kubun = commonInfo.getGakumuCodeName("KUBUN_CODE", code);
      code = (String)list.get(6);
      String req = commonInfo.getGakumuCodeName("REQ_CODE", code);
      code = (String)list.get(7);
      String unit = code + "単位";
      code = (String)list.get(8);
      String room = commonInfo.getGakumuCodeName("ROOM", code);
      String subjectName = (String)list.get(9);
      String englishName = (String)list.get(10);
      String teacherName = (String)list.get(11);
      
      code  = (String)list.get(12);
      String staffAttrib = commonInfo.getGakumuCodeName("STAFF_ATTRIB", code);
      code  = (String)list.get(13);
      if (code.equals("3")) {
	staffAttrib = "非常勤教員";
      }
      String mailAddress = (String)list.get(14);
      String weekHour = "";
      for (int i = 15; i < list.size(); i++) {
	String wh = (String)list.get(i);
	weekHour = weekHour + convertWeekHour(wh);
      }    
       
      sbuf.append("<HTML><body>").append(lineSeparator); 
      sbuf.append("<H3 align=\"left\"><font color=\"blue\">" + subjectName + "</font></H3>").append(lineSeparator);

      sbuf.append("<PRE>").append(lineSeparator);
      sbuf.append("  科目名: " + subjectName + " (" + classCode + ")　" + englishName).append(lineSeparator);
      sbuf.append("  担当教員: " + teacherName + " (" + staffAttrib + ")　" + mailAddress).append(lineSeparator);
      sbuf.append("            " + kubun+" " + req + " " + unit).append(lineSeparator);
      sbuf.append("            " + gakunen + "年 " + semester + " " + weekHour + " " + room).append(lineSeparator);
      sbuf.append("</PRE>").append(lineSeparator);
 
      StringTokenizer stk = new StringTokenizer(param, "|");
      String schoolYear = stk.nextToken();
      subjectCode = stk.nextToken();
      teacherCode = stk.nextToken();
      
      org.w3c.dom.Element element = null;
      String xmltext = commonMethods.getSyllabusXml(schoolYear, subjectCode, teacherCode);
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
	if (syllabusErrorLog != null) {
	  syllabusErrorLog.println("ParserConfigurationException:");
	  syllabusErrorLog.println("  " + param);
	} else {
	  System.out.println("ParserConfigurationException");
	}
      } catch(SAXException e) {
	if (syllabusErrorLog != null) {
	  syllabusErrorLog.println("SAXException:");
	  syllabusErrorLog.println("  " + param);
	} else {
	  System.out.println("SAXException");
	}      
      } catch(IOException e) {
	if (syllabusErrorLog != null) {
	  syllabusErrorLog.println("IOException:");
	  syllabusErrorLog.println("  " + param);
	} else {
	  System.out.println("IOException");
	}  
      }      
      if (element == null) {
	if (syllabusErrorLog != null) {
	  syllabusErrorLog.println("該当なし:");
	  syllabusErrorLog.println("  " + param);
	} else {
	  System.out.println("該当なし");
	  return;
	}
      }      
      DefaultMutableTreeNode root = UFile.makeJTTree(element);
      
      HTMLMakerJT3 maker = new HTMLMakerJT3();      
      UJTVisitor.traverse(root, maker);
      String text = maker.getText(); 
      sbuf.append(text).append(lineSeparator);
      
      sbuf.append("</body></HTML>").append(lineSeparator);
      fout.println(sbuf.toString());
      fout.close();
    } catch (Exception e) {
      e.printStackTrace();
    }
  }



  public void makeHtmlSyllabusBody2(ArrayList<Object> list,
				    String param, HashMap<String, String> map, File ofile) {
    StringBuffer sbuf = new StringBuffer();
    try {
      PrintWriter fout;    
      fout = new PrintWriter(new BufferedWriter(new FileWriter(ofile)));

      String code;
      String subjectCode = (String)list.get(0);
      String classCode   = (String)list.get(1);
      String teacherCode = (String)list.get(2);
      String gakunen   = (String)list.get(3);
      
      code = (String)list.get(4);
      String semester;
      if (code.trim().equals("")) {
	semester = "(本年度休講)";
      } else {
	switch (Integer.parseInt(code)) {
	case 1:
	  semester = "前期"; break;
	case 2:
	  semester = "後期"; break;
	case 3:
	  semester = "通年"; break;
	default:
	  semester = "(休講)"; break;
	}
      }
      code = (String)list.get(5);
      String kubun = commonInfo.getGakumuCodeName("KUBUN_CODE", code);
      code = (String)list.get(6);
      String req = commonInfo.getGakumuCodeName("REQ_CODE", code);
      code = (String)list.get(7);
      String unit = code + "単位";
      code = (String)list.get(8);
      String room = commonInfo.getGakumuCodeName("ROOM", code);
      String subjectName = (String)list.get(9);
      String englishName = (String)list.get(10);
      String teacherName = (String)list.get(11);
      
      code  = (String)list.get(12);
      String staffAttrib = commonInfo.getGakumuCodeName("STAFF_ATTRIB", code);
      code  = (String)list.get(13);
      if (code.equals("3")) {
	staffAttrib = "非常勤教員";
      }
      String mailAddress = (String)list.get(14);

      String deptGakunen = "";
      int deptCount = 0;
      String weekHour = "";
      
      for (int i = 15; i < list.size(); i++) {
	String tmp = (String)list.get(i);
	if (tmp.length() >= 5) {
	  deptGakunen = deptGakunen + convertDeptGakunen(tmp);
	  deptCount++;
	} else {
	  weekHour = weekHour + convertWeekHour(tmp);
	}  
      }
      if (deptCount >= 5) {
	deptCount = 1;
	deptGakunen = "共通科目 " + gakunen + "年 ";
      }   
       
      sbuf.append("<HTML><body>").append(lineSeparator); 
      sbuf.append("<H3 align=\"left\"><font color=\"blue\">" + subjectName + "</font></H3>").append(lineSeparator);

      sbuf.append("<PRE>").append(lineSeparator);
      sbuf.append("  科目名: " + subjectName + " (" + classCode + ")　" + englishName).append(lineSeparator);
      sbuf.append("  担当教員: " + teacherName + " (" + staffAttrib + ")　" + mailAddress).append(lineSeparator);
      sbuf.append("            " + kubun + " " + req + " " + unit).append(lineSeparator);

      if (deptCount == 1) {
	sbuf.append("            " + deptGakunen + " " + semester + " " + weekHour + " " + room).append(lineSeparator);
      } else {
	if (!deptGakunen.equals("")) {
	  sbuf.append("            " + deptGakunen).append(lineSeparator);
	}
	sbuf.append("            " + semester + " " + weekHour + " " + room).append(lineSeparator);
      }
      sbuf.append("</PRE>").append(lineSeparator);
 
      StringTokenizer stk = new StringTokenizer(param, "|");
      String schoolYear = stk.nextToken();
      subjectCode = stk.nextToken();
      teacherCode = stk.nextToken();
      
      org.w3c.dom.Element element = null;
      String xmltext = commonMethods.getSyllabusXml(schoolYear, subjectCode, teacherCode);
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
	if (syllabusErrorLog != null) {
	  syllabusErrorLog.println("ParserConfigurationException:");
	  syllabusErrorLog.println("  " + param);
	} else {
	  System.out.println("ParserConfigurationException");
	}
      } catch(SAXException e) {
	if (syllabusErrorLog != null) {
	  syllabusErrorLog.println("SAXException:");
	  syllabusErrorLog.println("  " + param);
	} else {
	  System.out.println("SAXException");
	}      
      } catch(IOException e) {
	if (syllabusErrorLog != null) {
	  syllabusErrorLog.println("IOException:");
	  syllabusErrorLog.println("  " + param);
	} else {
	  System.out.println("IOException");
	}  
      }      
      if (element == null) {
	if (syllabusErrorLog != null) {
	  syllabusErrorLog.println("該当なし:");
	  syllabusErrorLog.println("  " + param);
	} else {
	  System.out.println("該当なし");
	  return;
	}
      }      
      DefaultMutableTreeNode root = UFile.makeJTTree(element);
      
      HTMLMakerJT3 maker = new HTMLMakerJT3();      
      UJTVisitor.traverse(root, maker);
      String text = maker.getText(); 
      sbuf.append(text).append(lineSeparator);
      
      sbuf.append("</body></HTML>").append(lineSeparator);
      fout.println(sbuf.toString());
      fout.close();
    } catch (Exception e) {
      e.printStackTrace();
    }
  }




  public void appendHtmlHead(StringBuffer sbuf, String dept) {    
    sbuf.append("<HTML><body>").append(lineSeparator); 
    sbuf.append("<H3> " + dept + "</H3>").append(lineSeparator);
    sbuf.append("<UL>").append(lineSeparator);
  }

  public void appendHtmlSection(StringBuffer sbuf, String schoolYear, String deptName, int gakunen) {
    sbuf.append("<P>").append(lineSeparator);
    sbuf.append("<LI> " + schoolYear + "年度　" + deptName + "　" + gakunen + "年次").append(lineSeparator);
    sbuf.append("<P>").append(lineSeparator);
    sbuf.append("<UL>").append(lineSeparator);
  }

  public void appendHtmlSection2(StringBuffer sbuf, String schoolYear, String kubunName) {
    sbuf.append("<P>").append(lineSeparator);
    sbuf.append("<LI> " + schoolYear + "年度　" + kubunName  + " (学部共通科目)").append(lineSeparator);
    sbuf.append("<P>").append(lineSeparator);
    sbuf.append("<UL>").append(lineSeparator);
  }

  ////
  public void appendHtmlSection3(StringBuffer sbuf, String schoolYear, String kubunName) {
    sbuf.append("<P>").append(lineSeparator);
    sbuf.append("<LI> " + schoolYear + "年度　" + kubunName + " (大学院科目)").append(lineSeparator);
    sbuf.append("<P>").append(lineSeparator);
    sbuf.append("<UL>").append(lineSeparator);
  }

  public void appendHtmlSection4(StringBuffer sbuf, String schoolYear, String deptName, String kubunName) {
    sbuf.append("<P>").append(lineSeparator);
    sbuf.append("<LI> " + schoolYear + "年度　" + deptName + " " + kubunName + " (大学院科目)").append(lineSeparator);
    sbuf.append("<P>").append(lineSeparator);
    sbuf.append("<UL>").append(lineSeparator);
  }



  public void appendHtmlSubsection(StringBuffer sbuf, ArrayList<Object> list, 
				   String param, HashMap<String, String> map, String ofname) {
      String code;
      String subjectCode = (String)list.get(0);
      String classCode   = (String)list.get(1);
      String teacherCode = (String)list.get(2);
      String gakunen   = (String)list.get(3);
      
      code = (String)list.get(4);
      String semester;
      if (code.trim().equals("")) {
	semester = "(本年度休講)";
      } else {
	switch (Integer.parseInt(code)) {
	case 1:
	  semester = "前期"; break;
	case 2:
	  semester = "後期"; break;
	case 3:
	  semester = "通年"; break;
	default:
	  semester = "(休講)"; break;
	}
      }
      code = (String)list.get(5);
      String kubun = commonInfo.getGakumuCodeName("KUBUN_CODE", code);
      code = (String)list.get(6);
      String req = commonInfo.getGakumuCodeName("REQ_CODE", code);
      code = (String)list.get(7);
      String unit = code + "単位";
      code = (String)list.get(8);
      String room = commonInfo.getGakumuCodeName("ROOM", code);
      String subjectName = (String)list.get(9);
      String englishName = (String)list.get(10);
      String teacherName = (String)list.get(11);
            
      sbuf.append("<LI> <a href="+ofname+"> "+subjectName+" ("+teacherName+")  </a>").append(lineSeparator);
  }

  private String[][] undergradDeptInfo = { { "31", "10", "AI" },
					   { "32", "10", "ELEC" },
					   { "33", "10", "CONTROL" },
					   { "34", "10", "MECH" },
					   { "35", "10", "BIO" } };

  private String[][] undergradKubunInfo = { { "30", "10", "440", "HUMAN_INFO" },
					    { "30", "10", "444", "HUMAN_INTRO" },
					    { "30", "10", "445", "HUMAN_LECT" },
					    { "30", "10", "446", "ENGLISH" },
					    { "30", "10", "447", "LANGUAGE" },
					    { "30", "10", "448", "HEALTH" },
					    { "30", "10", "636", "COMMON_INFO" },
					    { "30", "10", "637", "COMMON_OBJ" },
					    { "30", "10", "991", "EDU_K" },
					    { "30", "10", "992", "EDU_S" },
					    { "30", "10", "627", "IIF" } }; 

  public void makeUndergradSyllabusLatex() {  
    String homeDir = System.getProperty("user.home");
    File dir = new File(homeDir, "デスクトップ");   
    File dir1 = new File(dir, "CD-SYLLABUS");  
    File dir2 = new File(dir1, "SyllabusPDF");  

    HashSet<String> subjectSet = new HashSet<String>();

    StringTokenizer stk, sstk;
    PrintWriter fout;
    String schoolYear = tabbedPane.getValueFromColumnCodeMap("SCHOOL_YEAR");
        
    for (String[] deptInfo : undergradDeptInfo) {
      String faculty = "11";
      String department = deptInfo[0];
      String course = deptInfo[1];
      String fname = deptInfo[2] + ".tex";
      File f = new File(dir2, fname);

      String facName = commonInfo.getGakumuCodeName("FACULTY", faculty);
      String deptName = commonInfo.getGakumuCodeName("DEPARTMENT", department);

      try {
	fout = new PrintWriter(new BufferedWriter(new FileWriter(f)));
      } catch (Exception e) {
	commonInfo.showMessage(e.toString());
	return;
      }

      StringBuffer sbuf = new StringBuffer();
      appendHead(sbuf, deptName);
    
      for (int gakunen = 1; gakunen <= 4; gakunen++) {
	appendSection(sbuf, schoolYear, deptName, gakunen);

	ArrayList<ArrayList<Object>> rows = new ArrayList<ArrayList<Object>>();
	String key = "QUERY|COMMON_QUERY|querySyllabusSubjectInfo";
	String paramValues = schoolYear+"|"+faculty+"|"+department+"|"+course+"|"+gakunen; 
	String answer = serverConn.queryCommon(key, paramValues);
	stk = new StringTokenizer(answer, "$");
	while (stk.hasMoreTokens()) {
	  ArrayList<Object> list = new ArrayList<Object>();
	  sstk = new StringTokenizer(stk.nextToken(), "|");
	  while (sstk.hasMoreTokens()) {
	    list.add(sstk.nextToken());
	  }
	  rows.add(list);
	}

	HashMap<String, String> map = new HashMap<String, String>();

	for (int i = 0; i < rows.size(); i++) {
	  try {
	    String subjectCode = (String)(rows.get(i)).get(0);
	    String classCode = (String)(rows.get(i)).get(1);
	    String teacherCode = (String)(rows.get(i)).get(2);
	    String weekHour;

	    key = "QUERY|COMMON_QUERY|querySyllabusWeekHour";
	    paramValues = schoolYear+"|"+subjectCode+"|"+classCode;
	    subjectSet.add(paramValues);
	    answer = serverConn.queryCommon(key, paramValues);
	    if (answer != null) {
	      stk = new StringTokenizer(answer, "$");
	      while (stk.hasMoreTokens()) {
		weekHour = stk.nextToken();
		(rows.get(i)).add(weekHour);
	      }
	      
	      paramValues = schoolYear+"|"+subjectCode+"|"+teacherCode;
	      appendSubsection(sbuf, rows.get(i), paramValues, map);
	      fout.println(sbuf.toString());  
	      sbuf = new StringBuffer(); 
	    } else {
	      System.out.println("Jikanwari is null: " + teacherCode+"_"+subjectCode+"_"+classCode);
	    }
	  } catch (Exception e) {
	    e.printStackTrace();
	  }
	}
      }
      fout.println(" \\end{document}");
      fout.close();
    }

    for (String[] kubunInfo : undergradKubunInfo) {
      String faculty = "11";
      String department = kubunInfo[0];
      String course = kubunInfo[1];
      String kubun = kubunInfo[2];
      String fname = kubunInfo[3] + ".tex";
      File f = new File(dir2, fname);

      String facName = commonInfo.getGakumuCodeName("FACULTY", faculty);
      String deptName = commonInfo.getGakumuCodeName("DEPARTMENT", department);
      String kubunName = commonInfo.getGakumuCodeName("KUBUN_CODE", kubun);

      try {
	fout = new PrintWriter(new BufferedWriter(new FileWriter(f)));
      } catch (Exception e) {
	commonInfo.showMessage(e.toString());
	return;
      }
      StringBuffer sbuf = new StringBuffer();
      appendHead(sbuf, deptName + " " + kubunName);
    
      appendSection2(sbuf, schoolYear, kubunName);
      
      ArrayList<ArrayList<Object>> rows = new ArrayList<ArrayList<Object>>();
      String paramValues = schoolYear+"|"+faculty+"|"+course+"|"+kubun;
      String key = "QUERY|COMMON_QUERY|querySyllabusKubunSubjectInfo";
      String answer = serverConn.queryCommon(key, paramValues);// 担当教員が設定されていない科目は除外
      stk = new StringTokenizer(answer, "$");
      while (stk.hasMoreTokens()) {
	ArrayList<Object> list = new ArrayList<Object>();
	sstk = new StringTokenizer(stk.nextToken(), "|");
	while (sstk.hasMoreTokens()) {
	  list.add(sstk.nextToken());
	}
	rows.add(list);
      }
      
      HashMap<String, String> map = new HashMap<String, String>();
      
      for (int i = 0; i < rows.size(); i++) {
	try {
	  String subjectCode = (String)(rows.get(i)).get(0);
	  String classCode = (String)(rows.get(i)).get(1);
	  String teacherCode = (String)(rows.get(i)).get(2);
	  String weekHour, deptGakunen;
	  
	  paramValues = schoolYear+"|"+subjectCode+"|"+classCode;
	  if (!subjectSet.contains(paramValues)) {
	    key = "QUERY|COMMON_QUERY|querySyllabusDeptGakunen";
	    answer = serverConn.queryCommon(key, paramValues);
	    if (answer != null) {
	      stk = new StringTokenizer(answer, "$");
	      while (stk.hasMoreTokens()) {
		deptGakunen = stk.nextToken();
		(rows.get(i)).add(deptGakunen);
	      }
	    }
	    
	    paramValues = schoolYear+"|"+subjectCode+"|"+classCode;
	    key = "QUERY|COMMON_QUERY|querySyllabusWeekHour";
	    answer = serverConn.queryCommon(key, paramValues);
	    if (answer != null) {
	      stk = new StringTokenizer(answer, "$");
	      while (stk.hasMoreTokens()) {
		weekHour = stk.nextToken();
		(rows.get(i)).add(weekHour);
	      }
	    }
	    
	    paramValues = schoolYear+"|"+subjectCode+"|"+teacherCode;
	    appendSubsection2(sbuf, rows.get(i), paramValues, map);
	    fout.println(sbuf.toString());  
	    sbuf = new StringBuffer(); 
	  }
	} catch (Exception e) {
	  e.printStackTrace();
	}
      }
      fout.println(" \\end{document}");
      fout.close();
    }
  }


  public void makeUndergradSyllabusHtml() {   
    String homeDir = System.getProperty("user.home");
    File dir = new File(homeDir, "デスクトップ");    
    File dir1 = new File(dir, "CD-SYLLABUS");  
    File dir2 = new File(dir1, "SyllabusHTML");  
    String ofname;

    HashSet<String> subjectSet = new HashSet<String>();

    StringTokenizer stk, sstk;
    PrintWriter fout;
    String schoolYear = tabbedPane.getValueFromColumnCodeMap("SCHOOL_YEAR");
       
    for (String[] deptInfo : undergradDeptInfo) {
      String faculty = "11";
      String department = deptInfo[0];
      String course = deptInfo[1];
      String fname = deptInfo[2] + ".html";
      File f = new File(dir2, fname);

      String facName = commonInfo.getGakumuCodeName("FACULTY", faculty);
      String deptName = commonInfo.getGakumuCodeName("DEPARTMENT", department);

      try {
	fout = new PrintWriter(new BufferedWriter(new FileWriter(f)));
      } catch (Exception e) {
	commonInfo.showMessage(e.toString());
	return;
      }
 
      StringBuffer sbuf = new StringBuffer();
      appendHtmlHead(sbuf, deptName);
      
      for (int gakunen = 1; gakunen <= 4; gakunen++) {
	appendHtmlSection(sbuf, schoolYear, deptName, gakunen);
	
	ArrayList<ArrayList<Object>> rows = new ArrayList<ArrayList<Object>>();
	String paramValues = schoolYear+"|"+faculty+"|"+department+"|"+course+"|"+gakunen; 
	String key = "QUERY|COMMON_QUERY|querySyllabusSubjectInfo";
	String answer = serverConn.queryCommon(key, paramValues);
	if (answer != null) {
	  stk = new StringTokenizer(answer, "$");
	  while (stk.hasMoreTokens()) {
	    ArrayList<Object> list = new ArrayList<Object>();
	    sstk = new StringTokenizer(stk.nextToken(), "|");
	    while (sstk.hasMoreTokens()) {
	      list.add(sstk.nextToken());
	    }
	    rows.add(list);
	  }
	
	  HashMap<String, String> map = new HashMap<String, String>();
	
	  for (int i = 0; i < rows.size(); i++) {
	    try {
	      String subjectCode = (String)(rows.get(i)).get(0);
	      String classCode = (String)(rows.get(i)).get(1);
	      String teacherCode = (String)(rows.get(i)).get(2);
	      String weekHour;
	      
	      paramValues = schoolYear+"|"+subjectCode+"|"+classCode;
	      subjectSet.add(paramValues);
	      key = "QUERY|COMMON_QUERY|querySyllabusWeekHour";
	      answer = serverConn.queryCommon(key, paramValues);
	      if (answer != null) {
		stk = new StringTokenizer(answer, "$");
		while (stk.hasMoreTokens()) {
		  weekHour = stk.nextToken();
		  (rows.get(i)).add(weekHour);
		}
		
		paramValues = schoolYear+"|"+subjectCode+"|"+teacherCode;
		ofname = schoolYear+"-"+subjectCode+"-"+teacherCode+".html";
		File ofile = new File(dir2, ofname);
		appendHtmlSubsection(sbuf, rows.get(i), paramValues, map, ofname);
		fout.println(sbuf.toString());  
		sbuf = new StringBuffer(); 
		makeHtmlSyllabusBody(rows.get(i), paramValues, map, ofile);
	      } else {
		System.out.println("Jikanwari is null: " + teacherCode+"_"+subjectCode+"_"+classCode);
	      }
	    } catch (Exception e) {
	      e.printStackTrace();
	    }
	  }
	}
	fout.println("</UL>");
      }
      fout.println("</UL></body></HTML>");
      fout.close();
    }
    
    for (String[] kubunInfo : undergradKubunInfo) {
      String faculty = "11";
      String department = kubunInfo[0];
      String course = kubunInfo[1];
      String kubun = kubunInfo[2];
      String fname = kubunInfo[3] + ".html";
      File f = new File(dir2, fname);
      
      String facName = commonInfo.getGakumuCodeName("FACULTY", faculty);
      String deptName = commonInfo.getGakumuCodeName("DEPARTMENT", department);
      String kubunName = commonInfo.getGakumuCodeName("KUBUN_CODE", kubun);
      
      try {
	fout = new PrintWriter(new BufferedWriter(new FileWriter(f)));
      } catch (Exception e) {
	commonInfo.showMessage(e.toString());
	return;
      }
      StringBuffer sbuf = new StringBuffer();
      appendHtmlHead(sbuf, deptName + " " + kubunName);
      
      appendHtmlSection2(sbuf, schoolYear, kubunName);
      
      ArrayList<ArrayList<Object>> rows = new ArrayList<ArrayList<Object>>();
      String paramValues = schoolYear+"|"+faculty+"|"+course+"|"+kubun;
      String key = "QUERY|COMMON_QUERY|querySyllabusKubunSubjectInfo";
      String answer = serverConn.queryCommon(key, paramValues);
      if (answer != null) {
	stk = new StringTokenizer(answer, "$");
	while (stk.hasMoreTokens()) {
	  ArrayList<Object> list = new ArrayList<Object>();
	  sstk = new StringTokenizer(stk.nextToken(), "|");
	  while (sstk.hasMoreTokens()) {
	    list.add(sstk.nextToken());
	  }
	  rows.add(list);
	}
	
	HashMap<String, String> map = new HashMap<String, String>();
	
	for (int i = 0; i < rows.size(); i++) {
	  String subjectCode = (String)(rows.get(i)).get(0);
	  String classCode = (String)(rows.get(i)).get(1);
	  String teacherCode = (String)(rows.get(i)).get(2);
	  String weekHour, deptGakunen;
	  
	  paramValues = schoolYear+"|"+subjectCode+"|"+classCode;
	  if (!subjectSet.contains(paramValues)) {
	    key = "QUERY|COMMON_QUERY|querySyllabusDeptGakunen";
	    answer = serverConn.queryCommon(key, paramValues);
	    if (answer != null) {
	      stk = new StringTokenizer(answer, "$");
	      while (stk.hasMoreTokens()) {
		deptGakunen = stk.nextToken();
		(rows.get(i)).add(deptGakunen);
	      }
	    }
	    
	    paramValues = schoolYear+"|"+subjectCode+"|"+classCode;
	    key = "QUERY|COMMON_QUERY|querySyllabusWeekHour";
	    answer = serverConn.queryCommon(key, paramValues);
	    if (answer != null) {
	      stk = new StringTokenizer(answer, "$");
	      while (stk.hasMoreTokens()) {
		weekHour = stk.nextToken();
		(rows.get(i)).add(weekHour);
	      }
	    }
	    
	    paramValues = schoolYear+"|"+subjectCode+"|"+teacherCode;
	    ofname = schoolYear+"-"+subjectCode+"-"+teacherCode+".html";
	    File ofile = new File(dir2, ofname);
	    appendHtmlSubsection(sbuf, rows.get(i), paramValues, map, ofname);  //
	    fout.println(sbuf.toString());  
	    sbuf = new StringBuffer(); 
	    makeHtmlSyllabusBody2(rows.get(i), paramValues, map, ofile);
	  }
	}
	fout.println("</UL>");
      }
      fout.println("</UL></body></HTML>");
      fout.close();
    }
  }

  private String[][] graduateKubunInfo = { { "32", "70", "00", "583", "GRAD_KISO" },
					   { "32", "70", "00", "584", "GRAD_INFO" },
					   { "32", "73", "00", "585", "GRAD_SCI_OBJ" },
					   { "32", "74", "00", "585", "GRAD_SYS_OBJ" },
					   { "32", "75", "00", "585", "GRAD_SOSEI_OBJ" },
					   { "32", "75", "90", "586", "GRAD_SOSEI_KISO" },
					   { "32", "70", "00", "997", "GRAD_EDU" } };


  public void makeGraduateSyllabusLatex() { 
    String homeDir = System.getProperty("user.home");
    File dir = new File(homeDir, "デスクトップ");  
    File dir1 = new File(dir, "CD-SYLLABUS");  
    File dir2 = new File(dir1, "SyllabusPDF");  
 
    StringTokenizer stk, sstk;
    String paramValues;
    String key;
    String answer;
    PrintWriter fout;
    String schoolYear = tabbedPane.getValueFromColumnCodeMap("SCHOOL_YEAR");
      
    for (String[] kubunInfo : graduateKubunInfo) {
      String faculty = kubunInfo[0];
      String department = kubunInfo[1];
      String course = kubunInfo[2];
      String kubun = kubunInfo[3];
      String fname = kubunInfo[4] + ".tex";
      File f = new File(dir2, fname);
      
      String facName = commonInfo.getGakumuCodeName("FACULTY", faculty);
      String deptName = commonInfo.getGakumuCodeName("DEPARTMENT", department);
      String kubunName = commonInfo.getGakumuCodeName("KUBUN_CODE", kubun);
      
      try {
	fout = new PrintWriter(new BufferedWriter(new FileWriter(f)));
      } catch (Exception e) {
	commonInfo.showMessage(e.toString());
	return;
      }

      StringBuffer sbuf = new StringBuffer();
      appendHead(sbuf, kubunName);
    
      appendSection4(sbuf, schoolYear, deptName, kubunName);

      if (kubun.equals("585")) {   // 大学院対象分野
	paramValues = schoolYear+"|"+faculty+"|"+department+"|"+kubun;	
	key = "QUERY|COMMON_QUERY|querySyllabusKubunSubjectInfo3";
	answer = serverConn.queryCommon(key, paramValues);
      } else {
	paramValues = schoolYear+"|"+faculty+"|"+course+"|"+kubun;
	key = "QUERY|COMMON_QUERY|querySyllabusKubunSubjectInfo2";
	answer = serverConn.queryCommon(key, paramValues);
      }
      ArrayList<ArrayList<Object>> rows = new ArrayList<ArrayList<Object>>();
  
      if (answer != null) {
	stk = new StringTokenizer(answer, "$");
	while (stk.hasMoreTokens()) {
	  ArrayList<Object> list = new ArrayList<Object>();
	  sstk = new StringTokenizer(stk.nextToken(), "|");
	  while (sstk.hasMoreTokens()) {
	    list.add(sstk.nextToken());
	  }
	  rows.add(list);
	}
	
	HashMap<String, String> map = new HashMap<String, String>();
            
	for (int i = 0; i < rows.size(); i++) { // 担当教員が設定されている科目
	  try {
	    String subjectCode = (String)(rows.get(i)).get(0);
	    String classCode = (String)(rows.get(i)).get(1);
	    String teacherCode = (String)(rows.get(i)).get(2);
	    String weekHour;
	    
	    paramValues = schoolYear+"|"+subjectCode+"|"+classCode;
	    key = "QUERY|COMMON_QUERY|querySyllabusWeekHour";
	    answer = serverConn.queryCommon(key, paramValues);
	    if (answer != null) {
	      stk = new StringTokenizer(answer, "$");
	      while (stk.hasMoreTokens()) {
		weekHour = stk.nextToken();
		(rows.get(i)).add(weekHour);
	      }
	    }
	    
	    paramValues = schoolYear+"|"+subjectCode+"|"+teacherCode;
	    appendSubsection(sbuf, rows.get(i), paramValues, map);
	    fout.println(sbuf.toString());  
	    sbuf = new StringBuffer(); 
	  } catch (Exception e) {
	    e.printStackTrace();
	  }
	}
      }
      fout.println(" \\end{document}");
      fout.close();      
    }
  }

  public void makeGraduateSyllabusHtml() {  
    String homeDir = System.getProperty("user.home");
    File dir = new File(homeDir, "デスクトップ");  
    File dir1 = new File(dir, "CD-SYLLABUS");  
    File dir2 = new File(dir1, "SyllabusHTML");  
    String ofname;
    String paramValues;
    String key;
    String answer;

    StringTokenizer stk, sstk;
    PrintWriter fout;
    String schoolYear = tabbedPane.getValueFromColumnCodeMap("SCHOOL_YEAR");
        
    for (String[] kubunInfo : graduateKubunInfo) {
      String faculty = kubunInfo[0];
      String department = kubunInfo[1];
      String course = kubunInfo[2];
      String kubun = kubunInfo[3];
      String fname = kubunInfo[4] + ".html";
      File f = new File(dir2, fname);
      
      String facName = commonInfo.getGakumuCodeName("FACULTY", faculty);
      String deptName = commonInfo.getGakumuCodeName("DEPARTMENT", department);
      String kubunName = commonInfo.getGakumuCodeName("KUBUN_CODE", kubun);
      
      try {
	fout = new PrintWriter(new BufferedWriter(new FileWriter(f)));
      } catch (Exception e) {
	commonInfo.showMessage(e.toString());
	return;
      }

      StringBuffer sbuf = new StringBuffer();
      appendHtmlHead(sbuf, kubunName);
    
      appendHtmlSection4(sbuf, schoolYear, deptName, kubunName);

      if (kubun.equals("585")) {   // 大学院対象分野
	paramValues = schoolYear+"|"+faculty+"|"+department+"|"+kubun;	
	key = "QUERY|COMMON_QUERY|querySyllabusKubunSubjectInfo3";
	answer = serverConn.queryCommon(key, paramValues);
      } else {
	paramValues = schoolYear+"|"+faculty+"|"+course+"|"+kubun;
	key = "QUERY|COMMON_QUERY|querySyllabusKubunSubjectInfo2";
	answer = serverConn.queryCommon(key, paramValues);
      }
      ArrayList<ArrayList<Object>> rows = new ArrayList<ArrayList<Object>>();
  
      if (answer != null) {
	stk = new StringTokenizer(answer, "$");
	while (stk.hasMoreTokens()) {
	  ArrayList<Object> list = new ArrayList<Object>();
	  sstk = new StringTokenizer(stk.nextToken(), "|");
	  while (sstk.hasMoreTokens()) {
	    list.add(sstk.nextToken());
	  }
	  rows.add(list);
	}
	
	HashMap<String, String> map = new HashMap<String, String>();
            
	for (int i = 0; i < rows.size(); i++) { // 担当教員が設定されている科目
	  try {
	    String subjectCode = (String)(rows.get(i)).get(0);
	    String classCode = (String)(rows.get(i)).get(1);
	    String teacherCode = (String)(rows.get(i)).get(2);
	    String weekHour;
	    
	    paramValues = schoolYear+"|"+subjectCode+"|"+classCode;
	    key = "QUERY|COMMON_QUERY|querySyllabusWeekHour";
	    answer = serverConn.queryCommon(key, paramValues);
	    if (answer != null) {
	      stk = new StringTokenizer(answer, "$");
	      while (stk.hasMoreTokens()) {
		weekHour = stk.nextToken();
		(rows.get(i)).add(weekHour);
	      }
	    }
	    
	    paramValues = schoolYear+"|"+subjectCode+"|"+teacherCode;
	    ofname = schoolYear+"-"+subjectCode+"-"+teacherCode+".html";
	    File ofile = new File(dir2, ofname);
	    appendHtmlSubsection(sbuf, rows.get(i), paramValues, map, ofname);  
	    fout.println(sbuf.toString());  
	    sbuf = new StringBuffer(); 
	    makeHtmlSyllabusBody2(rows.get(i), paramValues, map, ofile);
	  } catch (Exception e) {
	    e.printStackTrace();
	  }
	}
	fout.println("</UL>");	
      }
      fout.println("</UL></body></HTML>");
      fout.close(); 
    }
  }



  public void printStudentInitPasswdList() { 
    StringBuffer sbuf = new StringBuffer();
    makePasswdLatexSource(sbuf);
    latexToPrinter(sbuf); 
  }

  public void makePasswdLatexSource(StringBuffer sbuf) { 
    int lineNumber = 30;
    int cnt = 0;
    String title = "初期パスワード：";
    String faculty = tabbedPane.getDisplayFromColumnCodeMap("FACULTY");
    String department = tabbedPane.getDisplayFromColumnCodeMap("DEPARTMENT");
    String course = tabbedPane.getDisplayFromColumnCodeMap("COURSE");
    String gakunen = tabbedPane.getDisplayFromColumnCodeMap("GAKUNEN");
    String attrib = faculty + " " + department + " " + course + " " + gakunen;
    String today = "" + commonInfo.thisYear + "年" + commonInfo.thisMonth + "月"
      + commonInfo.thisDay + "日";

    if (!latexFlag) {
      showLatexTypeSelector();
    }

    if (platexIsSelected()) {
      sbuf.append("\\documentclass[11pt]{jarticle}\n");
    } else {
      sbuf.append("\\documentstyle[11pt]{jarticle}\n");
    }
    sbuf.append("\\setlength{\\oddsidemargin}{-10pt}\n");
    sbuf.append("\\setlength{\\topmargin}{-30pt}\n");
    sbuf.append("\\setlength{\\textwidth}{500pt}\n");
    sbuf.append("\\setlength{\\textheight}{720pt}\n");
    sbuf.append("\\begin{document}\n");
    sbuf.append("\\pagestyle{empty}\n");
    sbuf.append("\\large \n");

    String str1, str2, str3, str4;
    int colmns = 5;
    boolean end_flag = false;
    String ipasswd, status, statusCode, studentStatus;
    str1 = "\\begin{table}[t]\n \\begin{tabular}[t]{|r|r|r|r|r|r|} \\hline \n";
    str2 = "\\multicolumn{"+(colmns+1)+"}{|c|}{ " + title + " " + attrib + " \\hfill "+today+"}  \\\\ \\hline\\hline \n";
    str3 = "\\hspace{6mm} & 学年 & 学生番号 & 氏名  & 初期パスワード & 在籍状況  \\\\ \\hline\\hline \n";
    str4 = "\\end{tabular}\n \\end{table}\n";
    sbuf.append(str1);
    sbuf.append(str2);
    sbuf.append(str3);
    CellObject cobj;

    int rem = tableView.getRowCount() % lineNumber;
    for (int j = 0; j < tableView.getRowCount(); j++) {
      cnt = j+1;
      sbuf.append(""+cnt);
      sbuf.append("&"+gakunen);

      cobj = (CellObject)tableView.getValueAt(j, 3);
      String passStatusCode = cobj.getCodeValue().trim();
      String passStatusName = cobj.getDisplay();

      cobj = (CellObject)tableView.getValueAt(j, 0);
      sbuf.append(" & ").append(cobj.getDisplay());

      cobj = (CellObject)tableView.getValueAt(j, 1);
      sbuf.append(" & ").append(cobj.getDisplay());

      cobj = (CellObject)tableView.getValueAt(j, 2);
      String initPasswd = cobj.getCodeValue().trim();

      if (passStatusCode.equals("0")) {
	sbuf.append("& \\verb#  " + initPasswd + "  # ");
      } else {
	sbuf.append("& (" + passStatusName + ") ");
      }
      
      cobj = (CellObject)tableView.getValueAt(j, 5);
      sbuf.append(" & ").append(cobj.getDisplay());

      sbuf.append(" \\\\ \\hline\\hline \n");
      if ((cnt % lineNumber) == 0) {
	sbuf.append(str4);
	if (cnt < tableView.getRowCount()) {
	  sbuf.append(str1);
	  sbuf.append(str3);
	  sbuf.append(str2);
	} else {
	  end_flag = true;
	}
      } 
    }
    if (end_flag) {
      sbuf.append("\\end{document}\n");
    } else {
      for (int j = rem; j < lineNumber; j++) {
	cnt = j+1;
	sbuf.append(" ");
	for (int i = 0; i < colmns; i++) {
	  sbuf.append("& ");
	}
	sbuf.append("\\\\ \\hline\\hline \n");
      }
      sbuf.append(str4);  
      sbuf.append("\\end{document}\n");
    }
  }

  public void printAttendCheckSheet() {
    StringBuffer sbuf = new StringBuffer();
    makeAttendanceCardLatexSource(sbuf);
    latexToPrinter(sbuf); 
  }

  public void makeAttendCheckSheetLatex() {
    StringBuffer sbuf = new StringBuffer();
    makeAttendanceCardLatexSource(sbuf);
    
    File f = null;
    int ret = chooser.showSaveDialog(commonInfo.getFrame());
    if (ret == JFileChooser.APPROVE_OPTION) {
      f = chooser.getSelectedFile(); 
    } else {
      return;
    } 
    try {
      PrintWriter fout = new PrintWriter(new BufferedWriter(new FileWriter(f)));
      fout.println(sbuf.toString());
      fout.close();
    } catch (Exception e) {
      commonInfo.showMessage(e.toString());
      return;
    }
  }
  

  public void makeAttendanceCardLatexSource(StringBuffer sbuf) { 
    int lineNumber = 30;
    int cnt = 0;
    String title1 = "出席調査シート: ";
    String subjectName = tabbedPane.getDisplayFromColumnCodeMap("SUBJECT_CODE");
    String teacherName = tabbedPane.getDisplayFromColumnCodeMap("TEACHER_CODE");
    String classCode = tabbedPane.getDisplayFromColumnCodeMap("CLASS_CODE");
    String schoolYear = tabbedPane.getDisplayFromColumnCodeMap("SCHOOL_YEAR");
    if (schoolYear == null) {
      schoolYear = tabbedPane.getDisplayFromColumnCodeMap("THIS_SCHOOL_YEAR");
    }

    sbuf.append("\\documentclass[11pt]{jarticle}\n");
    sbuf.append("\\setlength{\\oddsidemargin}{-10pt}\n");
    sbuf.append("\\setlength{\\topmargin}{-30pt}\n");
    sbuf.append("\\setlength{\\textwidth}{500pt}\n");
    sbuf.append("\\setlength{\\textheight}{720pt}\n");
    sbuf.append("\\begin{document}\n");
    sbuf.append("\\pagestyle{empty}\n");
    sbuf.append("\\large \n");
    sbuf.append("\\renewcommand{\\arraystretch}{1.3}\n");

    String str1, str2, str3, str4;
    int columns = 8;
    boolean end_flag = false;
    int pageNumber = 1;
    str1 = "\\begin{table}[t]\n \\begin{tabular}[t]{|r|r|r|r|r|r|r|r|} \\hline \n";
    str2 = "\\multicolumn{"+columns+"}{|c|}{  "+title1+" \\hfill "+subjectName+ " \\hspace{3mm} (" + classCode + ")  \\hfill " +teacherName +" 教員担当 }  \\\\ \\hline\\hline \n";

    str3 = "\\multicolumn{3}{|r|}{ " +schoolYear + " 講義日 } & \\hspace{5mm}月 \\hspace{5mm}日 &  \\hspace{5mm}月 \\hspace{5mm}日 & \\hspace{5mm}月 \\hspace{5mm}日 &  \\hspace{5mm}月 \\hspace{5mm}日 & \\hspace{5mm}月 \\hspace{5mm}日 \\\\ \\hline\\hline \n";

    str4 = "\\end{tabular}\n \\end{table}\n";

    sbuf.append(str1);
    sbuf.append(str2);
    sbuf.append(str3);
    int rem = tableView.getRowCount() % lineNumber;

    CellObject cobj;
    for (int j = 0; j < tableView.getRowCount(); j++) {
      cnt = j+1;
      cobj = (CellObject)tableView.getValueAt(j, 1);
      String deptment = cobj.getDisplay();
      String dept = deptment.substring(0, 4);

      cobj = (CellObject)tableView.getValueAt(j, 3);
      String gakunen = cobj.getDisplay();

      cobj = (CellObject)tableView.getValueAt(j, 4);
      String studentCode = cobj.getDisplay();

      sbuf.append(dept);
      sbuf.append(" & ");
      sbuf.append(gakunen);
      sbuf.append(" & ");
      sbuf.append(studentCode);
      if ((cnt % 10) == 0) {
	sbuf.append("& & & & &  \\\\ \\hline\\hline \n");
      } else {
	sbuf.append("& & & & &  \\\\ \\hline \n");
      }
      if ((cnt % lineNumber) == 0) {
	sbuf.append("\\multicolumn{"+columns+"}{|c|}{ 出席学生は (学生番号、講義日) で指定された自分のスペースに署名すること。\\hfill ("+ pageNumber +"枚目) } \\\\  \\hline \n");
	sbuf.append(str4);
	pageNumber++; 
	if (cnt < tableView.getRowCount()) {
	  sbuf.append(str1);
	  sbuf.append(str3);
	  sbuf.append(str2);
	} else {
	  end_flag = true;
	}
      } 
    }
    if (end_flag) {
      sbuf.append("\\end{document}\n");
    } else {
      for (int j = rem; j < lineNumber; j++) {
	cnt = j+1;
	if ((cnt % 10) == 0) {
	  sbuf.append(" &  &  & & & & &  \\\\ \\hline\\hline \n");
	} else {
	  sbuf.append(" &  &  & & & & &  \\\\ \\hline \n");
	}
      }
      sbuf.append("\\multicolumn{"+columns+"}{|c|}{ 出席学生は (学生番号、講義日) で指定された自分のスペースに署名すること。\\hfill ("+ pageNumber +"枚目) } \\\\  \\hline \n");
      sbuf.append(str4);  
      sbuf.append("\\end{document}\n");
    }
  }

}
