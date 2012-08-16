package common;
import clients.*;
import java.util.*;
import javax.swing.*;
//import javax.swing.table.*;
//import java.awt.*;
//import java.awt.event.*;
import java.io.*;
import java.io.File;

public class NinteiTableView extends TableViewBase {
  static JFileChooser chooser = new JFileChooser();

  public NinteiTableView(String tableViewType,
			 String serviceName,
			 String nodePath,
			 String panelID,
			 CommonInfo commonInfo, 
			 TabbedPaneBase tabbedPane,
			 DataPanelBase dataPanel) {
    super(tableViewType, 
	  serviceName, nodePath, panelID,
	  commonInfo, tabbedPane, dataPanel);
  }
  

  public void makeNinteiForm() {
    int ret = chooser.showSaveDialog(commonInfo.getFrame());
    if (ret != JFileChooser.APPROVE_OPTION) return;
    File f = chooser.getSelectedFile();

    String studentCode = parentTabbedPane.getValueFromColumnCodeMap("STUDENT_CODE");
    String studentName = parentTabbedPane.getDisplayFromColumnCodeMap("STUDENT_CODE");
    String cYear       = parentTabbedPane.getDisplayFromColumnCodeMap("CURRICULUM_YEAR");
    
    try {
      PrintWriter fout = new PrintWriter(new BufferedWriter(new FileWriter(f)));
      String line = "NINTEI_FORM|" + studentCode + "|" + studentName;
      fout.println(line);
      fout.println("#  先頭行は「科目識別コード」なので変更しないで下さい。");
      fout.println("#----------------------------------------------------------------  ");
      fout.println("#  ３年次編入学生  単位認定フォーム  ");
      fout.println("#  ");
      fout.println("#    編入学年度: " + cYear );
      fout.println("#      学生番号: " + studentCode );
      fout.println("#      学生氏名: " + studentName );
      fout.println("#  ");
      fout.println("# ( "+commonInfo.thisYear+"年 "+commonInfo.thisMonth+"月 "+commonInfo.thisDay+"日 )");
      fout.println("#----------------------------------------------------------------  ");
      fout.println("#  以下のリストの「Y/N」欄と「認定の根拠科目」欄に、各々の科目の単位認定に ");
      fout.println("#  関する事項を記入して下さい。");
      fout.println("#  ");
      fout.println("#  ・ 認定する科目については、「Y/N」欄に Y を記入し、「認定の根拠」欄には ");
      fout.println("#     その科目の認定の根拠となる高専等での履修科目に関するデータを記載して下さい。    ");
      fout.println("#  ");
      fout.println("#  ・ 「認定科目I, II (自然、情報、対象)」については、履修課程表にはその「単位数」が");
      fout.println("#      設定されていません。「認定科目I, II (自然、情報、対象)」の単位を認定する場合には、");
      fout.println("#     「認定の根拠」欄の先頭部分に「認定する単位数」を文章で記載して下さい。");
      fout.println("#  ");
      fout.println("#  ・ 認定の対象としない科目については、「Y/N」欄に N と記入して下さい。    ");
      fout.println("#  ");
      fout.println("#  ・「認定の根拠科目」欄には、EUC や UNICODE などの文字セットに含まれない文字 ");
      fout.println("#     (全角のローマ数字 等) を記入しないで下さい。");
      fout.println("#  ");
      fout.println("#  ・「認定の根拠」欄に記入できるのは360文字 (全角文字は２文字と勘定) に制限され、");
      fout.println("#     それより長い文字列は切捨てられます。");
      fout.println("#     また、テキストには、デリミタとして用いられている : を使わないで下さい。");
      fout.println("#  ");
      fout.println("#----------------------------------------------------------------  ");
      fout.println("#  先頭文字が # である行はコメント行と見なされます。");
      fout.println("#----------------------------------------------------------------  ");
      fout.println("#  ");
      fout.println("# 科目コード: Y/N :  認定の根拠 (エビデンス)  : 科目名   : 年次  : 科目区分 : 単位区分  ");
      fout.println("#  ");
      ArrayList<String> list = makeDataForNinteiForm();
      for (int i = 0; i < list.size(); i++) {
	line = list.get(i);
	fout.println(line);
      }
      fout.close();
    } catch (Exception e) {
      commonInfo.showMessage(e.toString());
    }
  }

  public void readNinteiForm() {
    if (chooser.showOpenDialog(commonInfo.getFrame()) != JFileChooser.APPROVE_OPTION) return;
    File f = chooser.getSelectedFile();

    String studentCode = parentTabbedPane.getValueFromColumnCodeMap("STUDENT_CODE");
    String studentName = parentTabbedPane.getDisplayFromColumnCodeMap("STUDENT_CODE");
    String faculty     = parentTabbedPane.getDisplayFromColumnCodeMap("FACULTY");
    String department  = parentTabbedPane.getDisplayFromColumnCodeMap("DEPARTMENT");
        
    try {
      BufferedReader fin = new BufferedReader(new FileReader(f));
      String line = fin.readLine();
      String[] tokens = line.split("\\|");
      String code2        = tokens[0];
      String studentCode2 = tokens[1];
      String studentName2 = tokens[2];
      if ((code2.equals("NINTEI_FORM")) && (studentCode2.equals(studentCode))) {
	ArrayList<String> list = readFromNinteiForm(fin);	
	int num = commonInfo.commonInfoMethods.reportNinteiData(faculty, department, studentCode, list);
	if (num > 0) {
	  refreshTable();
	}
      } else {
	commonInfo.showMessageLong("読み込もうとする「認定フォーム」の学生は、$ 認定ツールに表示されている学生と一致しません。");
	fin.close();
	return;
      }
      fin.close();
    } catch (Exception e) {
      commonInfo.showMessage(e.toString());
    }
  }

  public ArrayList<String> makeDataForNinteiForm() { 
    ArrayList<String> list = new ArrayList<String>();
    for (int i = 0; i < tableModel.getRowCount(); i++) {
      String subjectCode   = tableModel.getCellValueAt(i, "授業科目名").trim();
      String subjectName   = tableModel.getCellDisplayAt(i, "授業科目名").trim();
      String kubunCode     = tableModel.getCellDisplayAt(i, "科目区分").trim();
      String reqCode       = tableModel.getCellDisplayAt(i, "単位区分").trim();
      String qualify       = tableModel.getCellValueAt(i, "認定").trim();
      String evidence      = tableModel.getCellDisplayAt(i, "認定の根拠科目＋付加的情報").trim();
      String gakunen       = tableModel.getCellDisplayAt(i, "学年").trim();
      
      StringBuffer sbuf = new StringBuffer();
      sbuf.append(subjectCode).append(" : ");
      sbuf.append(qualify).append(" : ");
      if (evidence.equals("")) {
	sbuf.append("\t").append(" : ");
      } else {
	sbuf.append(evidence).append(" : ");
      }
      sbuf.append(subjectName).append(" : ");
      sbuf.append(gakunen).append(" : ");
      sbuf.append(kubunCode).append(" : ");
      sbuf.append(reqCode);

      list.add(sbuf.toString());
    }
    return list;
  }

  public ArrayList<String> readFromNinteiForm(BufferedReader fin) {
    ArrayList<String> list = new ArrayList<String>();

    HashMap<String, Integer> subjectRowMap = new HashMap<String, Integer>();
    HashMap<String, String> subjectQualMap = new HashMap<String, String>();
    HashMap<String, String> subjectEviMap  = new HashMap<String, String>();

    for (int i = 0; i < getRowCount(); i++) {
      String subjectCode   = tableModel.getCellValueAt(i, "授業科目名").trim();
      String qualify       = tableModel.getCellValueAt(i, "認定").trim();
      String evidence      = tableModel.getCellDisplayAt(i, "認定の根拠科目＋付加的情報").trim();
      String gakunen       = tableModel.getCellDisplayAt(i, "学年").trim();

      subjectRowMap.put(subjectCode, new Integer(i));
      subjectQualMap.put(subjectCode, qualify);
      subjectEviMap.put(subjectCode, evidence);      
    }
    
    try {
      String line;
      while ((line = fin.readLine()) != null) {
	if (!(line.startsWith("#"))) {
	  String[] tokens = line.split("\\:");
	  String subjectCode = tokens[0].trim();
	  String qualify     = tokens[1].trim();
	  String evidence    = tokens[2].trim();
	  if (qualify.equals("")) qualify = "N";
	  if (evidence.equals("")) evidence = " ";

	  Integer intObj = subjectRowMap.get(subjectCode);
	  int row = intObj.intValue();
	  String qualify2  = subjectQualMap.get(subjectCode);
	  String evidence2 = subjectEviMap.get(subjectCode);
	  if ((!qualify2.equals(qualify)) || (!evidence2.equals(evidence))) {
	    String data = subjectCode + "|" + qualify + "|" + evidence;
	    list.add(data);
	  }
	}
      }
    } catch( Exception e ) {
      commonInfo.showMessage(e.toString());
    }
    return list;
  }
}
