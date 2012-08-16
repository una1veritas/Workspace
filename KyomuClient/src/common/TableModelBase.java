package common;

import clients.*;
import java.util.*;
import java.awt.*;
import java.awt.event.*;
//import javax.swing.*;
import javax.swing.table.*;

public class TableModelBase extends AbstractTableModel  {
  protected CommonInfo commonInfo;
  protected String serviceName;
  protected String panelID;

  protected ArrayList<String> columnTitleList;
  protected ArrayList<String> columnCodeList;
  protected ArrayList<String> columnDisplayList;

  protected ArrayList<ArrayList<CellObject>> listOfRows 
    = new ArrayList<ArrayList<CellObject>>();
  protected HashMap<String, Integer> columnCodeToColumnIndexMap
    = new HashMap<String, Integer>();
  protected HashMap<String, Integer> keyValueToRowIndexMap 
    = new HashMap<String, Integer>();
 
  public TableModelBase(CommonInfo commonInfo, 
			String serviceName,
			String panelID,
			ArrayList<String> columnTitleList,
			ArrayList<String> columnCodeList,
			ArrayList<String> columnDisplayList) {
    this.commonInfo = commonInfo;
    this.serviceName = serviceName;
    this.panelID = panelID;
    this.columnTitleList = columnTitleList;
    this.columnCodeList = columnCodeList;
    this.columnDisplayList = columnDisplayList;
    for (int i = 0; i < columnCodeList.size(); i++) {
      String columnCode = columnCodeList.get(i);
      Integer index = i;
      columnCodeToColumnIndexMap.put(columnCode, index);
    }
  }

  public void setColumnLists(ArrayList<String> columnTitleList,
			     ArrayList<String> columnCodeList,
			     ArrayList<String> columnDisplayList) {
    this.columnTitleList = columnTitleList;
    this.columnCodeList = columnCodeList;
    this.columnDisplayList = columnDisplayList;
  }

  //**** implementations of Table Model interface methods ***//
  
  public int getColumnCount() {
    return columnTitleList.size();
  }

  public String getColumnName(int col) {
    return columnTitleList.get(col);
  }

  public int getRowCount() {
    return listOfRows.size();
  }
  
  public Object getValueAt(int row, int col) {
    ArrayList<CellObject> list = listOfRows.get(row);
    return list.get(col);
  }  
  
  public Object getValueAt(int row, String columnName) {
    int col = -1;
    for (int i = 0; i < getColumnCount(); i++) {
      if (getColumnName(i).equals(columnName)) {
	col = i;
	break;
      }
    }
    if (col >= 0) {	
      return getValueAt(row, col);
    } else {
      return null;
    }
  }   
  
  public String getCellValueAt(int row, String columnName) {
    CellObject cobj = (CellObject) getValueAt(row, columnName);
    if (cobj != null) return cobj.getCodeValue();
    else return null;
  }    
  
  public String getCellDisplayAt(int row, String columnName) {
    CellObject cobj = (CellObject) getValueAt(row, columnName);
    if (cobj != null) return cobj.getDisplay().trim();
    else return null;
  }  

  public void setValueAt(CellObject obj, int row, int col) {
    ArrayList<CellObject> list = listOfRows.get(row);
    list.set(col, obj);
  }

  public void setValueAt(CellObject obj, int row, String columnName) {
    int col = -1;
    for (int i = 0; i < getColumnCount(); i++) {
      if (getColumnName(i).equals(columnName)) {
	col = i;
	break;
      }
    }
    if (col >= 0) {
      setValueAt(obj, row, col);
    }
  }

  public boolean isCellEditable(int row, int col) {
    return false;
  }
    
  //**** setTableData  ***//
  
  public void setTableData(String serviceName, String panelID, 
			   String switchCode, String queryParamValues) {  
    listOfRows.clear(); 

    if (queryParamValues != null) {
      String answer = commonInfo.getQueryResult(serviceName, panelID, switchCode, 
						queryParamValues); 
      if (answer != null) {
	if ((serviceName.equals("HolidaySelector")) ||
	    (serviceName.equals("SchoolEventSelector"))) {
	  HashSet<String> set = new HashSet<String>();
	  ArrayList<String> list = new ArrayList<String>();
	  String[] lines = answer.split("\\$");
	  for (String line : lines) {
	    String[] tokens = line.split("\\|");
	    for (String token : tokens) {
	      String[] elems = token.split(" ");
	      for (String elem : elems) {
		if (!elem.equals("")) {
		  if (!set.contains(elem)) {
		    list.add(elem);
		    set.add(elem);
		  }
		}
	      }
	    }
	  }
	  Collections.sort(list);
	  for (String elem : list) {
	    ArrayList<CellObject> rowList = new ArrayList<CellObject>();
	    rowList.add(new CellObject(elem));
	    listOfRows.add(rowList);
	  }
	} else {
	  String[] lines = answer.split("\\$");
	  for (String line : lines) {
	    String[] tokens = line.split("\\|");
	    LinkedList<String> linkedList = new LinkedList<String>();	    
	    for (String token : tokens) {
	      linkedList.add(token);
	    }	
	    ArrayList<CellObject> rowList = new ArrayList<CellObject>();
	    for (int i = 0; i < getColumnCount(); i++) {
	      String columnDisplay = columnDisplayList.get(i);
	      CellObject cobj = makeCellObject(columnDisplay, linkedList);
	      rowList.add(cobj);
	    }
	    listOfRows.add(rowList);
	  }
	}
      }
    }
    fireTableChanged(null); 
  }

  public void setVarTableData(String serviceName, String panelID,
			      String addDataSwitchCode,
			      String addDataQueryParamValues,
			      String addColumnDisplay,
			      String keyColumnCode) { 
    columnCodeToColumnIndexMap.clear();
    for (int i = 0; i < getColumnCount(); i++) {
      String columnCode = columnCodeList.get(i);
      columnCodeToColumnIndexMap.put(columnCode, i);
    } 
    keyValueToRowIndexMap.clear();
    int keyColumnIndex = columnCodeToColumnIndexMap.get(keyColumnCode);
    for (int i = 0; i < getRowCount(); i++) {
      CellObject cobj = (CellObject) getValueAt(i, keyColumnIndex);
      String keyRowValue = cobj.getCodeValue();
      keyValueToRowIndexMap.put(keyRowValue, i);

      ArrayList<CellObject> rowList = listOfRows.get(i);
      int n = rowList.size();
      for (int j = n; j < getColumnCount(); j++) {
	rowList.add(new CellObject(null, " "));
      }
    }    

    String answer = commonInfo.getQueryResult(serviceName, panelID, addDataSwitchCode,
					      addDataQueryParamValues);
    if (answer != null) {
      String[] lines = answer.split("\\$");
      for (String line : lines) {
	String[] tokens = line.split("\\|");    
	String rowKeyValue = tokens[0];
	String colKeyValue = tokens[1];
	LinkedList<String> linkedList = new LinkedList<String>();
	for (int i = 2; i < tokens.length; i++) {
	  linkedList.add(tokens[i]);
	}
	CellObject cobj = makeCellObject(addColumnDisplay, linkedList);
	try {  
	  int rowIndex = keyValueToRowIndexMap.get(rowKeyValue);
	  int colIndex = columnCodeToColumnIndexMap.get(colKeyValue);
	  setValueAt(cobj, rowIndex, colIndex);
        } catch (Exception ex) { 
	  //	  System.out.println("ignored data: " + line); //
        }                         
      }
    }
    fireTableChanged(null); 
  }

  protected CellObject makeCellObject(String colDisplay, LinkedList<String> list) {
    String codeVal, codeDisplay;
    try {
      codeVal = (list.removeFirst()).trim();
    } catch (Exception e) {
      return new CellObject(null, " ");
    }
    
    try { 
      if (colDisplay.equals("SIMPLE")) {
	return new CellObject(codeVal);      
      } else if (colDisplay.equals("COMPOSITE")) {
	codeDisplay = list.removeFirst();
	return new CellObject(codeVal, codeDisplay);
      } else if (colDisplay.startsWith("NUMBER")) { 
	int index = colDisplay.indexOf(":");
	String param = colDisplay.substring(index+1);
	int index2 = param.indexOf(":");
	if (index2 < 0) {
	  int n1 = Integer.parseInt(param);
	  if (codeVal.equals("")) {
	    return new CellObject(null, "");
	  } else {
	    return new CellObject(new Integer(codeVal), n1);
	  }
	} else {
	  String param2 = param.substring(0, index2);
	  String param3 = param.substring(index2+1);
	  int n2 = Integer.parseInt(param2);
	  int n3 = Integer.parseInt(param3);
	  if (codeVal.equals("")) {
	    return new CellObject(null, "");
	  } else {
	    return new CellObject(new Double(codeVal), n2, n3);
	  }
	}
      } else if (colDisplay.startsWith("YEAR")) { 
	if (codeVal.equals("")) {
	  return new CellObject(null, "");
	} else {	  
	  return new CellObject(new Integer(codeVal), codeVal + "年");	
	}
      } else if (colDisplay.startsWith("TOKUTEN")) { 
	if (codeVal.equals("")) {
	  return new CellObject(null, "");
	} else {
	  if (codeVal.equals("200")) {
	    return new CellObject(new Integer(codeVal), "単位認定");
	  } else if (codeVal.equals("300")) {
	    return new CellObject(new Integer(codeVal), "編入認定");
	  } else if (codeVal.equals("0")) {
	    return new CellObject(new Integer(codeVal), "(不合格)");
	  } else {
	    return new CellObject(new Integer(codeVal), 5);
	  }  
	}
      } else if (colDisplay.startsWith("MARKS")) { 
	if (codeVal.equals("")) {
	  codeVal = "-1";
	  codeDisplay = "";
	  return new CellObject(new Integer(codeVal), codeDisplay);
	} else {
	  return new CellObject(new Integer(codeVal), 4);
	}
      } else if (colDisplay.equals("QUALIFY_CODE")) { 
	if (codeVal.equals("")) {
	  return new CellObject(null, ""); 
	} else if (codeVal.equals("Y")) {
	  return new CellObject(codeVal, "◯"); 
	} else if (codeVal.equals("N")) {
	  return new CellObject(codeVal, "×"); 
	} else {
	  return new CellObject(codeVal, "？"); 
	}
      } else if (colDisplay.equals("HYOKA")) { 
	Integer obj;
	if (codeVal.equals("")) {
	  return new CellObject(null, "");
	} else {
	  int gain = Integer.parseInt(codeVal);
	  if (gain == 0) {
	    codeDisplay = "不合格";
	  } else if (gain < 60) {
	    codeDisplay = "再試験";
	  } else if (gain < 70) {
	    codeDisplay = "可";
	  } else if (gain < 80) {
	    codeDisplay = " 良";
	  } else if (gain < 90) {
	    codeDisplay = "  優";
	  } else if (gain <= 100) {
	    codeDisplay = "   秀";
	  } else if (gain == 200) {
	    codeDisplay = "認定";
	  } else if (gain == 200) {
	    codeDisplay = "JABEE認定";
	  } else {
	    codeDisplay = "";
	  }
	  return new CellObject(new Integer(codeVal), codeDisplay);     
	}
      } else if (colDisplay.startsWith("SEISEKI_STATUS")) { 
	String code2 = (list.removeFirst()).trim();
	if (codeVal.equals("T")) {
	  return new CellObject(codeVal, "仮履修申告"); 
	} else if (codeVal.equals("F")) {
	  return new CellObject(codeVal, "成績処理済"); 
	} else if (codeVal.equals("N")) {
	  return new CellObject(codeVal, " "); 
	} else {
	  return new CellObject(codeVal, "(仮) " + code2 ); 
	}	
      } else if (colDisplay.startsWith("PASSWORD_STATUS")) { 
	if (codeVal.equals("")) {
	  return new CellObject(codeVal, "-- 未登録 --"); 
	} else if (codeVal.equals("0")) {
	  return new CellObject(codeVal, "未設定(初期パスワード)"); 
	} else if (codeVal.equals("1")) {
	  return new CellObject(codeVal, "設定済み"); 
	} else if (codeVal.equals("9")) {
	  return new CellObject(codeVal, "-- 使用停止中 --"); 
	} else {
	  return new CellObject(codeVal, "-- ？？ --" ); 
	}	
      } else if (colDisplay.startsWith("PASSWORD")) { 
	return new CellObject(codeVal, "(非開示)");  
      } else if (colDisplay.startsWith("NINTEI_PLUS")) { 	
	String code2 = (list.removeFirst()).trim();
	Double obj;
	String disp = "";
	if (codeVal.equals("")) {
	  obj = null;
	  disp = "";
	} else {
	  obj = new Double(codeVal);
	  if (code2.equals("")) {
	    disp = codeVal;
	  } else {
	    disp = codeVal + " (" + code2 + ")";
	  }
	}
	int k = 12 - disp.length();
	StringBuffer sb = new StringBuffer();
	while (k > 0) {
	  sb.append(" ");
	  k--;
	}
	sb.append(disp); 
	return new CellObject(obj, sb.toString());
      } else if (colDisplay.startsWith("NINTEI_MARKS")) { 	
	String ninteiFlag = codeVal;
	String marks = (list.removeFirst()).trim();
	Integer val;
	if (ninteiFlag.equals("Q")) {
	  if (marks.equals("")) {
	    return new CellObject(null, "(認定)");
	  } else {
	    return new CellObject(new Integer(marks), "(認定)");
	  }
	} else if (ninteiFlag.equals("J")) {
	  if (marks.equals("")) {
	    return new CellObject(null, "(J認定)");
	  } else {
	    return new CellObject(new Integer(marks), "(J認定)");
	  }
	} else {
	  if (marks.equals("")) {
	    return new CellObject(null, "");
	  } else {
	    return new CellObject(new Integer(marks), 8);
	  }
	}
      } else if (colDisplay.startsWith("YOKEN_UNIT")) { 	
	String code2 = (list.removeFirst()).trim();
	if (codeVal.equals("")) {
	  return new CellObject(null, " ");
	} 
	CellObject cobj = new CellObject(new Double(codeVal), 3, 1);
	String disp = cobj.getDisplay();
	if (code2.equals("1")) {      
	  return new CellObject(new Double(codeVal), "  " + disp);
	} else if (code2.equals("-1"))  { 
	  return new CellObject(new Double(codeVal), "* " + disp); 
	} else if (code2.equals("2")) { 
	  return new CellObject(new Double(codeVal), "+ " + disp); 
	} else { 
	  return new CellObject(new Double(codeVal), "  " + disp);
	}	
      } else if (colDisplay.equals("ATTEND_FLAG")) { 
	if (codeVal.equals("")) {
	  return new CellObject(null, " ");
	} else if (codeVal.equals("1")) {
	  return new CellObject(codeVal, " ◯");
	} else if (codeVal.equals("2")) {
	  return new CellObject(codeVal, " △");
	} else if (codeVal.equals("3")) {
	  return new CellObject(codeVal, " ▲");
	} else {   
	  return new CellObject(codeVal, " ?");
	} 
      } else if (colDisplay.equals("COMPOSITE_ATTEND_FLAG")) { 
	String readFlag = (list.removeFirst()).trim();
	if (codeVal.equals("")) codeVal = " ";
	if (readFlag.equals("")) readFlag = " ";
	String attendCode = codeVal + "|" + readFlag;
	if (codeVal.equals("1")) {
	  return new CellObject(attendCode, " ◯");
	} else if (codeVal.equals("2")) {
	  return new CellObject(attendCode, " △");
	} else if (codeVal.equals("3")) {
	  return new CellObject(attendCode, " ▲");
	} else {   
	  return new CellObject(attendCode, "  ");
	} 
      } else if (colDisplay.equals("CLASS_DATE")) {  
	String[] tokens = codeVal.split("\\:");
	String yr = tokens[0];
	String mn = tokens[1];
	String dy = tokens[2];
	String dsp = yr + "年" + mn + "月" + dy + "日";
	return new CellObject(codeVal, dsp);
      } else if (colDisplay.equals("HOUR")) {
	if (codeVal.equals("")) {
	  return new CellObject(null, " ");
	} else if (codeVal.equals("0")) {
	  return new CellObject(codeVal, " ");
	} else {
	  return new CellObject(codeVal, " " + codeVal + "限目");
	}
      } else if (colDisplay.equals("MONTHDATE")) {
	if (codeVal.equals("")) {
	  return new CellObject(null, " ");      
	} else { 
	  String[] tokens = codeVal.split("\\:");	
	  String yr = tokens[0];
	  String mn = tokens[1];
	  String dt = tokens[2];
	  String date = yr + "年" + mn + "月" + dt + "日";
	  return new CellObject(codeVal, date); 
	}
      } else if (colDisplay.equals("HOURMINUTE")) {
	if (codeVal.equals("")) {
	  return new CellObject(null, " ");      
	} else {
	  String[] tokens = codeVal.split("\\:");	
	  String hr = tokens[0];
	  String min = tokens[1];
	  String sec = tokens[2];
	  String tm = hr + "時" + min + "分" + sec + "秒";
	  return new CellObject(codeVal, tm); 
	}
      } else if (colDisplay.equals("ROOM2")) {
	String sflag = (list.removeFirst()).trim();
	if (sflag.equals("")) {
	  return new CellObject(sflag, " ");
	} else if (sflag.equals("N")) {
	  return new CellObject(sflag, "(未開示)");
	} else {
	  String disp = commonInfo.getGakumuCodeShorterName("ROOM", codeVal);
	  return new CellObject(codeVal, disp);
	}
      } else if (colDisplay.equals("HOUR2")) {
	String sflag = (list.removeFirst()).trim();
	if (sflag.equals("")) {
	  return new CellObject(sflag, " ");
	} else if (sflag.equals("N")) {
	  return new CellObject(sflag, "(未開示)");
	} else {
	  if (codeVal.equals("")) {
	    return new CellObject(codeVal, " ");
	  } else if (codeVal.equals("0")) {
	    return new CellObject(codeVal, " ");
	  } else {
	    return new CellObject(codeVal, " " + codeVal + "限目");
	  }
	}
      } else if (colDisplay.equals("EXAM_DATE3")) {
	String jflag = (list.removeFirst()).trim();
	String sflag = (list.removeFirst()).trim();
	if (sflag.equals("")) {
	  return new CellObject(sflag, "(未設定)");
	} else if (sflag.equals("N")) {
	  return new CellObject(sflag, "(未開示)");
	} else {
	  if (jflag.equals("0")) {
	    String[] tokens = codeVal.split("\\-");	
	    String yr = tokens[0];
	    String mn = tokens[1];
	    String dt = tokens[2];
	    String date = yr + "年" + mn + "月" + dt + "日";
	    return new CellObject(codeVal, date); 
	  } else if (jflag.equals("1")) {
	    return new CellObject(jflag, "試験期間外に実施");
	  } else if (jflag.equals("2")) {
	    return new CellObject(jflag, "試験は実施しない");
	  } else if (jflag.equals("")) {
	    return new CellObject(jflag, "(未設定)");
	  } else {
	    return new CellObject(jflag, "(？)");
	  }	
	}
      } else if (colDisplay.equals("JISSHI_FLAG")) {	
	if (codeVal.equals("")) {
	  return new CellObject(" ", "(未設定)");
	} else if (codeVal.equals("0")) {
	  return new CellObject(codeVal, "試験を実施");
	} else if (codeVal.equals("1")) {
	  return new CellObject(codeVal, "期間外に実施");
	} else if (codeVal.equals("2")) {
	  return new CellObject(codeVal, "試験は実施せず");
	} else {
	  return new CellObject(codeVal, "？");
	} 
      } else if (colDisplay.equals("SHOW_FLAG")) {
	if (codeVal.equals("")) {
	  return new CellObject(codeVal, " ");      
	} else if (codeVal.equals("Y")) {
	  return new CellObject(codeVal, "◯"); 
	} else if (codeVal.equals("N")) {
	  return new CellObject(codeVal, "×"); 
	} else {
	  return new CellObject(codeVal, "？"); 
	}
      } else if (colDisplay.equals("ALIVE_FLAG")) {
	if (codeVal.equals("")) {
	  return new CellObject(codeVal, " ");      
	} else if (codeVal.equals("1")) {
	  return new CellObject(codeVal, "◯"); 
	} else if (codeVal.equals("0")) {
	  return new CellObject(codeVal, "×"); 
	} else {
	  return new CellObject(codeVal, "？"); 
	}
      } else if (colDisplay.equals("COLOR_DISP")) {
	// System.out.println(codeVal); //
	String r = list.removeFirst().trim();
	String g = list.removeFirst().trim();
	String b = list.removeFirst().trim();
	// System.out.println(r + ", " + g + ", " + b); //
	int red = Integer.parseInt(r);
	int green = Integer.parseInt(g);
	int blue = Integer.parseInt(b);
	Color color = new Color(red, green, blue);
	return new CellObject(color, codeVal); 
      } else if (codeVal.equals("")) {
	return new CellObject(null, " "); 
      } else if (commonInfo.gakumuCategorySetContains(colDisplay)) {	
	String disp = commonInfo.getGakumuCodeShorterName(colDisplay, codeVal);
	if (disp == null) {
	  disp = "("+codeVal+")";
	}
	return new CellObject(codeVal, disp);
      } else {
	commonInfo.showMessageLong("undefined ColumnDisplay = " + colDisplay );
	return new CellObject(codeVal);
      }
    } catch (Exception e) {
      commonInfo.showMessageLong("makeCellObject $ " + e.toString() + " $ columnDisplay = " + colDisplay );
    }
    return new CellObject(" ");
  }

  public void sortByColumn(int column, boolean ascending) {
    Collections.sort(listOfRows, new RowComparator(column));   
    if (!ascending) {
      Collections.reverse(listOfRows);
    }  
    fireTableDataChanged();
  } 
  
  class RowComparator implements Comparator<ArrayList<CellObject>> {  
    int column;    

    public RowComparator(int column) { 
      this.column = column;
    }    

    public int compare(ArrayList<CellObject> list1, ArrayList<CellObject> list2) { 
      CellObject cobj1 = list1.get(column);
      CellObject cobj2 = list2.get(column);
      
      if (cobj1 == null && cobj2 == null) {
	return 0; 
      } else if (cobj1 == null) { 
	return -1; 
      } else if (cobj2 == null) { 
	return 1; 
      }

      Object o1 = cobj1.getCode();
      Object o2 = cobj2.getCode();
      if (o1 == null && o2 == null) {
	return 0; 
      } else if (o1 == null) { 
	return -1; 
      } else if (o2 == null) { 
	return 1; 
      }

      if (o1 instanceof String) {
	String s1 = (String) o1;
	String s2 = (String) o2;
	return s1.compareTo(s2);
      } else if (o1 instanceof Integer) {
	Integer i1 = (Integer) o1;
	Integer i2 = (Integer) o2;
	return i1.compareTo(i2);
      } else if (o1 instanceof Double) {
	Double d1 = (Double) o1;
	Double d2 = (Double) o2;
	return d1.compareTo(d2);
      } else {
	return 0;
      }
    }
  }  
 
  public void addHeaderMouseListenerToSort(TableViewBase table) { 
    final TableViewBase jtable = table;
    jtable.setColumnSelectionAllowed(false); 
    MouseAdapter sorterMouseListener = new MouseAdapter() {
      public void mouseClicked(MouseEvent e) {
	TableColumnModel columnModel = jtable.getColumnModel();
	int viewColumn = columnModel.getColumnIndexAtX(e.getX()); 
	int column = jtable.convertColumnIndexToModel(viewColumn); 
	if(e.getClickCount() == 1 && column != -1) {
	  int shiftPressed = e.getModifiers() & InputEvent.SHIFT_MASK; 
	  boolean ascending = (shiftPressed == 0); 
	  sortByColumn(column, ascending); 
	}
      }
    };
    JTableHeader header = jtable.getTableHeader(); 
    header.addMouseListener(sorterMouseListener); 
  }  

  public void addHeaderMouseListenerToOrderedSort(TableViewBase table) { 
    final TableViewBase jtable = table;
    jtable.setColumnSelectionAllowed(false); 
    MouseAdapter orderedSorterMouseListener = new MouseAdapter() {
      public void mouseClicked(MouseEvent e) {
	TableColumnModel columnModel = jtable.getColumnModel();
	int viewColumn = columnModel.getColumnIndexAtX(e.getX()); 
	int column = jtable.convertColumnIndexToModel(viewColumn); 
	if(e.getClickCount() == 1 && column != -1) {
	  int shiftPressed = e.getModifiers() & InputEvent.SHIFT_MASK; 
	  boolean descending = (shiftPressed != 0); 
	  sortByColumn(column, descending); 
	}	
	int viewColumn2 = columnModel.getColumnIndex("順位");
	if (viewColumn < viewColumn2) {
	  columnModel.moveColumn(viewColumn2, viewColumn);
	} else if (viewColumn > viewColumn2) {
	  columnModel.moveColumn(viewColumn2, viewColumn - 1);
	} 
      }
    };
    JTableHeader header = jtable.getTableHeader(); 
    header.addMouseListener(orderedSorterMouseListener); 
  }   
}
