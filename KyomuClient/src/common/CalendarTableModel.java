package common;

import clients.*;
import java.util.*;

public class CalendarTableModel extends TableModelBase { 
  private CommonInfo commonInfo;
  private String serviceName;
  private String panelID;
  private ArrayList<String> columnTitleList;
  private ArrayList<String> columnCodeList;
  private ArrayList<String> columnDisplayList;

  public CalendarTableModel(CommonInfo commonInfo, 
			    String serviceName,
			    String panelID,
			    ArrayList<String> columnTitleList,
			    ArrayList<String> columnCodeList,
			    ArrayList<String> columnDisplayList) {
    super(commonInfo, serviceName, panelID, columnTitleList,
	  columnCodeList, columnDisplayList);
    this.commonInfo = commonInfo;
    this.serviceName = serviceName;
    this.panelID = panelID;
    this.columnTitleList = columnTitleList;
    this.columnCodeList = columnCodeList;
    this.columnDisplayList = columnDisplayList;
  }

  public void setTableData(String serviceName, String panelID, String switchCode, 
			   String queryParamValues) {  
    listOfRows.clear();  
    makeCalendarTableModel(queryParamValues);
  }

  private void makeCalendarTableModel(String paramValues) { 
    String DAY, HOLIDAY_INFO, SCHOOL_EVENTS, CLASS_WEEK;
    String[] tokens = paramValues.split("\\|");
    int year  = new Integer(tokens[0]);
    int month = new Integer(tokens[1]);
    makeEmptyCalendar(year, month);  
    String answer = commonInfo.getQueryResult(serviceName, panelID, "0",
					      paramValues); 
    if (answer == null) {
      fireTableChanged(null); 
      return;
    } else {
      String[] lines = answer.split("\\$");
      for (String line : lines) {
	tokens = line.split("\\|");
	DAY           = tokens[0];
	HOLIDAY_INFO  = tokens[1];
	SCHOOL_EVENTS = tokens[2];
	CLASS_WEEK    = tokens[3];
	setValue(DAY, HOLIDAY_INFO, SCHOOL_EVENTS, CLASS_WEEK);
      }
      fireTableChanged(null);     
    }
  }
   
  public void makeEmptyCalendar(int year, int month) {
    Calendar cal = new GregorianCalendar();    
    cal.set(year, month-1, 1);  
    int lastDay = cal.getActualMaximum(Calendar.DATE);  
    int week, day = 0;
    while (day < lastDay) {
      day  = cal.get(Calendar.DATE); 
      week = cal.get(Calendar.DAY_OF_WEEK) - 1;
      LinkedList<String> linkedList = new LinkedList<String>();
      String dayString;
      if (day < 10) {
	dayString = "0" + day;
      } else {
	dayString = "" + day;
      }
      linkedList.add(dayString);
      linkedList.add("" + week);
      linkedList.add(" ");
      linkedList.add(" ");
      ArrayList<CellObject> rowList = new ArrayList<CellObject>();
      for (int i = 0; i < getColumnCount(); i++) {
	String columnDisplay = columnDisplayList.get(i);
	CellObject cobj = makeCellObject(columnDisplay, linkedList);
	rowList.add(cobj);
      }
      listOfRows.add(rowList);
      cal.roll(Calendar.DATE, true);
    }
  }

  public void setValue(String DAY, String str2, String str3, String str4) {
    int row = Integer.parseInt(DAY) - 1;
    setValueAt(new CellObject(str2), row, 2);
    setValueAt(new CellObject(str3), row, 3);
    CellObject cobj = null;
    if (str4.equals("0")) {
      cobj = new CellObject(str4, "日曜日の授業を行う");      
    } else if (str4.equals("1")) {
      cobj = new CellObject(str4, "月曜日の授業を行う");   
    } else if (str4.equals("2")) {
      cobj = new CellObject(str4, "火曜日の授業を行う");   
    } else if (str4.equals("3")) {
      cobj = new CellObject(str4, "水曜日の授業を行う");   
    } else if (str4.equals("4")) {
      cobj = new CellObject(str4, "木曜日の授業を行う");   
    } else if (str4.equals("5")) {
      cobj = new CellObject(str4, "金曜日の授業を行う");   
    } else if (str4.equals("6")) {
      cobj = new CellObject(str4, "土曜日の授業を行う");   
    } 
    setValueAt(cobj, row, 4);
  } 

}
