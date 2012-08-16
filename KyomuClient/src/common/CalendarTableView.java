package common;

import clients.*;
import java.util.*;
import java.awt.*;
import javax.swing.*;
import javax.swing.border.*;

public class CalendarTableView extends TableViewBase {
  private CommonInfo commonInfo;
  private TabbedPaneBase tabbedPane;
//  private DataPanelBase  dataPanel;
// private String serviceName;
//  private String nodePath;
//  private String panelID;

  public CalendarTableView(String tableViewType,
			   String serviceName,
			   String nodePath,
			   String panelID,
			   CommonInfo commonInfo, 
			   TabbedPaneBase tabbedPane,
			   DataPanelBase dataPanel) {
    super(tableViewType, 
	  serviceName, nodePath, panelID,
	  commonInfo, tabbedPane, dataPanel, 
	  "calendar");
    this.commonInfo = commonInfo;
    this.tabbedPane = tabbedPane;
 //   this.dataPanel = dataPanel;

    setSwitchCode();
    setQueryParams();
    tableModel = new CalendarTableModel(commonInfo, serviceName, panelID,
					columnTitleList,
					columnCodeList,
					columnDisplayList);
    setModel(tableModel);
    setTableView();
  }

  public void setMonday(int[] rows) {
    int cnt = 0;
    for (int row : rows) {
      String day   = getCodeAt(row, "DAY");
      String year  = tabbedPane.getValueFromColumnCodeMap("YEAR");
      String month = tabbedPane.getValueFromColumnCodeMap("MONTH");          
      int res = commonInfo.commonInfoMethods.setClassWeek(year, month, day, "1");
      cnt += res;
    }
    if (cnt > 0) {
      refreshTable();
    }
  }

  public void setTuesday(int[] rows) {
    int cnt = 0;
    for (int row : rows) {
      String day   = getCodeAt(row, "DAY");
      String year  = tabbedPane.getValueFromColumnCodeMap("YEAR");
      String month = tabbedPane.getValueFromColumnCodeMap("MONTH");          
      int res = commonInfo.commonInfoMethods.setClassWeek(year, month, day, "2");
      cnt += res;
    }
    if (cnt > 0) {
      refreshTable();
    }
  }

  public void setWednesday(int[] rows) {
    int cnt = 0;
    for (int row : rows) {
      String day   = getCodeAt(row, "DAY");
      String year  = tabbedPane.getValueFromColumnCodeMap("YEAR");
      String month = tabbedPane.getValueFromColumnCodeMap("MONTH");          
      int res = commonInfo.commonInfoMethods.setClassWeek(year, month, day, "3");
      cnt += res;
    }
    if (cnt > 0) {
      refreshTable();
    }
  }

  public void setThursday(int[] rows) {
    int cnt = 0;
    for (int row : rows) {
      String day   = getCodeAt(row, "DAY");
      String year  = tabbedPane.getValueFromColumnCodeMap("YEAR");
      String month = tabbedPane.getValueFromColumnCodeMap("MONTH");          
      int res = commonInfo.commonInfoMethods.setClassWeek(year, month, day, "4");
      cnt += res;
    }
    if (cnt > 0) {
      refreshTable();
    }
  }

  public void setFriday(int[] rows) {
    int cnt = 0;
    for (int row : rows) {
      String day   = getCodeAt(row, "DAY");
      String year  = tabbedPane.getValueFromColumnCodeMap("YEAR");
      String month = tabbedPane.getValueFromColumnCodeMap("MONTH");          
      int res = commonInfo.commonInfoMethods.setClassWeek(year, month, day, "5");
      cnt += res;
    }
    if (cnt > 0) {
      refreshTable();
    }
  }

  public void deleteClassWeek(int[] rows) {
    int cnt = 0;
    for (int row : rows) {
      String day   = getCodeAt(row, "DAY");
      String year  = tabbedPane.getValueFromColumnCodeMap("YEAR");
      String month = tabbedPane.getValueFromColumnCodeMap("MONTH");          
      int res = commonInfo.commonInfoMethods.deleteClassWeek(year, month, day);
      cnt += res;
    }
    if (cnt > 0) {
      refreshTable();
    }
  }

  public void addHolidayInfo(int[] rows) {
    ArrayList<Object> list = new ArrayList<Object>();
    list.add(" 学年暦に記入すべき休日情報を指定して下さい。 ");
    list.add(" ");

    TabbedPaneBase tab = new TabbedPaneBase("HolidaySelector", "root", commonInfo, null, null);
    tab.setPreferredSize(new Dimension(500, 500));
    tab.pageOpened();
    list.add(tab);

    int ans =  JOptionPane.showOptionDialog(commonInfo.getFrame(),
					    list.toArray(), 
					    "休日情報の設定", 
					    JOptionPane.OK_CANCEL_OPTION,
					    JOptionPane.QUESTION_MESSAGE,
					    null,
					    null,
					    null);
    if (ans == JOptionPane.OK_OPTION) {
      String holidayInfo = tab.getValueFromColumnCodeMap("HOLIDAY_INFO");
      
      int cnt = 0;
      for (int row : rows) {
	String day   = getCodeAt(row, "DAY");
	String year  = tabbedPane.getValueFromColumnCodeMap("YEAR");
	String month = tabbedPane.getValueFromColumnCodeMap("MONTH");
	int res = commonInfo.commonInfoMethods.addHolidayInfo(year, month, day, holidayInfo);
	cnt += res;
      }
      if (cnt > 0) {
	refreshTable();
      }
    }
  }

  public void deleteHolidayInfo(int[] rows) {
    int cnt = 0;
    for (int row : rows) {
      String day   = getCodeAt(row, "DAY");
      String year  = tabbedPane.getValueFromColumnCodeMap("YEAR");
      String month = tabbedPane.getValueFromColumnCodeMap("MONTH");        
      int res = commonInfo.commonInfoMethods.deleteHolidayInfo(year, month, day);
      cnt += res;
    }
    if (cnt > 0) {
      refreshTable();
    }
  }

  public void addSchoolEvent(int[] rows) {
    ArrayList<Object> list = new ArrayList<Object>();
    list.add(" 学年暦に記入すべき学内行事を指定して下さい。 ");
    list.add(" ");

    TabbedPaneBase tab = new TabbedPaneBase("SchoolEventSelector", "root", commonInfo, null, null);
    tab.setPreferredSize(new Dimension(500, 500));
    tab.pageOpened();
    list.add(tab);
    
    int ans =  JOptionPane.showOptionDialog(commonInfo.getFrame(),
					    list.toArray(), 
					    "学内行事の設定", 
					    JOptionPane.OK_CANCEL_OPTION,
					    JOptionPane.QUESTION_MESSAGE,
					    null,
					    null,
					    null);
    if (ans == JOptionPane.OK_OPTION) {
      String schoolEvent = tab.getValueFromColumnCodeMap("SCHOOL_EVENT"); 
      
      int cnt = 0;
      for (int row : rows) {
	String day   = getCodeAt(row, "DAY");
	String year  = tabbedPane.getValueFromColumnCodeMap("YEAR");
	String month = tabbedPane.getValueFromColumnCodeMap("MONTH");
	int res = commonInfo.commonInfoMethods.addSchoolEvent(year, month, day, schoolEvent);
	cnt += res;
      }
      if (cnt > 0) {
	refreshTable();
      }
    }
  }
     
  public void deleteSchoolEvent(int[] rows) {
    int cnt = 0;
    for (int row : rows) {
      String day   = getCodeAt(row, "DAY");
      String year  = tabbedPane.getValueFromColumnCodeMap("YEAR");
      String month = tabbedPane.getValueFromColumnCodeMap("MONTH");        
      int res = commonInfo.commonInfoMethods.deleteSchoolEvent(year, month, day);
      cnt += res;
    }
    if (cnt > 0) {
      refreshTable();
    }
  }

  public void addNewHolidayInfo(int[] rows) {  
    ArrayList<Object> list = new ArrayList<Object>();
    list.add(" 学年暦に記入すべき新しい休日情報のテキストを ");
    list.add(" 入力して下さい。");

    JTextField editor = new JTextField(" ");
    editor.setEditable(true); 
    editor.setFont(new Font("DialogInput", Font.PLAIN, 12)); 
    editor.setBorder(new TitledBorder("新しい休日情報")); 
    list.add(editor);
    
    int ans =  JOptionPane.showOptionDialog(commonInfo.getFrame(),
					    list.toArray(), 
					    "休日情報の追加", 
					    JOptionPane.OK_CANCEL_OPTION,
					    JOptionPane.QUESTION_MESSAGE,
					    null,
					    null,
					    null);
    if (ans == JOptionPane.OK_OPTION) {
      String holidayInfo = editor.getText().trim();   
      
      int cnt = 0;
      for (int row : rows) {
	String day   = getCodeAt(row, "DAY");
	String year  = tabbedPane.getValueFromColumnCodeMap("YEAR");
	String month = tabbedPane.getValueFromColumnCodeMap("MONTH");
	int res = commonInfo.commonInfoMethods.addHolidayInfo(year, month, day, holidayInfo);
	cnt += res;
      } 
      if (cnt > 0) {
	refreshTable();
      }
    } 
  } 

  public void addNewSchoolEvent(int[] rows) { 
    ArrayList<Object> list = new ArrayList<Object>();
    list.add(" 学年暦に記入すべき新しい学内行事のテキストを ");
    list.add(" 入力して下さい。");

    JTextField editor = new JTextField(" ");
    editor.setEditable(true); 
    editor.setFont(new Font("DialogInput", Font.PLAIN, 12)); 
    editor.setBorder(new TitledBorder("新しい学内行事")); 
    list.add(editor);
    
    int ans =  JOptionPane.showOptionDialog(commonInfo.getFrame(),
					    list.toArray(), 
					    "学内行事の追加", 
					    JOptionPane.OK_CANCEL_OPTION,
					    JOptionPane.QUESTION_MESSAGE,
					    null,
					    null,
					    null);
    if (ans == JOptionPane.OK_OPTION) {
      String schoolEvent = editor.getText().trim();    
       
      int cnt = 0;
      for (int row : rows) {
	String day   = getCodeAt(row, "DAY");
	String year  = tabbedPane.getValueFromColumnCodeMap("YEAR");
	String month = tabbedPane.getValueFromColumnCodeMap("MONTH");
	int res = commonInfo.commonInfoMethods.addSchoolEvent(year, month, day, schoolEvent);
	cnt += res;
      } 
      if (cnt > 0) {
	refreshTable();
      }
    } 
  } 
  
}
