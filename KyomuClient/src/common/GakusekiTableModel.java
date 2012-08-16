package common;
import java.util.*;
import clients.*;

public class GakusekiTableModel extends TableModelBase { 
  public HashMap<String, String> titleMap = new HashMap<String, String>();
  public HashMap<String, String> titleValueMap = new HashMap<String, String>();
               
  private String[] columnPairs
    = {  "STUDENT_CODE",          "学生番号",
	 "FACULTY",               "所属学部 (研究科)",
	 "DEPARTMENT",            "所属学科 (専攻)",
	 "COURSE",                "コース (分野)",
	 "COURSE_2",              "コース２(IIF等)",
	 "GAKUNEN",               "現学年",
	 "STUDENT_NAME",          "学生氏名",	 
	 "SHORTER_NAME",          "学生氏名 (表示用)",
	 "KANA_NAME",             "カタカナ名",
	 "ENGLISH_NAME",          "English Name", 
	 "DATE_OF_BIRTH",         "生年月日",
	 "GENDER",                "性別",
	 "STUDENT_TYPE",          "学生タイプ",
	 "STUDENT_STATUS",        "在籍状況",
	 "CURRICULUM_YEAR",       "適用履修課程表",
	 "SUPERVISOR",            "指導教官",
	 "HOME_TYPE",             "(本人) 住所区分",
	 "HOME_POST_CODE",        "(本人) 郵便番号",
	 "HOME_ADDRESS",          "(本人) 現住所",
	 "HOME_PHONE_1",          "(本人) 電話番号",
	 "HOME_PHONE_2",          "(本人) 電話番号2 (携帯等)",
         "MAIL_ADDRESS",          "Mail Address",
	 "GUARANTOR_NAME",        "保証人名",
	 "GUARANTOR_TYPE",        "保証人区分",
	 "GUARANTOR_POST_CODE",   "保証人郵便番号",
	 "GUARANTOR_ADDRESS",     "保証人住所",
	 "GUARANTOR_PHONE",       "保証人電話番号",
	 "ENTRANCE_EXAM",         "入学試験",
	 "ENTRANCE_DATE",         "入学年月日",
	 "COURSE_CHANGE_DATE",    "コース変更年月日",
	 "ID_CARD_COUNT",         "学生証発効番号",
	 "TRUE_FACULTY",          "正式の所属学部 (大学院)",
	 "TRUE_DEPARTMENT",       "正式の所属学科 (専攻)"
	 };  

  public GakusekiTableModel(CommonInfo commonInfo, 
			    String serviceName,
			    String panelID,
			    ArrayList<String> columnTitleList,
			    ArrayList<String> columnCodeList,
			    ArrayList<String> columnDisplayList) {
    super(commonInfo, serviceName, panelID, columnTitleList, 
	  columnCodeList, columnDisplayList);

    int len = columnPairs.length / 2;
    for (int i = 0; i < len; i++) {
      titleMap.put(columnPairs[i * 2], columnPairs[i * 2 + 1]);
    }
  }
   
  public void setParamValues(String STUDENT_CODE) { 
    listOfRows.clear();
    String answer = commonInfo.commonInfoMethods.getStudentGakuseki(STUDENT_CODE);
    if (answer == null) {
      fireTableChanged(null); 
      return;
    }

//    CellObject cobj;
    titleValueMap.clear();
    try {
      String[] lines = answer.split("\\$");
      for (String line : lines) {
	String[] tokens = line.split("\\|");
	String itemTitle = tokens[0];
	String itemData = tokens[1];
	ArrayList<CellObject> rowList = new ArrayList<CellObject>();
	rowList.add(getItemName(itemTitle));
	rowList.add(getItemData(itemTitle, itemData));
	listOfRows.add(rowList);
	titleValueMap.put(itemTitle, itemData);
      }
    } catch (Exception e) {
      commonInfo.showMessage("Error in GakumuTableModel: " + e.toString());
    }
    fireTableChanged(null); 
  }
 
  public CellObject getItemName(String itemTitle) {
    CellObject cobj = new CellObject(itemTitle, titleMap.get(itemTitle));
    if (itemTitle.equals("HOME_TYPE")) {
      cobj.setHighlighted();
    } else if (itemTitle.equals("HOME_POST_CODE")) {
      cobj.setHighlighted();
    } else if (itemTitle.equals("HOME_ADDRESS")) {
      cobj.setHighlighted();
    } else if (itemTitle.equals("HOME_PHONE_1")) {
      cobj.setHighlighted();      
    } else if (itemTitle.equals("HOME_PHONE_2")) {
      cobj.setHighlighted();
    } else if (itemTitle.equals("MAIL_ADDRESS")) {
      cobj.setHighlighted();
    } else if (itemTitle.equals("ENGLISH_NAME")) {
      cobj.setHighlighted();
    }  
    return cobj;
  }  

  public CellObject getItemData(String itemTitle, String itemData) {
    String key = itemTitle;
    String val = itemData;
    if ((key.indexOf("DATE") != -1) && (val.length() > 8)) {
      val = val.substring(0, val.indexOf(" "));
    } else if (commonInfo.gakumuCodeMapContains(key, val)) {
      val = commonInfo.getGakumuCodeName(key, val);
    } else if (key.equals("SUPERVISOR")) {
      String name = commonInfo.commonInfoMethods.getStaffName(val);   
      if (name != null) {
	val = name;
      }
    }    
    return new CellObject(itemData, val);
  }  
}
  
