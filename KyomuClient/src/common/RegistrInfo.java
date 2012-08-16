package common;

import clients.*;
import java.util.*;
//import java.awt.*;
//import javax.swing.*;

public class RegistrInfo {
  private CommonInfo commonInfo;
  private CommonInfoMethods commonInfoMethods;
  private ServerConnectionBase serverConn;

  public RegistrInfo(CommonInfo commonInfo) {
    this.commonInfo = commonInfo;
    this.commonInfoMethods = commonInfo.commonInfoMethods;
    this.serverConn = commonInfo.serverConn;
    this.undergradRegistrUpperBound = commonInfo.undergradRegistrUpperBound;
    this.graduateRegistrUpperBound = commonInfo.graduateRegistrUpperBound;
  }

  //***** 履修登録しようとする学生に関するデータ **********//  
  private int studentGakunen;
  private int zaigakuYears;

  private HashSet<String> shutokuSubjectSet = new HashSet<String>();         // 修得科目の集合
  private HashSet<String> ninteiSubjectSet  = new HashSet<String>();         // 認定科目の集合
  private HashSet<String> rtorokuSubjectClassSet = new HashSet<String>();    // 履修登録科目の集合
  private HashSet<String> registrSubjectClassSet = new HashSet<String>();    // 仮履修登録科目の集合
  private HashSet<String> registrSubjectSet = new HashSet<String>();         // 仮履修登録科目の集合2

  private double  rtorokuJikanUnits = 0;               // 履修登録科目(集中以外)の単位数
  private HashMap<String, String> rtorokuJikanMap = new HashMap<String, String>();    
        // week+"|"+hour =>  履修登録科目
	//  key = week + "|" + hour;
	//  value = subjectCode +"|"+ classCode +"|"+ subjectName +"|"+ teacherName;

  private double  registrJikanUnits = 0;               // 仮履修登録科目(集中以外)の単位数
  private HashMap<String, String> registrJikanMap = new HashMap<String, String>();    
        // week+"|"+hour  =>  仮履修登録科目
	//  key = week + "|" + hour;
	//  value = subjectCode +"|"+ classCode +"|"+ subjectName +"|"+ teacherName;

  private HashMap<String, String> jikanwariSubjectMap = new HashMap<String, String>();
        //  学生の所属学科の時間割に存在する科目 subjectCode +"|"+ classCode に対して
        //  その科目の kubunCode+"|"+reqCode+"|"+unit を対応させる。
        //  履修課程表に存在しない科目 (新設科目) の履修登録に用いられる。


  //***** 履修登録に関連するデータ **********//  
  public boolean registrPeriodFlag = false;                 // 履修登録期間
  public boolean registrAmendPeriodFlag = false;            // 修正履修登録期間
  public boolean freshmanRegistrAmendPeriodFlag = false;    // 新入生修正申告期間

  public boolean registrAllowedPeriod = false;

  private double undergradRegistrUpperBound = 24.0;    // 学期毎に履修できる(集中を除く)単位数
  private double graduateRegistrUpperBound = 16.0;     // 学期毎に履修できる(集中を除く)単位数

  private HashSet<String> overlapSubjectClassSet = new HashSet<String>();    
        // 重複可能科目の集合
	//  key = subjectCode+"|"+classCode;

  private HashMap<String, String> yomikaeMap = new HashMap<String, String>();      
        // 開講科目 ==> 読替科目 
        // すなわち、学生が履修登録しようとする (開講される) 科目に対して、その科目を
        // 履修することによって修得することになる履修課程表の (本年度は開講されない)
        // 授業科目が対応する。

  private HashMap<String, String> curriculumReqMap     = new HashMap<String, String>();   
  private HashMap<String, String> curriculumGakunenMap = new HashMap<String, String>();
  private HashMap<String, String> curriculumInfoMap    = new HashMap<String, String>();
        // 学生に適用される履修課程表の各科目に対して次のデータを対応させる。
        //   科目 ==> 単位区分 
        //   科目 ==> 履修年次 
        //   科目 ==> その科目の全属性


  public int registrSubject(String schoolYear, String subjectCode, String classCode, String unit,
			    String subjectName, String teacherName) {    
    if (curriculumInfoMap.containsKey(subjectCode)) {
      String line = curriculumInfoMap.get(subjectCode);    
      String[] tokens = line.split("\\|");
      String kubunCode3   = tokens[0];
      String reqCode3     = tokens[1];
      String unit3        = tokens[2];      
      int res = commonInfoMethods.registrSubject(schoolYear, subjectCode, classCode, kubunCode3,
						 reqCode3, unit3, null, null);
      if (res == 1) {
	commonInfo.showMessageLong("「" + subjectName + " (" + teacherName + ")」は $ 仮履修登録されました。");
	return res;
      }
    } else {
      if (yomikaeMap.containsKey(subjectCode)) {
	String subjectCode2 = yomikaeMap.get(subjectCode);  
	if (shutokuSubjectSet.contains(subjectCode2)) {
	  commonInfo.showMessageLong("「" + subjectName + "」は「新旧科目の読み替え」の対象となります。$$ あなたの場合は、「読み替え対象の科目」を既に修得しているため、$ この科目を新たに履修登録することはできません。");
	  return 0;
	} 	
	if (curriculumInfoMap.containsKey(subjectCode2)) {
	  String line = curriculumInfoMap.get(subjectCode2); 
	  String[] tokens = line.split("\\|");
	  String kubunCode3   = tokens[0];
	  String reqCode3     = tokens[1];
	  String unit3        = tokens[2];
	  int res = commonInfoMethods.registrSubject(schoolYear, subjectCode, classCode, kubunCode3,
						     reqCode3, unit3, "新旧科目の読替え", subjectCode2);
	  if (res == 1) {
	    commonInfo.showMessageLong("「" + subjectName + " (" + teacherName + ")」は $ 「新旧科目の読替え」に関する設定を行った上で $ 仮履修登録されました。");
	    return res;
	  }
	}
      } else {	
	if (jikanwariSubjectMap.containsKey(subjectCode)) {
	  boolean ans = commonInfo.showNoticeLong("あなたが履修申告した「"+subjectName+"」$ は、あなたの所属学科が新設した「新設科目」です。$ 一般的には、新設科目を履修登録した場合、その単位区分は、時間割上の設定が $ 「必修」である場合であっても「選択科目」となります。$ $ 履修登録をしますか？ ");
	  if (!ans) {
	    return 0;
	  }	  
	  String line = jikanwariSubjectMap.get(subjectCode); 
	  String[] tokens = line.split("\\|");
	  String kubunCode3   = tokens[0];
	  String reqCode3     = tokens[1];
	  String unit3        = tokens[2];   
	  if (reqCode3.equals("9")) {	    
	    int res = commonInfoMethods.registrSubject(schoolYear, subjectCode, classCode, kubunCode3,
						       reqCode3, unit3, "新設科目", null);
	    if (res == 1) {
	      commonInfo.showMessageLong("「" + subjectName + " (" + teacherName + ")」は $ 「新設科目としての設定」を行った上で $ 仮履修登録されました。");  
	      return res;  
	    }	
	  } else {    
	    int res = commonInfoMethods.registrSubject(schoolYear, subjectCode, classCode, kubunCode3,
						       "4", unit3, "新設科目", null);
	    if (res == 1) {
	      commonInfo.showMessageLong("「" + subjectName + " (" + teacherName + ")」は $ 「新設科目としての設定」を行った上で $ 仮履修登録されました。$$ 単位区分が「選択」では不都合な学生は、学務係に申し出て変更する必要があります。");
	      return res;
	    }	
	  } 
	} else {  
	  boolean ans = commonInfo.showNoticeLong("「" + subjectName + "」は、履修課程表に存在しない科目であり、$ また、「新旧科目の読み替え」の対象にもなっていないため、「査定外」の「他学科科目」として $ 扱われることになります。$$  履修登録をしますか？ ");
	  if (!ans) {
	    return 0;
	  }    
	  int res = commonInfoMethods.registrSubject(schoolYear, subjectCode, classCode, "562",
						     "9", unit, "他学科科目", null);
	  if (res == 1) {
	    commonInfo.showMessageLong("「" + subjectName + " (" + teacherName + ")」は $ 「他学科科目」として仮履修登録されました。 $ $ この科目の単位区分が「査定外」では不都合な学生は、学務係に申し出る必要があります。");  
	    return res;  
	  }	
	}
      }
    }
    commonInfo.showMessageLong("「" + subjectName + " (" + teacherName + ")」は $ 仮履修登録されませんでした。");
    return 0;
  }

  public int cancelRegistrSubject(String schoolYear, String subjectCode, String classCode,
				  String subjectName, String teacherName) {
    int res = commonInfoMethods.cancelRegistration(schoolYear, subjectCode, classCode);
    if (res == 1) {
      commonInfo.showMessageLong("「" + subjectName + " (" + teacherName + ")」の $ 仮履修登録は削除されました。");
      return res;
    } else {
      commonInfo.showMessageLong("「" + subjectName + " (" + teacherName + ")」の $ 仮履修登録は削除されませんでした。");
      return 0;
    }
  }

  // 履修登録しようとする科目 (subjectCode, classCode, unit) が「履修登録を許される」科目
  // である場合に true を返す。
  public boolean checkRegistrAllowed(String subjectCode, String classCode, String unit) {
    ArrayList<String> weekHourList = getSubjectWeekHour(subjectCode, classCode);
    if (weekHourList != null) {
      boolean res = checkOverlap(subjectCode, classCode, weekHourList);
      if (!res) return false;

      double unitDouble = Double.parseDouble(unit);
      if ((commonInfo.STUDENT_FACULTY.equals("11")) && 
	  (commonInfo.STUDENT_COURSE.equals("10")) && 
	  (Integer.parseInt(commonInfo.STUDENT_CURRICULUM_YEAR) >= 2001)) {
	double sum = unitDouble + rtorokuJikanUnits + registrJikanUnits;
	if (sum > undergradRegistrUpperBound) {
	  commonInfo.showMessageLong("この科目を履修登録すると、一学期に履修登録することのできる $ 単位数の上限である 24 単位を超過します。 $ $ この上限値を超えて履修登録する必要がある場合には学務係に申し出て下さい。 ");
	  return false;
	}
      } else if ((!commonInfo.STUDENT_FACULTY.equals("11")) && 
		 (Integer.parseInt(commonInfo.STUDENT_CURRICULUM_YEAR) >= 2005)) {
	double sum = unitDouble + rtorokuJikanUnits + registrJikanUnits;
	if (sum > graduateRegistrUpperBound) {
	  commonInfo.showMessageLong("この科目を履修登録すると、一学期に履修登録することのできる $ 単位数の上限である 16 単位を超過します。 $ $ この上限値を超えて履修登録する必要がある場合には学務係に申し出て下さい。 ");
	  return false;
	}
      }      
    }

    boolean res = checkRishuNenji(subjectCode);
    if (!res) return false;

    return true;
  }
  
  // 履修登録しようとする科目 subjectCode の履修年次が、学生に
  // 履修登録することを許される履修年次である場合に true を返す。
  public boolean checkRishuNenji(String subjectCode) {
    if (curriculumInfoMap.containsKey(subjectCode)) {
      String reqCode = curriculumReqMap.get(subjectCode);
      String gakunen = curriculumGakunenMap.get(subjectCode);
      int gakunenInt = Integer.parseInt(gakunen);
      if (reqCode.equals("1")) {
	if (studentGakunen < gakunenInt) {
	  commonInfo.showMessageLong("上級学年で開講される必修科目を履修登録する場合には、$ 学務係に申し出て下さい。$ 所属学科の教務委員の許可が必要です。");
	  return false;
	}
      } else {
	if (zaigakuYears + 1 < gakunenInt ) {
	  commonInfo.showMessageLong("上級学年で開講される選択必修科目/選択科目履修登録する場合 $ には、学務係に申し出て下さい。$ 所属学科の教務委員の許可が必要です。");
	  return false;
	}
      }	
      return true;
    } else {
      if (yomikaeMap.containsKey(subjectCode)) {
	String subjectCode2 = yomikaeMap.get(subjectCode);
	if (curriculumInfoMap.containsKey(subjectCode2)) {
	  String reqCode = curriculumReqMap.get(subjectCode2);
	  String gakunen = curriculumGakunenMap.get(subjectCode2);
	  int gakunenInt = Integer.parseInt(gakunen);
	  if (reqCode.equals("1")) {
	    if (studentGakunen < gakunenInt) {
	      commonInfo.showMessageLong("上級学年で開講される必修科目を履修登録する場合には、$ 学務係に申し出て下さい。$ 所属学科の教務委員の許可が必要です。");
	      return false;
	    }
	  } else {
	    if (zaigakuYears + 1 < gakunenInt ) {
	      commonInfo.showMessageLong("上級学年で開講される選択必修科目/選択科目履修登録する場合 $ には、学務係に申し出て下さい。$ 所属学科の教務委員の許可が必要です。");
	      return false;
	    }
	  }	
	}
      }      
      return true;
    }
  }
   
  // 履修登録しようとする科目 (subjectCode, classCode) のコマが履修登録科目および
  // 仮履修登録科目のコマと重複していないことを確認する。
  // 重複していない場合 (重複していても重複可能科目である場合) に true を返す。
  public boolean checkOverlap(String subjectCode, String classCode, 
			      ArrayList<String> weekHourList) {   
    boolean overlapAllowed = false;

    String key = subjectCode+"|"+classCode;
    if (overlapSubjectClassSet.contains(key)) {
      overlapAllowed = true;
    }

    for (int i = 0; i < weekHourList.size(); i++) {
      String weekHour = weekHourList.get(i);
      if (rtorokuJikanMap.containsKey(weekHour)) {
	String val = rtorokuJikanMap.get(weekHour);
	String[] tokens = val.split("\\|");
	String subjectCode2 = tokens[0]; 
	String classCode2   = tokens[1]; 
	String subjectName2 = tokens[2]; 
	String teacherName2 = tokens[3]; 
	if (!overlapAllowed) {
	  String msg = "履修登録しようとする科目のコマは、履修登録されている科目 $「" + subjectName2 + " (" + teacherName2 + ")」 $ のコマと重複しています。";
	  commonInfo.showMessageLong(msg);
	  return false;
	} else {
	  String key3 = subjectCode2+"|"+classCode2;
	  if (!overlapSubjectClassSet.contains(key3)) {
	    String msg = "履修登録しようとする科目のコマは、履修登録されている科目 $「" + subjectName2 + " (" + teacherName2 + ")」 $ のコマと重複しています。";
	    commonInfo.showMessageLong(msg);
	    return false;
	  }
	}
      }

      if (registrJikanMap.containsKey(weekHour)) {
	String val = registrJikanMap.get(weekHour);
	String[] tokens = val.split("\\|");
	String subjectCode2 = tokens[0]; 
	String classCode2   = tokens[1]; 
	String subjectName2 = tokens[2]; 
	String teacherName2 = tokens[3];
	if (!overlapAllowed) {
	  String msg = "履修登録しようとする科目のコマは、仮履修登録されている科目 $「" + subjectName2 + " (" + teacherName2 + ")」 $ のコマと重複しています。";
	  commonInfo.showMessageLong(msg);
	  return false;
	} else {
	  String key3 = subjectCode2+"|"+classCode2;
	  if (!overlapSubjectClassSet.contains(key3)) {
	    String msg = "履修登録しようとする科目のコマは、仮履修登録されている科目 $「" + subjectName2 + " (" + teacherName2 + ")」 $ のコマと重複しています。";
	    commonInfo.showMessageLong(msg);
	    return false;
	  }
	}
      }
    }
    return true;
  }

  public int getSubjectStatus(String SUBJECT_CODE, String CLASS_CODE) {
    if (shutokuSubjectSet.contains(SUBJECT_CODE)) {
      return 1;
    } else if (ninteiSubjectSet.contains(SUBJECT_CODE)) {
      return 2;
    } else {
      String key = SUBJECT_CODE + "|" + CLASS_CODE;
      if (rtorokuSubjectClassSet.contains(key)) {
	return 3;
      } else if (registrSubjectClassSet.contains(key)) {
	return 4;
      } else if (registrSubjectSet.contains(SUBJECT_CODE)) {
	return 5;
      } else {
	return 0;
      }
    }
  }  

  public ArrayList<String> getSubjectWeekHour(String SUBJECT_CODE, String CLASS_CODE) {
    String key = "QUERY|COMMON_QUERY|querySubjectWeekHour";
    String paramValues = SUBJECT_CODE+"|"+CLASS_CODE+"|"+commonInfo.thisSchoolYear+"|"+commonInfo.thisSemester;
    String answer = serverConn.queryCommon(key, paramValues);

    ArrayList<String> weekHourList = new ArrayList<String>();
    if (answer == null) {
      return null;
    } else {
      String[] lines = answer.split("\\$");
      for (String line : lines) {
	String[] tokens = line.split("\\|");
	String week = tokens[0];
	String hour = tokens[1];
	weekHourList.add(week+"|"+hour);
      }
      return weekHourList;      
    }
  }

  public void setZaigakuYears() {
    if ((commonInfo.STUDENT_FACULTY.equals("11")) && (commonInfo.STUDENT_COURSE.equals("30"))) {
      zaigakuYears = commonInfo.thisSchoolYear - Integer.parseInt(commonInfo.STUDENT_CURRICULUM_YEAR) + 1 + 2;
    } else {
      zaigakuYears = commonInfo.thisSchoolYear - Integer.parseInt(commonInfo.STUDENT_CURRICULUM_YEAR) + 1;
    }
    studentGakunen = Integer.parseInt(commonInfo.STUDENT_GAKUNEN);
  }

  public void setSeisekiRTorokuInfo() { 
    setZaigakuYears();

    String answer, subjectCode, ninteiFlag, classCode, unit, key, value, paramValues;
    String week, hour, subjectName, teacherName;

    shutokuSubjectSet.clear();
    ninteiSubjectSet.clear();
    key = "QUERY|COMMON_QUERY|querySeisekiList";
    paramValues = commonInfo.STUDENT_CODE;
    answer = serverConn.queryCommon(key, paramValues);

    if (answer != null) {
      String[] lines = answer.split("\\$");
      for (String line : lines) {
	String[] tokens = line.split("\\|");
	subjectCode = tokens[0];
	ninteiFlag  = tokens[1];
	if (ninteiFlag.equals("A")) {
	  shutokuSubjectSet.add(subjectCode);
	} else if ((ninteiFlag.equals("Q")) || (ninteiFlag.equals("J"))) {
	  ninteiSubjectSet.add(subjectCode);
	}
      }
    }

    paramValues = commonInfo.STUDENT_CODE;
    key = "QUERY|COMMON_QUERY|querySeisekiList2";
    answer = serverConn.queryCommon(key, paramValues);
    if (answer != null) {
      String[] lines = answer.split("\\$");
      for (String line : lines) {
	String[] tokens = line.split("\\|");
	subjectCode = tokens[0];
	ninteiFlag  = tokens[1];
	if (ninteiFlag.equals("A")) {
	  shutokuSubjectSet.add(subjectCode);
	} 
      }
    }

    rtorokuSubjectClassSet.clear();
    paramValues = commonInfo.STUDENT_CODE+"|"+commonInfo.thisSchoolYear;
    key = "QUERY|COMMON_QUERY|queryRTorokuList";
    answer = serverConn.queryCommon(key, paramValues);
    if (answer != null) {
      String[] lines = answer.split("\\$");
      for (String line : lines) {
	String[] tokens = line.split("\\|");
	subjectCode = tokens[0];
	classCode   = tokens[1];
	key = subjectCode + "|" + classCode;
	rtorokuSubjectClassSet.add(key);
      }
    }

    if (registrAllowedPeriod) {
      overlapSubjectClassSet.clear();
      paramValues = ""+commonInfo.thisSchoolYear + "|" + commonInfo.thisSemester;
      key = "QUERY|COMMON_QUERY|queryOverlapSubjects";
      answer = serverConn.queryCommon(key, paramValues);
      if (answer != null) {
	String[] lines = answer.split("\\$");
	for (String line : lines) {
	  String[] tokens = line.split("\\|");
	  subjectCode = tokens[0];
	  classCode   = tokens[1];
	  key = subjectCode + "|" + classCode;
	  overlapSubjectClassSet.add(key);
	}
      }

      rtorokuJikanMap.clear();
      rtorokuJikanUnits = 0.0;
      paramValues = commonInfo.STUDENT_CODE+"|"+commonInfo.thisSchoolYear+"|"+commonInfo.thisSemester;
      key = "QUERY|COMMON_QUERY|queryRTorokuJikanwari";
      answer = serverConn.queryCommon(key, paramValues);
      if (answer != null) {
	HashSet<String> tempSet = new HashSet<String>();
	String[] lines = answer.split("\\$");
	for (String line : lines) {
	  String[] tokens = line.split("\\|");
	  week        = tokens[0]; 
	  hour        = tokens[1]; 
	  subjectCode = tokens[2]; 
	  classCode   = tokens[3]; 
	  unit        = tokens[4]; 
	  subjectName = tokens[5]; 
	  teacherName = tokens[6];
	  key = week + "|" + hour;
	  value = subjectCode +"|"+ classCode +"|"+ subjectName +"|"+ teacherName;
	  rtorokuJikanMap.put(key, value);
	  if (!tempSet.contains(subjectCode +"|"+ classCode)) {
	    rtorokuJikanUnits += Double.parseDouble(unit);
	    tempSet.add(subjectCode +"|"+ classCode);
	  }
	}
      }

      yomikaeMap.clear();
      key = "QUERY|COMMON_QUERY|queryYomikaeSubject";
      paramValues = ""+commonInfo.thisSchoolYear+"|"+commonInfo.STUDENT_CURRICULUM_YEAR+"|"+commonInfo.STUDENT_FACULTY+"|"+commonInfo.STUDENT_DEPARTMENT+"|"+commonInfo.STUDENT_COURSE;
      answer = serverConn.queryCommon(key, paramValues);
      if (answer != null) {
	String[] lines = answer.split("\\$");
	for (String line : lines) {
	  String[] tokens = line.split("\\|");
	  String subjectCode1 = tokens[0];
	  String subjectCode2 = tokens[1];
	  yomikaeMap.put(subjectCode1, subjectCode2);
	}
      } 

      curriculumGakunenMap.clear();
      curriculumReqMap.clear();
      curriculumInfoMap.clear();
      key = "QUERY|COMMON_QUERY|queryCurriculumInfo";
      paramValues = commonInfo.STUDENT_CURRICULUM_YEAR+"|"+commonInfo.STUDENT_FACULTY+"|"+commonInfo.STUDENT_DEPARTMENT+"|"+commonInfo.STUDENT_COURSE;
      answer = serverConn.queryCommon(key, paramValues);
      if (answer != null) {
	String[] lines = answer.split("\\$");
	for (String line : lines) {
	  String[] tokens = line.split("\\|");
	  String subjectCode3 = tokens[0];
	  String kubunCode3 = tokens[1];
	  String reqCode3   = tokens[2];
	  String unit3      = tokens[3];
	  String gakunen3   = tokens[4];
	  curriculumReqMap.put(subjectCode3, reqCode3);
	  curriculumGakunenMap.put(subjectCode3, gakunen3);
	  curriculumInfoMap.put(subjectCode3, kubunCode3+"|"+reqCode3+"|"+unit3);
	}
      } 
     
      jikanwariSubjectMap.clear();
      key = "QUERY|COMMON_QUERY|queryJikanwariSubject";
      paramValues = ""+commonInfo.thisSchoolYear+"|"+commonInfo.STUDENT_FACULTY+"|"+commonInfo.STUDENT_DEPARTMENT;
      answer = serverConn.queryCommon(key, paramValues);
      if (answer != null) {
	String[] lines = answer.split("\\$");
	for (String line : lines) {
	  String[] tokens = line.split("\\|");
	  String subjectCode3 = tokens[0];
	  String kubunCode3 = tokens[1];
	  String reqCode3   = tokens[2];
	  String unit3      = tokens[3];
	  jikanwariSubjectMap.put(subjectCode3, kubunCode3+"|"+reqCode3+"|"+unit3);
	}
      } 
    }
    setRegistrInfo();
  }

  public void setRegistrInfo() {
    String answer, subjectCode, classCode, unit, key, value, paramValues;
    String week, hour, subjectName, teacherName;

    registrSubjectClassSet.clear();
    registrSubjectSet.clear();
    key = "QUERY|COMMON_QUERY|queryRegistrList";
    paramValues = commonInfo.STUDENT_CODE+"|"+commonInfo.thisSchoolYear;
    answer = serverConn.queryCommon(key, paramValues);
    if (answer != null) {
      String[] lines = answer.split("\\$");
      for (String line : lines) {
	String[] tokens = line.split("\\|");
	subjectCode = tokens[0];
	classCode   = tokens[1];
	key = subjectCode + "|" + classCode;
	registrSubjectClassSet.add(key);
	registrSubjectSet.add(subjectCode);
      }
    }

    if (registrAllowedPeriod) {
      registrJikanMap.clear();
      registrJikanUnits = 0.0;
      key = "QUERY|COMMON_QUERY|queryRegistrJikanwari";
      paramValues = commonInfo.STUDENT_CODE+"|"+commonInfo.thisSchoolYear+"|"+commonInfo.thisSemester;
      answer = serverConn.queryCommon(key, paramValues);
      if (answer != null) {
	HashSet<String> tempSet = new HashSet<String>();
	String[] lines = answer.split("\\$");
	for (String line : lines) {
	  String[] tokens = line.split("\\|");
	  week        = tokens[0];
	  hour        = tokens[1];
	  subjectCode = tokens[2];
	  classCode   = tokens[3];
	  unit        = tokens[4];
	  subjectName = tokens[5];
	  teacherName = tokens[6];
	  key = week + "|" + hour;
	  value = subjectCode +"|"+ classCode +"|"+ subjectName +"|"+ teacherName;
	  registrJikanMap.put(key, value);
	  if (!tempSet.contains(subjectCode +"|"+ classCode)) {
	    registrJikanUnits += Double.parseDouble(unit);
	    tempSet.add(subjectCode +"|"+ classCode);
	  }
	}
      }
    }
  }

  public void showRegistrWarning() {
    String schoolEvent = getTodaySchoolEvent();
    int index1 = schoolEvent.indexOf("履修申告期間");
    if (index1 < 0) {
      registrPeriodFlag = false;  
    } else {
      registrPeriodFlag = true;
    }
      
    int index3 = schoolEvent.indexOf("新入生修正申告期間");
    if (index3 < 0) {
      freshmanRegistrAmendPeriodFlag = false;  
    } else {
      freshmanRegistrAmendPeriodFlag = true;
    }

    if (freshmanRegistrAmendPeriodFlag == false) {
      int index2 = schoolEvent.indexOf("修正申告期間");
      if (index2 < 0) {
	registrAmendPeriodFlag = false;  
      } else {
	registrAmendPeriodFlag = true;
      }
    }

    if (commonInfoMethods.isFreshman()) {
      if ((registrPeriodFlag) || 
	  (registrAmendPeriodFlag) ||
	  (freshmanRegistrAmendPeriodFlag)) {
	registrAllowedPeriod = true;
      } else {
	registrAllowedPeriod = false;
	commonInfo.showMessage("現在はあなたが「履修(修正)申告」できる期間ではありません。");
      }
    } else {
      if ((registrPeriodFlag) || 
	  (registrAmendPeriodFlag)) {
	registrAllowedPeriod = true;
      } else {
	registrAllowedPeriod = false;
	commonInfo.showMessage("現在はあなたが「履修(修正)申告」できる期間ではありません。");
      }
    } 
  }

  public String getTodaySchoolEvent() {
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

}
