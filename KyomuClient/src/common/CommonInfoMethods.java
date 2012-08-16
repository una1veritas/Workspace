package common;
import javax.swing.*;
import java.util.*;
import clients.*;


public class CommonInfoMethods {
  private CommonInfo commonInfo;
  private ServerConnectionBase serverConn;
  private ServerConnectionMethods serverMethods;

  //***** 履修登録に関連するデータを管理するオブジェクト **********//  
  public RegistrInfo registrInfo = null;

  //***  Popup graphs に関連するオブジェクト ***//
  private GraphPanel5 gp5 = null;
  private GraphPanel6 gp6 = null;

  public CommonInfoMethods(CommonInfo commonInfo,
			   ServerConnectionBase serverConn) {
    this.commonInfo = commonInfo;
    this.serverConn = serverConn;
    this.serverMethods = serverConn.serverConnectionMethods;
  }

  public void setRegistrInfo() {
    registrInfo = new RegistrInfo(commonInfo);
  }


  //*****  checkMethodUtilities  ******// 
  private TabbedPaneBase tabbedPaneBase;

  public boolean checkQualification(String qualificationMethod, String param, TabbedPaneBase tabbedPane) {
    // param is ignored
    this.tabbedPaneBase = tabbedPane;
    if (qualificationMethod == null) return true;
    else return checkQualification(qualificationMethod);
  } 

  public boolean checkQualification(String qualificationMethod) {
    try {
      RetBoolean ret = new RetBoolean();
      getClass().getMethod(qualificationMethod, new Class[] { RetBoolean.class }).invoke(this, new Object[] { ret } );
      return ret.getValue();
    } catch (Exception ex) {
      commonInfo.showMessageLong("qualificationMethod is not invoked $ " + qualificationMethod);  
      return false;
    }
  }

  public void disableMethod(RetBoolean ret) {
    ret.reject();
  }

  public void gakumuStaffOnly(RetBoolean ret) {
    if (checkGakumuStaff()) {
      ret.accept();
    } else {
      ret.reject();
    }
  }

  public void higherThanTeacher(RetBoolean ret) {
    if (checkHigherThanTeacher()) {
      ret.accept();
    } else {
      ret.reject();
    }
  }

  public void checkIIFCourse(RetBoolean ret) { 
    if (!commonInfo.structControlDebugMode) {						 
      String course2 = commonInfo.getValueFromCommonCodeMap("MY_COURSE_2");
      if (course2 == null) {
	course2 = tabbedPaneBase.getValueFromColumnCodeMap("COURSE_2");
      }
      if (course2 == null) {
	ret.reject();
      } else if (course2.equals("09")) {
	ret.accept();
      } else {
	ret.reject();
      }
    } else {					 
      String course2 = commonInfo.getValueFromCommonCodeMap("MY_COURSE_2_D");
      if (course2 == null) {
	course2 = tabbedPaneBase.getValueFromColumnCodeMap("COURSE_2");
      }
      if (course2 == null) {
	ret.reject();
      } else if (course2.equals("09")) {
	ret.accept();
      } else {
	ret.reject();
      }
    }
  }

  public void checkJabeeCourse(RetBoolean ret) { 
    if (!commonInfo.structControlDebugMode) {
      String currYear = commonInfo.getValueFromCommonCodeMap("MY_CURRICULUM_YEAR");
      if (currYear == null) {
	currYear = tabbedPaneBase.getValueFromColumnCodeMap("CURRICULUM_YEAR");
      }
      if (currYear == null) {
	ret.reject();
      } else {
	int cyear = Integer.parseInt(currYear);
	if ((cyear >= 2002) && (cyear <= 2003)) {
	  ret.accept();
	} else {
	  ret.reject();
	}
      }
    } else {
      String currYear = commonInfo.getValueFromCommonCodeMap("MY_CURRICULUM_YEAR_D");
      if (currYear == null) {
	currYear = tabbedPaneBase.getValueFromColumnCodeMap("CURRICULUM_YEAR");
      }
      if (currYear == null) {
	ret.reject();
      } else {
	int cyear = Integer.parseInt(currYear);
	if ((cyear >= 2002) && (cyear <= 2003)) {
	  ret.accept();
	} else {
	  ret.reject();
	}
      }
    }
  }

  public boolean checkSomuStaff() {
    if (!commonInfo.structControlDebugMode) {
      if (commonInfo.STAFF_QUALIFICATION.equals("7")) {
	return true;
      } else {
	return false;
      }
    } else {
      if (commonInfo.STAFF_QUALIFICATION_D.equals("7")) {
	return true;
      } else {
	return false;
      }
    } 
  }

  public boolean checkAdministrator() {
    if (!commonInfo.structControlDebugMode) {
      if (commonInfo.STAFF_QUALIFICATION.equals("8")) {
	return true;
      } else {
	return false;
      }
    } else {
      if (commonInfo.STAFF_QUALIFICATION_D.equals("8")) {
	return true;
      } else {
	return false;
      }
    }
  }

  public boolean checkGakumuStaff() {
	//  return true;
    if (!commonInfo.structControlDebugMode) {
      if (commonInfo.STAFF_QUALIFICATION.equals("8")) {
	return true;
      } else if (commonInfo.STAFF_QUALIFICATION.equals("9")) {
	return true;
      } else {
	return false;
      }
    } else {
      if (commonInfo.STAFF_QUALIFICATION_D.equals("8")) {
	return true;
      } else if (commonInfo.STAFF_QUALIFICATION_D.equals("9")) {
	return true;
      } else {
	return false;
      }
    }
    
  }

  
  public boolean checkHigherThanTeacher() {
    int qual;
    if (!commonInfo.structControlDebugMode) {
      try {
	qual = Integer.parseInt(commonInfo.STAFF_QUALIFICATION);
      } catch (Exception e) {
	qual = 0;
      }
      if ((qual >= 2) && (qual != 6) && (qual != 7)) {
	return true;
      } else {
	return false;
      }
    } else {
      try {
	qual = Integer.parseInt(commonInfo.STAFF_QUALIFICATION_D);
      } catch (Exception e) {
	qual = 0;
      }
      if ((qual >= 2) && (qual != 6) && (qual != 7)) {
	return true;
      } else {
	return false;
      }
    }
  }

  public boolean checkHigherThanEduCommittee() {
    int qual;
    if (!commonInfo.structControlDebugMode) {
      try {
	qual = Integer.parseInt(commonInfo.STAFF_QUALIFICATION);
      } catch (Exception e) {
	qual = 0;
      }
      if ((qual >= 4) && (qual != 6) && (qual != 7)) {
	return true;
      } else {
	return false;
      }
    } else {
      try {
	qual = Integer.parseInt(commonInfo.STAFF_QUALIFICATION_D);
      } catch (Exception e) {
	qual = 0;
      }
      if ((qual >= 4) && (qual != 6) && (qual != 7)) {
	return true;
      } else {
	return false;
      }
    }
  }

  public boolean checkHigherThanEduAssistant(String faculty,
					     String department) {
    int qual;
    if (!commonInfo.structControlDebugMode) {
      try {
	qual = Integer.parseInt(commonInfo.STAFF_QUALIFICATION);
      } catch (Exception e) {
	qual = 0;
      }
      if ((qual >= 4) && (qual != 6) && (qual != 7)) {
	return true;
      } else if (qual == 3) {  
	if ((faculty == null) || (department == null)) {
	  return false;
	}
	if (!faculty.equals("11")) {
	  return true;
	} else if (department.equals(commonInfo.STAFF_DEPARTMENT)) {
	  return true;
	} else {
	  return false;
	}
      } else {
	return false;
      }
    } else {
      try {
	qual = Integer.parseInt(commonInfo.STAFF_QUALIFICATION_D);
      } catch (Exception e) {
	qual = 0;
      }
      if ((qual >= 4) && (qual != 6) && (qual != 7)) {
	return true;
      } else if (qual == 3) {  
	if ((faculty == null) || (department == null)) {
	  return false;
	}
	if (!faculty.equals("11")) {
	  return true;
	} else if (department.equals(commonInfo.STAFF_DEPARTMENT_D)) {
	  return true;
	} else {
	  return false;
	}
      } else {
	return false;
      }
    }
  }

  public boolean checkQualifiedForNintei(String faculty,
					 String department,
					 String curriculumYear) {
    if (!checkHigherThanEduAssistant(faculty, department)) {
      return false;
    }    
    if ((commonInfo.thisSemester == 1) && (commonInfo.thisMonth <= 6)) {
      return true;
    } else {
      return false;
    }
  }

  public boolean checkHigherThanTeacherHimself(String faculty,
					       String department,
					       String teacherCode) {
    if (!commonInfo.structControlDebugMode) {
      if (checkHigherThanEduCommittee()) {
	return true;
      } else if (teacherCode == null) {
	return false;
      } else if (teacherCode.equals(commonInfo.STAFF_CODE)) {
	return true;
      } else if (checkHigherThanEduAssistant(faculty, department)) {
	return true;
      } else {
	return false;
      }
    } else {
      if (checkHigherThanEduCommittee()) {
	return true;
      } else if (teacherCode == null) {
	return false;
      } else if (teacherCode.equals(commonInfo.STAFF_CODE_D)) {
	return true;
      } else if (checkHigherThanEduAssistant(faculty, department)) {
	return true;
      } else {
	return false;
      }
    }
  }



  public String getSyllabusXml(String schoolYear, String subjectCode, String teacherCode) {
    if ((schoolYear == null) || (subjectCode == null) || (teacherCode == null)) {
      commonInfo.showMessageLong("開講義年度 = " + schoolYear + "$科目コード = " + subjectCode + "$教員コード = " + teacherCode  + "$ が正しく設定されていません。");
      return null;
    }
    String paramValues = schoolYear + "|" + subjectCode + "|" + teacherCode;  
    String text = serverMethods.getSyllabusXml(paramValues);
    if ((text == null) || text.trim().equals("")) {
      return makeNullSyllabusXml(subjectCode, teacherCode);
    } else {
      return text;
    }
  }

  public int updateSyllabusXml(String schoolYear, String subjectCode, String teacherCode,
			       String xmlText) {
    if ((schoolYear == null) || (subjectCode == null) || (teacherCode == null)) {
      commonInfo.showMessageLong("開講義年度 = " + schoolYear + "$科目コード = " + subjectCode + "$教員コード = " + teacherCode  + "$ が正しく設定されていません。");
      return 0;
    }
    String paramValues = schoolYear + "|" + subjectCode + "|" + teacherCode;  
    int n = serverMethods.updateSyllabusXml(paramValues, xmlText);
    return n;
  }

  public String makeNullSyllabusXml(String subjectCode, String teacherCode) { 
    String today = "" + commonInfo.thisYear + "-" + commonInfo.thisMonth + "-" + commonInfo.thisDay;
    String xmltext = "<?xml version=\"1.0\" encoding=\"EUC-JP\"?>\n" +
      "<講義 科目コード=\"" + subjectCode + "\" 教官コード=\"" + teacherCode + "\" 変更日時=\"" + today + "\" 変更者=\"" + commonInfo.STAFF_NAME + "\">\n" +
        "\t<授業の概要>\n" +
	  "\t\t<P>この授業の概要を記載する。</P>\n" +
        "\t</授業の概要>\n" +
        "\t<カリキュラムにおけるこの授業の位置付け>\n" +
          "\t\t<P>カリキュラムにおけるこの科目の位置、この科目の前提科目、この科目を前提とする後継科目などを記載する。</P>\n" +
	"\t</カリキュラムにおけるこの授業の位置付け>\n" +
	"\t<授業項目>\n" +
	  "\t\t<OL>\n" +
	    "\t\t\t<LI> 授業進行の流れ（授業計画）に沿って14項目の授業項目を記載する。</LI>\n" +
	    "\t\t\t<LI> 期末試験を実施する場合は15項目に「期末試験」を記載する。</LI>\n" +
	  "\t\t</OL>\n" +		
	"\t</授業項目>\n" +
	"\t<授業の進め方>\n" +
	  "\t\t<P>授業の形態、試験や演習の実施などに関する事項を記載する。</P>\n" +
	"\t</授業の進め方>\n" +
	"\t<授業の達成目標>\n" +
	  "\t\t<P>この科目は対象学科が掲げる「学習・教育目標」の目標（A）〜（Z）の内のどの目標をカバーしているか。</P>\n" +
	  "\t\t<P>この科目は具体的にはどの様な教育目標を達成しようとしているか。</P>\n" +
	"\t</授業の達成目標>\n" +
	"\t<成績評価の基準および評価方法>\n" + 
	  "\t\t<P>前項の「達成目標」の達成度をどの様な方法で（試験やレポート等）また各々をどの様な比重で評価するかを記載する。</P>\n" +
	"\t</成績評価の基準および評価方法>\n" +
	"\t<キーワード>\n" + 
	  "\t\t<P>キーワードは必ず記載して下さい。</P>\n" +
	"\t</キーワード>\n" +
	"\t<教科書></教科書>\n" +
	"\t<参考書></参考書>\n" +
	"\t<備考></備考>\n" +
      "</講義>\n";
    return xmltext;
  }


  public String getStudentAttrib(String studentCode) { 
    String key = "QUERY|COMMON_QUERY|queryStudentAttrib";
    return serverConn.queryCommon(key, studentCode);
  }


  public String getStaffAttrib(String staffCode) { 
    String key = "QUERY|COMMON_QUERY|queryStaffAttrib";
    return serverConn.queryCommon(key, staffCode);
  }

  public ImageIcon getStudentPhoto(String studentCode) { 
    String key = "PHOTO|STUDENT|SEND";
    return serverMethods.getStudentPhoto(key, studentCode);
  }

  public String getStudentGakuseki(String studentCode) { 
    String key = "GAKUSEKI|STUDENT|SEND";
    return serverMethods.getStudentGakuseki(key, studentCode);
  }

  public String getStaffName(String staffCode) { 
    String key = "STAFFNAME|STAFF|SEND";
    return serverMethods.getStaffName(key, staffCode);
  }

  //*** show Popup graphs ***//

  public void showSubjectSeisekiGraph(int marks, 
				      String SUBJECT_CODE, 
				      String TEACHER_CODE, String SCHOOL_YEAR,
				      String SUBJECT_NAME, String TEACHER_NAME) {
    if (gp5 == null) {
      gp5 = new GraphPanel5();
    }
    String key = "QUERY|COMMON_QUERY|querySubjectSeisekiGraph";
    String paramValues = SCHOOL_YEAR + "|" + SUBJECT_CODE + "|" + TEACHER_CODE;
    String ans = serverConn.queryCommon(key, paramValues);

    gp5.setGraph(4, ans, marks,
		 "  授業科目名:  " + SUBJECT_NAME,
		 "    担当教員:  " + TEACHER_NAME,
		 "履修者数 (" + SCHOOL_YEAR + "年度 履修者):  ");
    JOptionPane.showMessageDialog(commonInfo.getFrame(), gp5, 
				  "得点分布図", 
				  JOptionPane.INFORMATION_MESSAGE);   
  }
  
  public void showYokenGPAGraph(double gpa, 
				 String YOKEN_CODE, String FACULTY,
				 String DEPARTMENT, String COURSE, String GAKUNEN,
				 String YOKEN_CODE_NAME ) {
				 
    if (gp6 == null) {
      gp6 = new GraphPanel6();
    }
    int gpaIndex = (int)(gpa * 5.0);
    String key = "QUERY|COMMON_QUERY|queryYokenGPAGraph";
    String paramValues = YOKEN_CODE+"|"+FACULTY+"|"+DEPARTMENT+"|"+COURSE+"|"+GAKUNEN;
    String ans = serverConn.queryCommon(key, paramValues);

    gp6.setGraph(2, ans, gpaIndex,
		 " " + YOKEN_CODE_NAME + " のGPA値の分布",
		 "   GPA評価値:  " + gpa,
		 " 対象となる同級生数: " );
    JOptionPane.showMessageDialog(commonInfo.getFrame(), gp6, 
				  "GAP評価値分布図", 
				  JOptionPane.INFORMATION_MESSAGE); 
  }

  public void showTotalGPAGraph(double gpa, 
				  String FACULTY, String DEPARTMENT, 
				  String COURSE, String GAKUNEN ) {	
    if (gp6 == null) {
      gp6 = new GraphPanel6();
    }
    int gpaIndex = (int)(gpa * 5.0);
    String key = "QUERY|COMMON_QUERY|queryTotalGPAGraph";
    String paramValues = FACULTY+"|"+DEPARTMENT+"|"+COURSE+"|"+GAKUNEN;
    String ans = serverConn.queryCommon(key, paramValues);

    gp6.setGraph(2, ans, gpaIndex,
		 " 全履修登録科目のGPA値の分布",
		 "   GPA評価値:   " + gpa,
		 " 対象となる同級生数: " );
    JOptionPane.showMessageDialog(commonInfo.getFrame(), gp6, 
				  "GAP評価値分布図", 
				  JOptionPane.INFORMATION_MESSAGE); 
  }

  public void showGokakuGPAGraph(double gpa, 
				  String FACULTY, String DEPARTMENT, 
				  String COURSE, String GAKUNEN ) {	
    if (gp6 == null) {
      gp6 = new GraphPanel6();
    }
    int gpaIndex = (int)(gpa * 5.0);
    String key = "QUERY|COMMON_QUERY|queryGokakuGPAGraph";
    String paramValues = FACULTY+"|"+DEPARTMENT+"|"+COURSE+"|"+GAKUNEN;
    String ans = serverConn.queryCommon(key, paramValues);

    gp6.setGraph(2, ans, gpaIndex,
		 " 合格科目のみのGPA値の分布",
		 "   GPA評価値:   " + gpa,
		 " 対象となる同級生数: " );
    JOptionPane.showMessageDialog(commonInfo.getFrame(), gp6, 
				  "GAP評価値分布図", 
				  JOptionPane.INFORMATION_MESSAGE); 
  }

  //*** send Mail ***//
//  private SMTPSender smtp = null;
/*  
  public void sendMail(int cnt, String names, String addresses) {
    if (smtp == null) {
      smtp = new SMTPSender(commonInfo);
    }

    String toDay = "" + commonInfo.thisYear + "年 " + commonInfo.thisMonth + "月 " + commonInfo.thisDay + "日";
    String mailBody;
    boolean bccFlag;
    if (cnt <= 4) {
      bccFlag = false;
    } else if (cnt <= 9) {
      bccFlag = smtp.showMailTypeDialog(cnt);
    } else {
      bccFlag = true;
    }
    mailBody = smtp.showMailDialog(bccFlag, names, addresses, 
				   commonInfo.STAFF_NAME, commonInfo.MAIL_ADDRESS, commonInfo.LOCAL_ATTRIB, toDay);
    if ((bccFlag) && (mailBody != null)) {
      boolean ans = smtp.showConfirmMailDialog();
      if (ans) {
	smtp.emitConfirmMail(commonInfo.STAFF_NAME, commonInfo.MAIL_ADDRESS, mailBody, names, addresses);
      }
    }
  }

  public void changeSMTPSite() {
    if (smtp == null) {
      smtp = new SMTPSender(commonInfo);
    }

    smtp.changeSMTPHostDialog();
  }
*/

  //*** Nintei Report ***//

  public int reportNinteiData(String faculty, String department, 
			      String key, ArrayList<String> list) {
    if (checkHigherThanEduAssistant(faculty, department)) {
      int size = list.size();
      int cnt = serverMethods.reportNinteiData(key, list);
      commonInfo.showMessageLong("入力された "+size+" 科目の認定データのうち$ "+cnt+" 科目分がデータベースにセーブされました。"); 
      return cnt;   
    } else {
      commonInfo.showMessage("あなたには「単位認定」を設定する権限がありません。");
      return 0;
    } 
  }


  //*** Registration ***//

  public int cancelRegistration(String schoolYear, String subjectCode, String classCode) {
    String paramValues = commonInfo.STUDENT_CODE+"|"+schoolYear+"|"+subjectCode+"|"+classCode;
    int cnt = serverMethods.cancelRegistration(paramValues);
    return cnt;
  }
    
  public int registrSubject(String schoolYear, String subjectCode, String classCode,
			    String kubunCode, String reqCode, String unit,
			    String remark, String yomikaeSubject) {
    if (remark == null) remark = " ";
    if (yomikaeSubject == null) yomikaeSubject = " ";
    String paramValues = commonInfo.STUDENT_CODE+"|"+schoolYear+"|"+subjectCode+"|"+classCode+"|"+kubunCode+"|"+reqCode+"|"+unit+"|"+remark+"|"+yomikaeSubject;
    int cnt = serverMethods.registrSubject(paramValues);
    return cnt;
  }

  public boolean isFreshman() { 
    String THIS_SCHOOL_YEAR = "" + commonInfo.thisSchoolYear;
    if (commonInfo.STUDENT_CURRICULUM_YEAR.equals(THIS_SCHOOL_YEAR)) {
      return true;
    } else {
      return false;
    }
  }

  //*** reset Attend Control Params ***//
    
  public int resetAttendControlParamsOfAllSubjects(String schoolYear, String semester) {
    String paramValues = schoolYear+"|"+semester;
    int cnt = serverMethods.resetAttendControlParamsOfAllSubjects(paramValues);
    return cnt;   
  } 

  public int resetAttendControlParams(String schoolYear, String semester, 
				      String subjectCode, String classCode) {
    String paramValues = schoolYear+"|"+semester+"|"+subjectCode+"|"+classCode;
    int cnt = serverMethods.resetAttendControlParams(paramValues);
    return cnt;   
  } 


  //*** update Calendar data ***//

  public int setClassWeek(String year, String month, String day, String week) {
    String paramValues = year+"|"+month+"|"+day+"|"+week;
    int cnt = serverMethods.setClassWeek(paramValues);
    return cnt;   
  }

  public int deleteClassWeek(String year, String month, String day) {
    String paramValues = year+"|"+month+"|"+day;
    int cnt = serverMethods.deleteClassWeek(paramValues);
    return cnt;   
  }

  public int addHolidayInfo(String year, String month, String day, String holidayInfo) {
    String paramValues = year+"|"+month+"|"+day+"|"+holidayInfo;
    int cnt = serverMethods.addHolidayInfo(paramValues);
    return cnt;   
  }

  public int deleteHolidayInfo(String year, String month, String day) {
    String paramValues = year+"|"+month+"|"+day;
    int cnt = serverMethods.deleteHolidayInfo(paramValues);
    return cnt;   
  }

  public int addSchoolEvent(String year, String month, String day, String schoolEvent) {
    String paramValues = year+"|"+month+"|"+day+"|"+schoolEvent;
    int cnt = serverMethods.addSchoolEvent(paramValues);
    return cnt;   
  }

  public int deleteSchoolEvent(String year, String month, String day) {
    String paramValues = year+"|"+month+"|"+day;
    int cnt = serverMethods.deleteSchoolEvent(paramValues);
    return cnt;   
  }


  //*** update Gakuseki data ***//

  public int updateGakusekiData(String studentCode, String columnCode, String value) {
    String paramValues = studentCode + "|" + columnCode + "|" + value;
    int cnt = serverMethods.updateGakusekiData(paramValues);
    return cnt;
  }
    
  //*** init staff password  ***//

  public String initializeStaffPassword(String userID) {
    return serverConn.initializeStaffPassword(userID);
  }

  public String registrateStaffPassword(String param) { 
    return serverConn.registrateStaffPassword(param);
  }


  //*** Nintei To Seiseki ***//

  public int ninteiToSeiseki(String param) {  
    return serverMethods.ninteiToSeiseki(param);
  }

  //*** Copy Curriculum, Jikanari etc. of ThisSchoolYear to NextSchoolYear ***//

  public void copyCurriculumToNextYear(int fromYear, int toYear) {
    if (checkGakumuStaff()) {      	
      serverMethods.copyCurriculumToNextYear(fromYear, toYear);
    }    
  }

  public void copyEduCurriculumToNextYear(int fromYear, int toYear) {
    if (checkGakumuStaff()) {      	
      serverMethods.copyEduCurriculumToNextYear(fromYear, toYear);
    }    
  }

  public void copyClassInfoToNextYear(int fromYear, int toYear) {
    if (checkGakumuStaff()) {      	
      serverMethods.copyClassInfoToNextYear(fromYear, toYear);
    }    
  }

  public void copyJikanwariToNextYear(int fromYear, int toYear) {
    if (checkGakumuStaff()) {      	
      serverMethods.copyJikanwariToNextYear(fromYear, toYear);
    }    
  }

  public void copyJikanwariOverlapToNextYear(int fromYear, int toYear) {
    if (checkGakumuStaff()) {      	
      serverMethods.copyJikanwariOverlapToNextYear(fromYear, toYear);
    }    
  }

  public void copyGradYokenNextYear(int fromYear, int toYear) {
    if (checkGakumuStaff()) {      	
      serverMethods.copyGradYokenNextYear(fromYear, toYear);
    }    
  }

  public void copyEdYokenToNextYear(int fromYear, int toYear) {
    if (checkGakumuStaff()) {      	
      serverMethods.copyEdYokenToNextYear(fromYear, toYear);
    }    
  }

  public void copyModuleDefToNextYear(int fromYear, int toYear) {
    if (checkGakumuStaff()) {      	
      serverMethods.copyModuleDefToNextYear(fromYear, toYear);
    }    
  }

  public void copyIIFCurriculumToNextYear(int fromYear, int toYear) {
    if (checkGakumuStaff()) {      	
      serverMethods.copyIIFCurriculumToNextYear(fromYear, toYear);
    }    
  }

  public void copyYomikaeToNextYear(int fromYear, int toYear) {
    if (checkGakumuStaff()) {      	
      serverMethods.copyYomikaeToNextYear(fromYear, toYear);
    }    
  }

  public void copySyllabusToNextYear(int fromYear, int toYear) {
    if (checkGakumuStaff()) {      	
      serverMethods.copySyllabusToNextYear(fromYear, toYear);
    }    
  }

  public void copyClassTimeZoneToNextYear(int fromYear, int toYear) {
    if (checkGakumuStaff()) {      	
      serverMethods.copyClassTimeZoneToNextYear(fromYear, toYear);
    }    
  }

  //*** Report To Seiseki ***//

  public void rtorokuToSeiseki(String text, String subjectName, String teacherName) {
    if (checkGakumuStaff()) {
      int cnt = serverMethods.rtorokuToSeiseki(text);
      if (cnt >= 1) {
	String message = subjectName + " (" + teacherName + ") の成績報告データを $ 「成績表」に記入し、また、報告データに  $ 「処理済み」のマークを付けました。";
	commonInfo.showMessageLong(message);
      }
    } else {
      commonInfo.showMessage("あなたには成績報告データを成績表に記入する権限が与えられていません。");
    }      
  }

  public void saishiToSeiseki(String text, String subjectName, String teacherName) {
    if (checkGakumuStaff()) {
      int cnt = serverMethods.saishiToSeiseki(text);
      if (cnt >= 1) {
	String message = subjectName + " (" + teacherName + ") の再試成績報告を $ 「成績表」に記入し、また、報告データに  $ 「処理済み」のマークを付けました。";
	commonInfo.showMessageLong(message);
      }
    } else {
      commonInfo.showMessage("あなたには再試の成績報告を成績表に記入する権限が与えられていません。");
    }      
  }

  public void sendTransferNotice(String teacherCode, String message) {  ////
    String attr = getStaffAttrib(teacherCode);
    StringTokenizer sstk = new StringTokenizer(attr, "|"); 
    String teacherName = sstk.nextToken();
    String address = sstk.nextToken();	
    String teacherAddress = address.trim();
    if (teacherAddress.length() == 0) {
      teacherAddress = null;
    } 

    String text = "「成績報告の処理」に関するメールを " + teacherName + " 教員宛に $ 送信しますか？";
    boolean ret = commonInfo.showNoticeLong(text);
    if (ret) {    
      if (teacherAddress != null) {
	StringTokenizer stk = new StringTokenizer(message, "$");
	StringBuffer sbuf = new StringBuffer();
	while (stk.hasMoreTokens()) {
	  sbuf.append(stk.nextToken() + commonInfo.lineSeparator);
	}
	/*
	if (smtp == null) {
	  smtp = new SMTPSender(commonInfo);
	}
	String msg = smtp.sendSimpleMail(teacherAddress, 
					 commonInfo.gakumuAddress, 
					 "Notice on SEISEKI REPORT (from GAKUMU)", 
					 sbuf.toString());
					 */
      } else {
	text = " " + teacherName + " 教員のメールアドレスは登録されていないため、 $ 「成績報告の処理」に関するメールを発信することはできません。";
	commonInfo.showMessageLong(text);
      }
    }
  }

  //*** SyllabusTool ***//

  public int setTeacherEnglishName(String param) {
    // param = SUBJECT_CODE|TEACHER_CODE|TEACHER_NAME_ENGLISH
    return serverMethods.setTeacherEnglishName(param);
  }
    
  public int setSubjectEnglishName(String param) {
    // param = SUBJECT_CODE|TEACHER_CODE|SUBJECT_NAME_ENGLISH
    return serverMethods.setSubjectEnglishName(param);
  }
    
    
  //*** Jikawnari Control ***//

  public int addSubjectClassToJikanwari(String param) {
    // param = SCHOOL_YEAR|FACULTY|SEMESTER|GAKUNEN|WEEK|DEPARTMENT|HOUR|SUBJECT_CODE|CLASS_CODE|  
    if (checkGakumuStaff()) {
      return serverMethods.addSubjectClassToJikanwari(param);
    } else {
      commonInfo.showMessage("あなたには時間割表を編集する権限が与えられていません。");
      return 0;
    }      
  }

  public int deleteSubjectClassFromJikanwari(String param) {
    // param = SCHOOL_YEAR|FACULTY|SEMESTER|GAKUNEN|WEEK|SUBJECT_CODE|CLASS_CODE|DEPARTMENT|HOUR|
    if (checkGakumuStaff()) {
      return serverMethods.deleteSubjectClassFromJikanwari(param);
    } else {
      commonInfo.showMessage("あなたには時間割表を編集する権限が与えられていません。");
      return 0;
    }      
  }

  public int deleteSubjectClassFromJikanwari2(String param) {
    // param = SCHOOL_YEAR|FACULTY|SEMESTER|GAKUNEN|WEEK|SUBJECT_CODE|CLASS_CODE|DEPARTMENT|HOUR|
    if (checkGakumuStaff()) {
      return serverMethods.deleteSubjectClassFromJikanwari2(param);
    } else {
      commonInfo.showMessage("あなたには時間割表を編集する権限が与えられていません。");
      return 0;
    }      
  }

  public int setRoomToSubjectClassOfJikanwari(String param) {
    // param = SCHOOL_YEAR|FACULTY|SEMESTER|GAKUNEN|WEEK|SUBJECT_CODE|CLASS_CODE|DEPARTMENT|HOUR|ROOM|
    if (checkGakumuStaff()) {
      return serverMethods.setRoomToSubjectClassOfJikanwari(param);
    } else {
      commonInfo.showMessage("あなたには時間割表を編集する権限が与えられていません。");
      return 0;
    }      
  }

  public int setRoomToSubjectClassOfJikanwari2(String param) {
    // param = SCHOOL_YEAR|FACULTY|SEMESTER|GAKUNEN|WEEK|SUBJECT_CODE|CLASS_CODE|DEPARTMENT|HOUR|ROOM|
    if (checkGakumuStaff()) {
      return serverMethods.setRoomToSubjectClassOfJikanwari2(param);
    } else {
      commonInfo.showMessage("あなたには時間割表を編集する権限が与えられていません。");
      return 0;
    }      
  }
  
  public int addSubjectClassToShuchuList(String param) {
    // param = SCHOOL_YEAR|FACULTY|SEMESTER|DEPARTMENT|GAKUNEN|SUBJECT_CODE|CLASS_CODE|
    if (checkGakumuStaff()) {
      return serverMethods.addSubjectClassToShuchuList(param);
    } else {
      commonInfo.showMessage("あなたには時間割表を編集する権限が与えられていません。");
      return 0;
    }      
  }

  public int addSubjectClassToGraduateJikanwari(String param) {
    // param = SCHOOL_YEAR|FACULTY|SEMESTER|GAKUNEN|WEEK|DEPARTMENT|HOUR|SUBJECT_CODE|CLASS_CODE|
    if (checkGakumuStaff()) {
      return serverMethods.addSubjectClassToGraduateJikanwari(param);
    } else {
      commonInfo.showMessage("あなたには時間割表を編集する権限が与えられていません。");
      return 0;
    }      
  }

  public int addSubjectClassToGraduateShuchuList(String param) {
    // param = SCHOOL_YEAR|FACULTY|SEMESTER|DEPARTMENT|GAKUNEN|SUBJECT_CODE|CLASS_CODE|
    if (checkGakumuStaff()) {
      return serverMethods.addSubjectClassToGraduateShuchuList(param);
    } else {
      commonInfo.showMessage("あなたには時間割表を編集する権限が与えられていません。");
      return 0;
    }      
  } 

  public int deleteSubjectClassFromShuchuList(String param) {  
    // param = SCHOOL_YEAR|FACULTY|SEMESTER|DEPARTMENT|GAKUNEN|SUBJECT_CODE|CLASS_CODE|
    if (checkGakumuStaff()) {
      return serverMethods.deleteSubjectClassFromShuchuList(param);
    } else {
      commonInfo.showMessage("あなたには時間割表を編集する権限が与えられていません。");
      return 0;
    }      
  }  

  public int deleteSubjectClassFromShuchuList2(String param) {  
    // param = SCHOOL_YEAR|FACULTY|SEMESTER|DEPARTMENT|GAKUNEN|SUBJECT_CODE|CLASS_CODE|
    if (checkGakumuStaff()) {
      return serverMethods.deleteSubjectClassFromShuchuList2(param);
    } else {
      commonInfo.showMessage("あなたには時間割表を編集する権限が与えられていません。");
      return 0;
    }      
  }  

    
  //*** Seiseki Report ***//

  public int saveKimatsuData(String key, String teacherCode, ArrayList<String> list) {
    if ((checkGakumuStaff()) || (teacherCode.equals(commonInfo.STAFF_CODE))) {
      int size = list.size();
      int cnt = serverMethods.saveKimatsuData(key, list);
      commonInfo.showMessageLong("入力された "+size+" 人分の期末成績データのうち$ "+cnt+" 人分のデータがセーブ (仮報告) されました。");
      return cnt;
    } else {
      commonInfo.showMessage("あなたには成績データを仮報告する権限が与えられていません。");
      return 0;
    }      
  }

  public int reportKimatsuData(String key, String teacherCode, ArrayList<String> list) {
    if ((checkGakumuStaff()) || (teacherCode.equals(commonInfo.STAFF_CODE))) {
      int size = list.size();
      int cnt = serverMethods.reportKimatsuData(key, list);
      commonInfo.showMessageLong("入力された "+size+" 人分の期末成績データのうち$ "+cnt+" 人分のデータが「成績報告」されました。"); 
      return cnt;   
    } else {
      commonInfo.showMessage("あなたには成績データを「成績報告」する権限が与えられていません。");
      return 0;
    } 
  }

  public int saveSaishiData(String key, String teacherCode, ArrayList<String> list) {
    if ((checkGakumuStaff()) || (teacherCode.equals(commonInfo.STAFF_CODE))) {
      int size = list.size();
      int cnt = serverMethods.saveSaishiData(key, list);
      commonInfo.showMessageLong("入力された "+size+" 人分の再試験の成績データのうち$ "+cnt+" 人分のデータがセーブ (仮報告) されました。");
      return cnt;   
    } else {
      commonInfo.showMessage("あなたには成績データを仮報告する権限が与えられていません。");
      return 0;
    } 
  }

  public int reportSaishiData(String key, String teacherCode, ArrayList<String> list) {
    if ((checkGakumuStaff()) || (teacherCode.equals(commonInfo.STAFF_CODE))) {
      int size = list.size();
      int cnt = serverMethods.reportSaishiData(key, list);  
      commonInfo.showMessageLong("入力された "+size+" 人分の再試験の成績データのうち$ "+cnt+" 人分のデータが「成績報告」されました。"); 
      return cnt;   
    } else {
      commonInfo.showMessage("あなたには成績データを「成績報告」する権限が与えられていません。");
      return 0;
    } 
  }

  public void emitKimatsuReportMail(String schoolYear, 
				    String subjectName, String classCode, 
				    String teacherName) {  
    String msg = "  情報工学部 学務係 御中： $ $ 次の授業科目の「期末試験」の成績報告を入力しましたので $ 成績報告データを「成績表」に記入して下さい。$ $  開講年度: " + schoolYear + " $  授業科目名： " + subjectName + "$     クラス： " + classCode + "$    担当教員： " + teacherName + "$ $        報告者：" + commonInfo.STAFF_NAME + "$      報告日時：" + commonInfo.thisYear + "年" + commonInfo.thisMonth + "月" + commonInfo.thisDay + "日" ; 
    if (commonInfo.MAIL_ADDRESS.length() == 0) {
      commonInfo.showMessageLong(" 次の内容のメールを「学務係」宛に発信したいのですが $ あなたのメールアドレスが不明であるため発信できません。 $ $ 成績報告を実行するには「学務係」に出向いて以下のメッセージを $ 伝える必要があります。$ $ "+msg );
    } else {
      StringTokenizer stk = new StringTokenizer(msg, "$");
      StringBuffer sbuf = new StringBuffer();
      while (stk.hasMoreTokens()) {
	sbuf.append(stk.nextToken()).append(commonInfo.lineSeparator);
      }
/*
      if (smtp == null) {
	smtp = new SMTPSender(commonInfo);
      }
      String ret = smtp.sendSimpleMail(commonInfo.gakumuAddress, 
				       commonInfo.MAIL_ADDRESS, 
				       "kimatsu seiseki hokoku", 
				       sbuf.toString());
      if (ret != null) {
	commonInfo.showMessageLong(" 次の内容のメールが「学務係」宛に送信されました：$ $ "+msg );
      }
      */ 
    }
  }

  public void emitSaishiReportMail(String schoolYear, 
				   String subjectName, String classCode, 
				   String teacherName) {
    String msg = "  情報工学部 学務係 御中： $ $ 次の授業科目の「再試験」の成績報告を入力しましたので $ 成績報告データを「成績表」に記入して下さい。$  $  開講年度: " + schoolYear + " $  再試験科目名： " + subjectName + "$     クラス： " + classCode + "$    担当教員： " + teacherName + "$ $        報告者：" + commonInfo.STAFF_NAME + "$      報告日時：" + commonInfo.thisYear + "年" + commonInfo.thisMonth + "月" + commonInfo.thisDay + "日" ;
    if (commonInfo.MAIL_ADDRESS.length() == 0) {
      commonInfo.showMessageLong(" 次の内容のメールを「学務係」宛に発信したいのですが $ あなたのメールアドレスが不明であるため発信できません。 $ $ 成績報告を実行するには「学務係」に出向いて以下のメッセージを $ 伝える必要があります。$ $ "+msg );
    } else {
      StringTokenizer stk = new StringTokenizer(msg, "$");
      StringBuffer sbuf = new StringBuffer();
      while (stk.hasMoreTokens()) {
	sbuf.append(stk.nextToken()).append(commonInfo.lineSeparator);
      }
/*
      if (smtp == null) {
	smtp = new SMTPSender(commonInfo);
      }
      String ret = smtp.sendSimpleMail(commonInfo.gakumuAddress, 
				       commonInfo.MAIL_ADDRESS, 
				       "saishi seiseki hokoku", 
				       sbuf.toString());
				       
      if (ret != null) {
	commonInfo.showMessageLong(" 次の内容のメールが「学務係」宛に送信されました：$ $ "+msg );
      } 
      */
    }
  }

  //***  StructControlTool のユーティリティ ***//

  public void makeBackupOfStructTable(String tableName) {
    serverConn.makeBackupOfStructTable(tableName);
  }

  public void readBackupOfStructTable(String tableName) {
    serverConn.readBackupOfStructTable(tableName);
  }
}
