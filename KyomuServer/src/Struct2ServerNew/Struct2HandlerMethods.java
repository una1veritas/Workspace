package Struct2ServerNew;
import common.*;
import java.net.*;
import java.io.*;
import java.util.*;
import java.awt.*;
import java.sql.*;
import syllabusNew.*;

public class Struct2HandlerMethods {
  private CommonInfo commonInfo;  
  private Socket socket;  
  private InputStream istream;
  private OutputStream ostream;    
  private BufferedReader cin;
  private PrintWriter cout;

  private JDBCConnection jdbc;
  private Person me;
  private String PASSWORD = "";
  private boolean qualified = false;
  private Person originalMe;
  private SyllabusControl syllabus;

  public Struct2HandlerMethods(CommonInfo commonInfo,
			       Socket socket,
			       InputStream istream,
			       OutputStream ostream,
			       BufferedReader cin,
			       PrintWriter cout) {
    this.commonInfo = commonInfo;
    this.socket = socket;
    this.istream = istream;
    this.ostream = ostream;
    this.cin = cin;
    this.cout = cout;
    jdbc = (JDBCConnection) commonInfo.jdbc;
    me = new Person();
    syllabus = commonInfo.syllabus;
  }
 
  public void sendPasswordCheckResult(String paramValues) {
    String userID;
    String rawPassword;
    String ans = "";
    String STAFF_CODE = "";
    String PASSWORD_STATUS = "";
    String QUALIFICATION = "";
    String inetAddress = socket.getInetAddress().getHostAddress();

    String[] tokens = paramValues.split("\\|");
    userID = tokens[0];
    rawPassword = tokens[1];
        
    try {
      String quest = "select STAFF_CODE, PASSWORD, PASSWORD_STATUS, QUALIFICATION from TEACHER.STAFF_PASSWD where STAFF_ID = '" + userID + "'";
      ans = jdbc.executeKyomuQuery(quest);
      
      if (ans == null) {
	cout.println("たぶん USER_ID が間違っています。");
	qualified = false;
	return;
      }	else {
	String[] tokens2 = ans.split("\\|");
	STAFF_CODE = tokens2[0].trim();
	PASSWORD = tokens2[1].trim(); 
	PASSWORD_STATUS = tokens2[2].trim();
	QUALIFICATION = tokens2[3].trim();
      } 
            
      if (PASSWORD_STATUS.equals("9")) { 
	cout.println("利用停止中: 学務係に申し出て利用停止を解除して下さい。");
	qualified = false;
	return;
      }

      if ((!QUALIFICATION.equals("8")) && (!QUALIFICATION.equals("9"))) {
	cout.println("利用不可:このツールを利用するには学務職員以上の権限が必要です。");
	qualified = false;
	return;
      }	      
      int cnt = getUserMapCount(userID);
      if (cnt > 4) {
	cout.println("パスワードの入力ミスが規定回数を超えたため本日中はログインできません。");
	qualified = false;
	return;
      } 
      String salt = PASSWORD.substring(0, 2);
      String cryptPassword = commonInfo.crypt(rawPassword, salt);
      if (cryptPassword.equals(PASSWORD)) {
	appendLog("1", userID, STAFF_CODE, QUALIFICATION, inetAddress);
	setUserMapCount(userID, 0);
	me.USER_ID = userID;
	me.STAFF_CODE = STAFF_CODE;
	me.QUALIFICATION = QUALIFICATION;
	me.inetAddress = inetAddress;

	String query = "select STAFF_TYPE, STAFF_OCCUPATION, STAFF_STATUS, STAFF_ATTRIB, LOCAL_ATTRIB, STAFF_NAME, MAIL_ADDRESS from MASTER.STAFF where STAFF_CODE = '" + me.STAFF_CODE + "'";
	ans = jdbc.executeKyomuQuery(query);
	if (ans != null) {
	  String[] tokens3 = ans.split("\\|");
	  me.STAFF_TYPE        = tokens3[0].trim();
	  me.STAFF_OCCUPATION  = tokens3[1].trim();
	  me.STAFF_STATUS      = tokens3[2].trim();
	  me.STAFF_ATTRIB      = tokens3[3].trim();
	  me.LOCAL_ATTRIB      = tokens3[4].trim();
	  me.STAFF_NAME        = tokens3[5].trim();
	  me.MAIL_ADDRESS      = tokens3[6].trim();
	} 
	commonInfo.addLoginMap(me);
	cout.println("success|" + STAFF_CODE); 
	qualified = true;
	return;
      } else {
	cnt++;
	setUserMapCount(userID, cnt);
	cout.println("パスワードが正しくありません。");
	appendLog("0", userID, STAFF_CODE, QUALIFICATION, inetAddress);
	qualified = false;
	return;
      }
    } catch (Exception e) {
      cout.println("ERROR: " + e.toString().trim());
      qualified = false;
      return;
    }
  }

  public void sendPasswordChangeResult(String paramValues) {	
    try {	
      if (!qualified) {
	cout.println("ERROR: ユーザ認証に合格していません。"); 
	return;
      }
      String[] tokens = paramValues.split("\\|");
      String userID = tokens[0];
      String oldPasswd = tokens[1];
      String newPasswd = tokens[2];      
	    
      String salt = PASSWORD.substring(0, 2);
      String cryptPassword = commonInfo.crypt(oldPasswd, salt);
      if ((cryptPassword.equals(PASSWORD)) && (userID.equals(me.USER_ID))) {
	salt = newPasswd.substring(0, 2);
	cryptPassword = commonInfo.crypt(newPasswd, salt);
	String update = "update TEACHER.STAFF_PASSWD set PASSWORD = '" + cryptPassword + "', PASSWORD_STATUS = '1', REVISED_DATE = sysdate where STAFF_ID = '" + userID + "'";
	int res = jdbc.executeKyomuUpdate(update); 
	if (res == 1) {
	  cout.println("success");
	} else {
	  cout.println("パスワードの変更に失敗しました。");
	}
      } else {
	cout.println("USER_ID または PASSWORD が間違っています。");
      }
    } catch (Exception e) {
      cout.println("ERROR： " + e.toString().trim());
    }
  } 

  public void setStaffUserForDebug(String userID) {
    try {
      originalMe = me;
      me = new Person();
      String quest = "select STAFF_CODE, QUALIFICATION from TEACHER.STAFF_PASSWD where STAFF_ID = '" + userID + "'";
      String ans = jdbc.executeKyomuQuery(quest);
      String[] tokens2 = ans.split("\\|");
      me.USER_ID = userID;
      me.STAFF_CODE = tokens2[0].trim();
      me.QUALIFICATION = tokens2[1].trim();
      String query = "select STAFF_TYPE, STAFF_OCCUPATION, STAFF_STATUS, STAFF_ATTRIB, LOCAL_ATTRIB, STAFF_NAME, MAIL_ADDRESS from MASTER.STAFF where STAFF_CODE = '" + me.STAFF_CODE + "'";
      ans = jdbc.executeKyomuQuery(query);
      String[] tokens3 = ans.split("\\|");
      me.STAFF_TYPE        = tokens3[0].trim();
      me.STAFF_OCCUPATION  = tokens3[1].trim();
      me.STAFF_STATUS      = tokens3[2].trim();
      me.STAFF_ATTRIB      = tokens3[3].trim();
      me.LOCAL_ATTRIB      = tokens3[4].trim();
      me.STAFF_NAME        = tokens3[5].trim();
      me.MAIL_ADDRESS      = tokens3[6].trim(); 
      cout.println("ok");
    } catch (Exception e) {
      cout.println("ERROR： " + e.toString().trim());
    }
  }

  public void setStudentUserForDebug(String studentCode) {
    try {      
      originalMe = me;
      me = new Person();
      me.QUALIFICATION = "0"; 
      me.USER_ID = studentCode;
      me.STUDENT_CODE = studentCode;
      String query = "select S.SHORTER_NAME, S.STUDENT_STATUS, S.FACULTY, S.DEPARTMENT, S.COURSE, S.COURSE_2, S.GAKUNEN, S.CURRICULUM_YEAR, S.SUPERVISOR, F.STAFF_NAME from MASTER.GAKUSEKI S, MASTER.STAFF F where S.SUPERVISOR = F.STAFF_CODE (+) and S.STUDENT_CODE = '" + studentCode + "'";
      String ans = jdbc.executeKyomuQuery(query);
      String[] tokens = ans.split("\\|");
      StringTokenizer stk = new StringTokenizer(ans, "|"); 
      me.STUDENT_NAME            = tokens[0].trim();
      me.STUDENT_STATUS          = tokens[1].trim();
      me.STUDENT_FACULTY         = tokens[2].trim();
      me.STUDENT_DEPARTMENT      = tokens[3].trim();
      me.STUDENT_COURSE          = tokens[4].trim();
      me.STUDENT_COURSE_2        = tokens[5].trim();
      me.STUDENT_GAKUNEN         = tokens[6].trim();
      me.STUDENT_CURRICULUM_YEAR = tokens[7].trim();
      me.STUDENT_SUPERVISOR      = tokens[8];
      me.STUDENT_SUPERVISOR_NAME = tokens[9]; 
      cout.println("ok");
    } catch (Exception e) {
      cout.println("ERROR： " + e.toString().trim());
    }
  }

  public void resetUserForDebug(String dummy) {
    try { 
      me = originalMe;
      cout.println("ok");
    } catch (Exception e) {
      cout.println("ERROR： " + e.toString().trim());
    }
  }

  public void sendCommonQueryResult(String commandCode, String paramValues) {
    try { 
      if (paramValues.equals("empty")) {
	paramValues = null;
      }
      if (commandCode.equals("queryCommonQueryResult")) {
	// paramValues = queryName + "#" + paramValues;
	String[] tokens = paramValues.split("\\#");
	String queryName = tokens[0];
	String paramValues2 = tokens[1];
	String result = jdbc.getCommonQueryResultByQueryName(queryName, paramValues2);
	if (result != null) {
	  cout.println(result);
	} else {
	  cout.println("null");
	}
      } else {
	String result = jdbc.getCommonQueryResult(commandCode, paramValues);
	if (result != null) {
	  cout.println(result);
	} else {
	  cout.println("null");
	}
      }
    } catch (Exception e) {	
      cout.println("ERROR$" + e.toString().trim()); 
    }
  }

  public void sendQueryResult(String serviceName, 
				 String commandCode,
				 String paramValues) {	   
    try { 
      if (qualified) {
	if (paramValues.equals("empty")) {
	  paramValues = null;
	} 
	// commandCode = panelID+"#"+switchCode;
	String[] tokens = commandCode.split("\\#");
	String panelID = tokens[0];
	String switchCode = tokens[1];
	String str = jdbc.getQueryResult(serviceName, panelID, switchCode, 
					 paramValues, me);
	if (str == null) {
	  cout.println("null");
	} else {
	  cout.println(str);
	} 
      } else {
	cout.println("ERROR: ユーザ認証に合格していません。"); 
      }
    } catch (Exception e) {	
      cout.println("ERROR$" + e.toString().trim()); 
    }
  }  

  public void sendSingleStructQueryResult(String serviceName, 
					     String commandCode,
					     String paramValues) {
    try { 
      if (qualified) {
	if (paramValues.equals("empty")) {
	  paramValues = null;
	} 
	// commandCode = panelID+"#"+switchCode;
	String[] tokens = commandCode.split("\\#");
	String panelID = tokens[0];
	String switchCode = tokens[1];
	String str = jdbc.getSingleStructQuery(serviceName, 
					       panelID, switchCode, 
					       paramValues);
	if (str == null) {
	  cout.println("null");
	} else {
	  cout.println(str);
	} 
      } else {
	cout.println("ERROR: ユーザ認証に合格していません。"); 
      }
    } catch (Exception e) {	
      cout.println("ERROR$" + e.toString().trim()); 
    }
  }  

  public void sendDeleteResult(String serviceName, 
				  String commandCode,
				  String paramValues) {	
    try {
      if (qualified) {
	if (paramValues.equals("empty")) {
	  paramValues = null;
	}
	int res = jdbc.deleteCommand(serviceName, commandCode, paramValues, me);
	cout.println("" + res);
      } else {
	cout.println("ERROR: ユーザ認証に合格していません。"); 
      }
    } catch (Exception e) {	
      cout.println("ERROR$" + e.toString().trim()); 
    }
  }

  public void sendUpdateResult(String serviceName, 
				  String commandCode,
				  String paramValues) {		
    try {
      if (qualified) {
	if (paramValues.equals("empty")) {
	  paramValues = null;
	}
	int res = jdbc.updateCommand(serviceName, commandCode, paramValues, me);
	cout.println("" + res);
      } else {
	cout.println("ERROR: ユーザ認証に合格していません。"); 
      }
    } catch (Exception e) {	
      cout.println("ERROR$" + e.toString().trim()); 
    }
  }
    
  public void sendInsertResult(String serviceName, 
				  String commandCode,
				  String paramValues) {		
    try {
      if (qualified) {
	if (paramValues.equals("empty")) {
	  paramValues = null;
	}
	int res = jdbc.insertCommand(serviceName, commandCode, paramValues, me);
	cout.println("" + res);
      } else {
	cout.println("ERROR: ユーザ認証に合格していません。"); 
      }
    } catch (Exception e) {	
      cout.println("ERROR$" + e.toString().trim()); 
    }
  }  

  public void sendQueryStaffAttribResult(String paramValues) {	
    try {	
      if (!qualified) {
	cout.println("ERROR: ユーザ認証に合格していません。"); 
	return;
      }
      String mailAddress = me.MAIL_ADDRESS;
      String localAttrib = me.LOCAL_ATTRIB;
      if ((mailAddress == null) || (mailAddress.equals(""))) {
	mailAddress = " ";
      }
      if ((localAttrib == null) || (localAttrib.equals(""))) {
	localAttrib = " ";
      }

      StringBuffer sbuf = new StringBuffer();
      sbuf.append(me.STAFF_CODE).append("|");
      sbuf.append(me.STAFF_NAME).append("|");
      sbuf.append(me.STAFF_ATTRIB).append("|");
      sbuf.append(me.QUALIFICATION).append("|");
      sbuf.append(mailAddress).append("|");
      sbuf.append(localAttrib).append("|");
      sbuf.append(me.STAFF_DEPARTMENT).append("|");
      cout.println(sbuf.toString());
    } catch (Exception e) {
      cout.println("ERROR： " + e.toString().trim());
    }
  }  
  
  public void sendQueryStudentAttribResult(String paramValues) {	
    try {	
      if (!qualified) {
	cout.println("ERROR: ユーザ認証に合格していません。"); 
	return;
      }
      StringBuffer sbuf = new StringBuffer();
      sbuf.append(me.STUDENT_CODE).append("|");
      sbuf.append(me.STUDENT_NAME).append("|");
      sbuf.append(me.STUDENT_STATUS).append("|");
      sbuf.append(me.STUDENT_FACULTY).append("|");
      sbuf.append(me.STUDENT_DEPARTMENT).append("|");
      sbuf.append(me.STUDENT_COURSE).append("|");
      sbuf.append(me.STUDENT_COURSE_2).append("|");
      sbuf.append(me.STUDENT_GAKUNEN).append("|");
      sbuf.append(me.STUDENT_CURRICULUM_YEAR).append("|");
      sbuf.append(me.STUDENT_SUPERVISOR).append("|");
      sbuf.append(me.STUDENT_SUPERVISOR_NAME).append("|");
      sbuf.append(me.inetAddress).append("|");
      cout.println(sbuf.toString());
    } catch (Exception e) {
      cout.println("ERROR： " + e.toString().trim());
    }
  }  	  

  public void sendSyllabusXml(String paramValues) {
    try {
      String[] tokens = paramValues.split("\\|");
      String schoolYear = tokens[0].trim();
      String subjectCode = tokens[1].trim();
      String teacherCode = tokens[2].trim();

      String str = syllabus.getElementXml(schoolYear, teacherCode, subjectCode); 
      if (str != null) {
	cout.println(str);
	cout.println(".");
      } else {
	cout.println("null");
	cout.println(".");
      }
    } catch (Exception e) {	
      cout.println("ERROR$" + e);
      cout.println(".");
    }
  }
  
  public void saveSyllabusXml(String paramValues) {	
    try {
      String[] tokens = paramValues.split("\\|");
      String schoolYear = tokens[0].trim();
      String subjectCode = tokens[1].trim();
      String teacherCode = tokens[2].trim();

      if (schoolYear.trim().equals("")) {
	cout.println("ERROR$ 開講年度が指定されていません。 "); 
	return;
      }
      if (subjectCode.trim().equals("")) {
	cout.println("ERROR$ 科目が指定されていません。 "); 
	return;
      }
      if (teacherCode.trim().equals("")) {
	cout.println("ERROR$ 担当教官が指定されていません。 "); 
	return;
      }   

      String line;
      StringBuffer sbuf = new StringBuffer();      
      while ((line = cin.readLine()) != null) {
	if (line.equals(".")) break;
	sbuf.append(line).append("\n");
      }
      String xmlText = sbuf.toString();   
      
      if (qualifiedToUpdateSyllabus(teacherCode, me)) {
	syllabus.updateElement(schoolYear, xmlText);
	cout.println("1"); 
      } else {
	cout.println("ERROR$ あなたは教授要目を変更する資格がありません。"); 
      }
    } catch (Exception e) {	
      cout.println("ERROR$" + e); 
    }
  }

  public void syllabusTool(String commandCode, String paramValues) {
    String update = "";
    if (commandCode.equals("SETTEACHERENGLISHNAME")) {
      try {
	String[] tokens = paramValues.split("\\|");
	String subjectCode = tokens[0].trim();
	String teacherCode = tokens[1].trim();
	String teacherEnglishName = tokens[2].trim();
	if (teacherEnglishName.equals("")) {
	  update = "update MASTER.STAFF set ENGLISH_NAME = null where STAFF_CODE = '" + teacherCode + "'";	  
	} else {
	  update = "update MASTER.STAFF set ENGLISH_NAME = '"+teacherEnglishName+"' where STAFF_CODE = '" + teacherCode + "'";	  
	}
	int res = jdbc.executeKyomuUpdate(update); 
	if (res <= 0) {
	  cout.println("0");
	} else {
	  cout.println("1");
	}		
	return; 
      } catch (Exception e) {
	cout.println("0");    
      }
    } else if (commandCode.equals("SETSUBJECTENGLISHNAME")) {
      try {
	String[] tokens = paramValues.split("\\|");
	String subjectCode = tokens[0].trim();
	String teacherCode = tokens[1].trim();
	String subjectEnglishName = tokens[2].trim();
	if (subjectEnglishName.equals("")) {
	  update = "update MASTER.SUBJECT set ENGLISH_NAME = null where SUBJECT_CODE = '" + subjectCode + "'";	  
	} else {
	  update = "update MASTER.SUBJECT set ENGLISH_NAME = '"+subjectEnglishName+"' where SUBJECT_CODE = '" + subjectCode + "'";	  
	}
	int res = jdbc.executeKyomuUpdate(update); 
	if (res <= 0) {
	  cout.println("0");
	} else {
	  cout.println("1");
	}		
	return; 
      } catch (Exception e) {
	cout.println("0");    
      }
    }
  }

  public void saveKimatsuData(String paramValues) { 
    HashMap<String, String> rtorokuMap = new HashMap<String, String>(); 
    try {
      String[] tokens2 = paramValues.split("\\|");
      String schoolYear  = tokens2[0].trim();
      String subjectCode = tokens2[1].trim();
      String classCode   = tokens2[2].trim();
      String teacherCode = tokens2[3].trim();
      String msg = "select STUDENT_CODE, FIXED from KYOMU.RTOROKU where SCHOOL_YEAR = '" + schoolYear + "' and SUBJECT_CODE = '" + subjectCode + "' and CLASS_CODE = '" + classCode + "'";
      String res = jdbc.executeKyomuQuery(msg);      
      String[] lines = res.split("\\$");
      for (String line : lines) {
	String[] tokens = line.split("\\|");	
	String scode = tokens[0];
	String fixed = tokens[1];
	rtorokuMap.put(scode, fixed);
      }
      if (qualifiedToSeisekiReport(teacherCode, me)) {
	int count = 0;
	String line;
	while ((line = cin.readLine()) != null) {
	  if (line.equals(".")) break;
	  String[] tokens = line.split("\\|");	
	  String studentCode = tokens[0];
	  String marks = tokens[1];
	  String fixed = tokens[2];
	  String fixed2 = rtorokuMap.get(studentCode);	  
	  if ((fixed2 == null) || (fixed2.equals("F")) || (fixed2.equals("Y"))) continue;

	  String upd = "update KYOMU.RTOROKU set MARKS = " + marks + ", HOKOKU_DATE = sysdate, FIXED = '" + fixed + "', HOKOKU_BY = '" + me.STAFF_CODE + "' where SCHOOL_YEAR = '" + schoolYear + "' and STUDENT_CODE = '" + studentCode + "' and SUBJECT_CODE = '" + subjectCode + "' and CLASS_CODE = '" + classCode + "'";
	  int n = jdbc.executeKyomuUpdate(upd);
	  count += n;
	}
	cout.println("" + count); 
      } else {
	cout.println("0"); 
      }
    } catch (Exception e) {
      cout.println("0");    
    }
  }

  public void saveSaishiData(String paramValues) { 
    HashMap<String, String> saishiMap = new HashMap<String, String>();

    try {
      String[] tokens2 = paramValues.split("\\|");
      String schoolYear  = tokens2[0].trim();
      String subjectCode = tokens2[1].trim();
      String classCode   = tokens2[2].trim();
      String teacherCode = tokens2[3].trim();
      String msg = "select STUDENT_CODE, FIXED from KYOMU.SAISHI where SCHOOL_YEAR = '" + schoolYear + "' and SUBJECT_CODE = '" + subjectCode + "' and CLASS_CODE = '" + classCode + "'";
      String res = jdbc.executeKyomuQuery(msg);   
      String[] lines = res.split("\\$");
      for (String line : lines) {
	String[] tokens = line.split("\\|");	
	String scode = tokens[0];
	String fixed = tokens[1];
	saishiMap.put(scode, fixed);
      }  
      if (qualifiedToSeisekiReport(teacherCode, me)) {
	int count = 0;
	String line;
	while ((line = cin.readLine()) != null) {
	  if (line.equals(".")) break;
	  String[] tokens = line.split("\\|");	
	  String studentCode = tokens[0];
	  String marks = tokens[1];
	  String fixed = tokens[2];
	  String fixed2 = saishiMap.get(studentCode);	
	  if ((fixed2 == null) || (fixed2.equals("F")) || (fixed2.equals("Y"))) continue;

	  String upd = "update KYOMU.SAISHI set MARKS = " + marks + ", HOKOKU_DATE = sysdate, FIXED = '" + fixed + "', HOKOKU_BY = '" + me.STAFF_CODE + "' where SCHOOL_YEAR = '" + schoolYear + "' and STUDENT_CODE = '" + studentCode + "' and SUBJECT_CODE = '" + subjectCode + "' and CLASS_CODE = '" + classCode + "'";
	  int n = jdbc.executeKyomuUpdate(upd);
	  count += n;
	}
	cout.println("" + count); 
      } else {
	cout.println("0"); 
      }
    } catch (Exception e) {
      cout.println("0");    
    }
  }

  public void reportNinteiData(String paramValues) { 
    String studentCode = paramValues;
    try {
      String line;
      int count = 0;
      while ((line = cin.readLine()) != null) {
	if (line.equals(".")) break;
	String[] tokens = line.split("\\|");	
	String subjectCode = tokens[0];
	String qualify     = tokens[1];
	String evidence    = tokens[2];

	String upd = "";
	if (qualify.equals("N")) {
	  upd = "update TEACHER.QUALIFY_INPUT set EVIDENCE = null, QUALIFIED_DATE = null, QUALIFIED_BY = null, QUALIFIED = 'N' where STUDENT_CODE = '" + studentCode + "' and SUBJECT_CODE = '" + subjectCode + "'";
	} else if (qualify.equals("Y")) {
	  int index = evidence.indexOf("\n");
	  if (index > 0) {
	    evidence = evidence.substring(0, index);
	  }
	  upd = "update TEACHER.QUALIFY_INPUT set EVIDENCE = '" + evidence + "', QUALIFIED_DATE = sysdate, QUALIFIED_BY = '" + me.STAFF_CODE + "', QUALIFIED =  'Y' where STUDENT_CODE = '" + studentCode + "' and SUBJECT_CODE = '" + subjectCode + "'";
	}
	int n = jdbc.executeKyomuUpdate(upd);
	count += n;
      }
      cout.println("" + count); 
    } catch (Exception e) {
      cout.println("0");    
    }
  }

  public void sendStudentPhoto(String studentCode) {
    String path = commonInfo.studentPhotoDir + studentCode;
    byte[] data = new byte[60000];
    int len = 0;
    try {
      InputStream fin = new FileInputStream(path);
      len = fin.read(data);
      fin.close();
    } catch (IOException e) {
      len = 0;
    }
    try {
      cout.println("" + len);
      ostream.write(data, 0, len);
      ostream.flush();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
    

  public void cancelRegistration(String paramValues) {
    try {
      String[] tokens = paramValues.split("\\|");
      String studentCode = tokens[0].trim();
      String schoolYear  = tokens[1].trim();
      String subjectCode = tokens[2].trim();
      String classCode   = tokens[3].trim();
      
      String del = "delete from STUDENT.REGISTR where SCHOOL_YEAR = '" + schoolYear + "' and STUDENT_CODE = '" + studentCode + "' and SUBJECT_CODE = '" + subjectCode + "' and CLASS_CODE = '" + classCode + "'";
      int res = jdbc.executeKyomuUpdate(del); 
      if (res <= 0) {
	cout.println("0");
      } else {
	cout.println("1");
      }
    } catch (Exception e) {
      cout.println("ERROR： " + e.toString().trim());
    }
  }

  public void addRegistration(String paramValues) {
    try {
      String[] tokens = paramValues.split("\\|");
      String studentCode = tokens[0];
      String schoolYear  = tokens[1];
      String subjectCode = tokens[2];
      String classCode   = tokens[3];
      String kubunCode   = tokens[4];
      String reqCode     = tokens[5];
      String unit        = tokens[6];
      String remark      = tokens[7];
      String yomikaeSubject = tokens[8];
      
      String ins = "insert into STUDENT.REGISTR ( STUDENT_CODE, SCHOOL_YEAR, SUBJECT_CODE, CLASS_CODE, KUBUN_CODE, REQ_CODE, UNIT, REMARK, YOMIKAE_SUBJ, REGISTR_DATE, REGISTR_BY, FIXED ) values ('" + studentCode + "','" + schoolYear + "','" + subjectCode + "','" + classCode + "','" + kubunCode + "','" + reqCode + "'," + unit;
      if (remark.equals(" ")) {
	ins = ins + ", null";
      } else {
	ins = ins + ",'" + remark.trim() + "'";
      }
      if (yomikaeSubject.equals(" ")) {
	ins = ins + ", null";
      } else {
	ins = ins + ",'" + yomikaeSubject.trim() + "'";
      }
      ins = ins + ",sysdate, '" + me.STUDENT_CODE + "', 'N')";
      int res = jdbc.executeKyomuUpdate(ins); 
      if (res <= 0) {
	cout.println("0");
      } else {
	cout.println("1");
      }
    } catch (Exception e) {
      cout.println("ERROR： " + e.toString().trim());
    }
  }

  public void updateGakuseki(String paramValues) {
    try {
      if (!qualified) {
	cout.println("ERROR: ユーザ認証に合格していません。"); 
	return;
      }
      String[] tokens = paramValues.split("\\|");
      String studentCode = tokens[0];
      String columnCode  = tokens[1];
      String value       = tokens[2];
      String upd;
      if (!value.equals(" ")) {      
	upd = "update MASTER.GAKUSEKI set " + columnCode + " = '" + value + "' where STUDENT_CODE = '" + studentCode + "'";
      } else {   
	upd = "update MASTER.GAKUSEKI set " + columnCode + " = null where STUDENT_CODE = '" + studentCode + "'";
      }	
      int res = jdbc.executeKyomuUpdate(upd); 
      if (res <= 0) {
	cout.println("0");
      } else {
	cout.println("1");
      }
    } catch (Exception e) {
      cout.println("ERROR： " + e.toString().trim());
    }
  }

  public void sendStudentGakuseki(String studentCode) {  
    try { 
      if (qualified) {
	if (studentCode.equals(me.STUDENT_CODE)) {
	  String str = jdbc.getGakusekiInfo(studentCode);
	  if (str != null) {
	    cout.println(str);
	  } else {
	    cout.println("null");
	  }
	} else { 
	  cout.println("ERROR: 学籍情報要求者が本人以外です。"); 	  
	}
      } else {
	cout.println("ERROR: ユーザ認証に合格していません。"); 
      }
    } catch (Exception e) {	
      cout.println("ERROR$" + e.toString().trim()); 
    }  
  }


  private boolean qualifiedToUpdateSyllabus(String teacherCode, Person me) {
    String staffCode = me.STAFF_CODE;
    int qual = 0;
    try {
      qual = Integer.parseInt(me.QUALIFICATION);
    } catch (Exception e) { }
    
    if (staffCode.equals(teacherCode)) {
      return true;
    } else if ((qual >= 3) && (qual != 7)) {
      return true;
    } else {
      return false;
    }
  }

  private boolean qualifiedToSeisekiReport(String teacherCode, Person me) {
    String staffCode = me.STAFF_CODE;
    int qual = 0;
    try {
      qual = Integer.parseInt(me.QUALIFICATION);
    } catch (Exception e) { }
    
    if (staffCode.equals(teacherCode)) {
      return true;
    } else if ((qual == 8) || (qual == 9)) {
      return true;
    } else {
      return false;
    }
  }  

  public void sendErrorMessage(String message) {
    try {
      cout.println(message);
    } catch (Exception ex) {
      ex.printStackTrace();
    }
  }

  public int getUserMapCount(String userID) {
    if (commonInfo.userMap.containsKey(userID)) {
      return commonInfo.userMap.get(userID);
    } else {
      return 0;
    }
  }
    
  public void setUserMapCount(String userID, int count) {
    if (commonInfo.userMap.containsKey(userID)) {
      commonInfo.userMap.put(userID, count);
    } else {
      commonInfo.userMap.put(userID, 0);
    }      
  }

  public void removeLoginMap() {
    commonInfo.removeLoginMap(me);
  }

  public void appendLog(String res, String userID, String STAFF_CODE, 
			   String QUALIFICATION, String inetAddress) { 
    try {
      String insert = "insert into TEACHER.TEACHER_LOG (STAFF_ID, STAFF_CODE, QUALIFICATION, LOGIN_DATE, LOGIN_RESULT, INET_ADDRESS) values ('" + userID + "','" + STAFF_CODE  + "','" + QUALIFICATION + "', sysdate, '" + res + "','" + inetAddress + "')";
      jdbc.executeKyomuUpdate(insert); 
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  /*** Backup related methods ***/

  public void sendMakeBackupResult(String tableName) { 
    try {       
      if (tableName.equals("TABBED_PANE_STRUCT")) {
	jdbc.backupTabbedPaneStruct(cin, cout);
      } else if (tableName.equals("DATA_PANEL_STRUCT")) {
	jdbc.backupDataPanelStruct(cin, cout);
      } else if (tableName.equals("TABLE_VIEW_STRUCT")) {
	jdbc.backupTableViewStruct(cin, cout);
      } else if (tableName.equals("VAR_TABLE_VIEW_STRUCT")) {
	jdbc.backupVarTableViewStruct(cin, cout);      
      } else if (tableName.equals("TABLE_COLUMN_STRUCT")) {
	jdbc.backupTableColumnStruct(cin, cout);
      } else if (tableName.equals("JIKANWARI_VIEW_STRUCT")) {
	jdbc.backupJikanwariViewStruct(cin, cout);
      } else if (tableName.equals("JIKANWARI_COLUMN_STRUCT")) {
	jdbc.backupJikanwariColumnStruct(cin, cout);
      } else if (tableName.equals("HTML_VIEW_STRUCT")) {
	jdbc.backupHtmlViewStruct(cin, cout);
      } else if (tableName.equals("SIMPLE_BUTTON_STRUCT")) {
	jdbc.backupSimpleButtonStruct(cin, cout);
      } else if (tableName.equals("UPDATE_BUTTON_STRUCT")) {
	jdbc.backupUpdateButtonStruct(cin, cout);
      } else if (tableName.equals("SERVER_DELETE_STRUCT")) {
	jdbc.backupServerDeleteStruct(cin, cout);
      } else if (tableName.equals("SERVER_UPDATE_STRUCT")) {
	jdbc.backupServerUpdateStruct(cin, cout);      
      } else if (tableName.equals("SERVER_INSERT_STRUCT")) {
	jdbc.backupServerInsertStruct(cin, cout);      
      } else if (tableName.equals("SERVER_SPECIAL_STRUCT")) {
	jdbc.backupServerSpecialStruct(cin, cout);      
      } else if (tableName.equals("SERVER_QUERY_STRUCT")) {
	jdbc.backupServerQueryStruct(cin, cout);      
      } else if (tableName.equals("COMMON_QUERY_STRUCT")) {
	jdbc.backupCommonQueryStruct(cin, cout);            
      } else if (tableName.equals("GAKUMU_CODE_DEF")) {
	jdbc.backupGakumuCodeDef(cin, cout);            
      } else if (tableName.equals("COLOR_NAME_DEF")) {
	jdbc.backupColorNameDef(cin, cout);                  
      }   
      cout.println(".");
    } catch (Exception e) {	
      cout.println("ERROR$" + e.toString().trim()); 
    }
  }

  public void sendReadBackupResult(String tableName) {
    try {
      if (tableName.equals("TABBED_PANE_STRUCT")) {
	jdbc.readTabbedPaneStruct(cin, cout);
      } else if (tableName.equals("DATA_PANEL_STRUCT")) {
	jdbc.readDataPanelStruct(cin, cout);
      } else if (tableName.equals("TABLE_VIEW_STRUCT")) {
	jdbc.readTableViewStruct(cin, cout);
      } else if (tableName.equals("VAR_TABLE_VIEW_STRUCT")) {
	jdbc.readVarTableViewStruct(cin, cout);      
      } else if (tableName.equals("TABLE_COLUMN_STRUCT")) {
	jdbc.readTableColumnStruct(cin, cout);
      } else if (tableName.equals("JIKANWARI_VIEW_STRUCT")) {
	jdbc.readJikanwariViewStruct(cin, cout);
      } else if (tableName.equals("JIKANWARI_COLUMN_STRUCT")) {
	jdbc.readJikanwariColumnStruct(cin, cout);
      } else if (tableName.equals("HTML_VIEW_STRUCT")) {
	jdbc.readHtmlViewStruct(cin, cout);
      } else if (tableName.equals("SIMPLE_BUTTON_STRUCT")) {
	jdbc.readSimpleButtonStruct(cin, cout);
      } else if (tableName.equals("UPDATE_BUTTON_STRUCT")) {
	jdbc.readUpdateButtonStruct(cin, cout);
      } else if (tableName.equals("SERVER_DELETE_STRUCT")) {
	jdbc.readServerDeleteStruct(cin, cout);
      } else if (tableName.equals("SERVER_UPDATE_STRUCT")) {
	jdbc.readServerUpdateStruct(cin, cout);      
      } else if (tableName.equals("SERVER_INSERT_STRUCT")) {
	jdbc.readServerInsertStruct(cin, cout);      
      } else if (tableName.equals("SERVER_SPECIAL_STRUCT")) {
	jdbc.readServerSpecialStruct(cin, cout);      
      } else if (tableName.equals("SERVER_QUERY_STRUCT")) {
	jdbc.readServerQueryStruct(cin, cout);      
      } else if (tableName.equals("COMMON_QUERY_STRUCT")) {
	jdbc.readCommonQueryStruct(cin, cout);            
      } else if (tableName.equals("GAKUMU_CODE_DEF")) {
	jdbc.readGakumuCodeDef(cin, cout);            
      } else if (tableName.equals("COLOR_NAME_DEF")) {
	jdbc.readColorNameDef(cin, cout);                  
      }   
      cout.println(".");
    } catch (Exception e) {	
      cout.println("ERROR$" + e.toString().trim()); 
    }
  }
  
}
