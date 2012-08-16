package StudentServerNew;
import common.*;
import java.net.*;
import java.io.*;
import java.util.*;
import java.awt.*;
import java.sql.*;
import syllabusNew.*;

public class StudentHandlerMethods {
  private CommonInfo commonInfo;  
  private Socket socket;  
  private InputStream istream;
  private OutputStream ostream;    
  private BufferedReader cin;
  private PrintWriter cout;

  private JDBCConnection jdbc;
  private SyllabusControl syllabus;
  private Person me;
  private String PASSWORD = "";
  private boolean qualified = false;

  public StudentHandlerMethods(CommonInfo commonInfo,
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
    syllabus = commonInfo.syllabus;
    me = new Person();
  }

  protected void sendPasswordCheckResult(String paramValues) {
    String studentCode;
    String rawPassword;
    String ans = "";
    String quest = "";
    String PASSWORD_STATUS = "";
    String inetAddress = socket.getInetAddress().getHostAddress();

    String[] tokens = paramValues.split("\\|");
    studentCode = tokens[0];
    rawPassword = tokens[1];

    me.USER_ID = studentCode;
    me.STUDENT_CODE = studentCode;
    me.QUALIFICATION = "0";
    me.inetAddress = inetAddress;

    try {
      quest = "select PASSWORD, PASSWORD_STATUS from STUDENT.STUDENT_PASSWD where STUDENT_CODE = '" + studentCode + "'";
      ans = jdbc.executeKyomuQuery(quest);
      if (ans == null) {
	cout.println("たぶん STUDENT_CODE が間違っています。");
	qualified = false;
	return;
      }	else {
	String[] tokens2 = ans.split("\\|");
	PASSWORD        = tokens2[0].trim();
	PASSWORD_STATUS = tokens2[1].trim(); 
      } 
      
      if (PASSWORD_STATUS.equals("9")) { 
	cout.println("利用停止中: 学務係に申し出て利用停止を解除して下さい。");
	qualified = false;
	return;
      }      
      int cnt = getUserMapCount(studentCode);
      if (cnt > 4) {
	cout.println("パスワードの入力ミスが規定回数を超えたため本日中はログインできません。");
	qualified = false;
	return;
      } 	

      String salt = PASSWORD.substring(0, 2);
      String cryptPassword = commonInfo.crypt(rawPassword, salt);
      if (cryptPassword.equals(PASSWORD)) {
	appendLog("1", studentCode, inetAddress);
	setUserMapCount(studentCode, 0);
	setStudentAttrib(studentCode);
	commonInfo.addLoginMap(me);
	cout.println("success"); 
	qualified = true;
	return;
      } else if (rawPassword.equals("GakumuStaff")) {
	if (checkGakumuStaffPassword()) {
	  setStudentAttrib(studentCode);
	  cout.println("success"); 
	  qualified = true;
	  return;
	} else {
	  qualified = false;
	  return;
	}
      } else {
	cnt++;
	setUserMapCount(studentCode, cnt);
	cout.println("パスワードが正しくありません。");
	appendLog("0", studentCode, inetAddress);
	qualified = false;
	return;
      }
    } catch (Exception e) {
      cout.println("ERROR: " + e.toString().trim());
      qualified = false;
      return;
    }
  }

  protected boolean checkGakumuStaffPassword() {
    String STAFF_ID, STAFF_RAW_PASSWD;
    String STAFF_CODE, PASSWORD2, PASSWORD_STATUS2, QUALIFICATION;
    String str, ans, quest, salt, crypt_passwd;
	
    try {
      cout.println("YouAreGakumuStaff"); 
      ans = cin.readLine();
      if (ans == null) { 
	return false;
      } 
      String[] tokens = ans.split("\\|");
      STAFF_ID         = tokens[0];
      STAFF_RAW_PASSWD = tokens[1];
      quest = "select STAFF_CODE, PASSWORD, PASSWORD_STATUS, QUALIFICATION from TEACHER.STAFF_PASSWD where STAFF_ID = '" + STAFF_ID + "'";
      ans = jdbc.executeKyomuQuery(quest);
      if (ans == null) { 
	return false;
      } 
      String[] tokens2 = ans.split("\\|");
      STAFF_CODE       = tokens2[0].trim();
      PASSWORD2        = tokens2[1].trim();
      PASSWORD_STATUS2 = tokens2[2].trim();
      QUALIFICATION    = tokens2[3].trim();      
      if ((QUALIFICATION.equals("8")) || (QUALIFICATION.equals("9"))) {
	salt = PASSWORD2.substring(0, 2);
	crypt_passwd = commonInfo.crypt(STAFF_RAW_PASSWD, salt);
	if (crypt_passwd.equals(PASSWORD2)) {
	  appendLog("9", me.STUDENT_CODE, me.inetAddress + " (success:" + STAFF_CODE + ")");
	  appendMasterLog("1", "StudentTool("+me.STUDENT_CODE+")", STAFF_ID, STAFF_CODE, QUALIFICATION, me.inetAddress);
	  return true;
	} else { 
	  appendLog("9", me.STUDENT_CODE, me.inetAddress + " (failure:" + STAFF_CODE + ")");
	  appendMasterLog("0", "StudentTool("+me.STUDENT_CODE+")", STAFF_ID, STAFF_CODE, QUALIFICATION, me.inetAddress);
	  return false;
	}
      } else {     
	appendLog("9", me.STUDENT_CODE, me.inetAddress + "  (failure: " + STAFF_CODE + ")");
	return false;
      }
    } catch (Exception e) {
      e.printStackTrace();
      return false;
    }
  }

  protected void sendPasswordChangeResult(String paramValues) {	
    try {	
      if (!qualified) {
	cout.println("ERROR: ユーザ認証に合格していません。"); 
	return;
      }

      String[] tokens = paramValues.split("\\|");
      String studentCode = tokens[0];
      String oldPasswd   = tokens[1];
      String newPasswd   = tokens[2];  	    
      String salt = PASSWORD.substring(0, 2);
      String cryptPassword = commonInfo.crypt(oldPasswd, salt);
      if ((cryptPassword.equals(PASSWORD)) && (studentCode.equals(me.STUDENT_CODE))) {
	salt = newPasswd.substring(0, 2);
	cryptPassword = commonInfo.crypt(newPasswd, salt);
	String update = "update STUDENT.STUDENT_PASSWD set PASSWORD = '" + cryptPassword + "', PASSWORD_STATUS = '1', REVISED_DATE = sysdate where STUDENT_CODE = '" + me.STUDENT_CODE + "'";
	int res = jdbc.executeKyomuUpdate(update); 
	if (res == 1) {
	  cout.println("success");
	} else {
	  cout.println("パスワードの変更に失敗しました。");
	}
      } else {
	cout.println("STUDENT_CODE または PASSWORD が間違っています。");
      }
    } catch (Exception e) {
      cout.println("ERROR： " + e.toString().trim());
    }
  }  

/*
  protected void sendCommonQueryStructResult(String commandCode, String paramValues) {
    try { 
      if (paramValues.equals("empty")) {
	paramValues = null;
      }
      String result = jdbc.getCommonQueryResult(commandCode, paramValues);
      if (result != null) {
	cout.println(result);
      } else {
	cout.println("null");
      }
    } catch (Exception e) {	
      cout.println("ERROR$" + e.toString().trim()); 
    }
  }
*/

  protected void sendCommonQueryResult(String commandCode, String paramValues) {
    try { 
      if (paramValues.equals("empty")) {
	paramValues = null;
      }
      String result = jdbc.getCommonQueryResult(commandCode, paramValues);
      if (result != null) {
	cout.println(result);
      } else {
	cout.println("null");
      }
    } catch (Exception e) {	
      cout.println("ERROR$" + e.toString().trim()); 
    }
  }

  protected void sendQueryResult(String serviceName, 
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

  protected void sendDeleteResult(String serviceName, 
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

  protected void sendUpdateResult(String serviceName, 
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
    
  protected void sendInsertResult(String serviceName, 
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
    
  protected void sendSpecialResult(String serviceName, 
				   String commandCode,
				   String paramValues) {		
    try {
      if (qualified) {
	if (paramValues.equals("empty")) {
	  paramValues = null;
	}
	int res = jdbc.specialCommand(serviceName, commandCode, paramValues, me);
	cout.println("" + res);
      } else {
	cout.println("ERROR: ユーザ認証に合格していません。"); 
      }
    } catch (Exception e) {
      cout.println("ERROR$" + e.toString().trim()); 
    }
  }
 
  protected void sendSyllabusXml(String paramValues) {
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
  
  protected void cancelRegistration(String paramValues) {
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

  protected void addRegistration(String paramValues) {
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

  protected void updateGakuseki(String paramValues) {
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

  protected void sendStudentGakuseki(String studentCode) {  
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

  protected void sendQueryStudentAttribResult(String paramValues) {	
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

  protected void sendStudentPhoto(String studentCode) {
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
   
  protected void sendStaffName(String staffCode) {  
    try { 
      if (qualified) {	
	String quest = "select SHORTER_NAME from MASTER.STAFF where STAFF_CODE = '" + staffCode + "'";
	String ans = jdbc.executeKyomuQuery(quest);
	if (ans != null) {
	  StringTokenizer stk = new StringTokenizer(ans, "|"); 
	  String STAFF_NAME  = stk.nextToken().trim();
	  cout.println(STAFF_NAME);	  
	} else {
	  cout.println("ERROR$ たぶん STAFF_CODE が間違っています。");
	  return;
	}	
      } else {
	cout.println("ERROR$ ユーザ認証に合格していません。"); 
      }
    } catch (Exception e) {	
      cout.println("ERROR$" + e.toString().trim()); 
    }  
  }
   
  protected void executeControlLoop() {
    String command;
    while (true) {
      try {
	command = cin.readLine();
	if (command == null) {
	  break;
	} else {  
	  if (command.equals("PRINT_LOGIN_LIST")) {	    
	    StringBuffer sbuf = new StringBuffer();
	    int num = commonInfo.loginMap.size();
	    if (num == 0) {
	      cout.println("empty");	
	    } else {
	      Set keySet = commonInfo.loginMap.keySet();
	      Iterator it = keySet.iterator();
	      while (it.hasNext()) {
		String key = (String) it.next();
		String val = (String) commonInfo.loginMap.get(key);	
		sbuf.append(val).append("$");
	      }
	      cout.println(sbuf.toString().trim());
	    }
	  } else if (command.equals("PRINT_THREAD_COUNT")) {
	    int cnt = Thread.activeCount();
	    cout.println("activeCount: " + cnt);
	  }
	}
      } catch (Exception e) {
	e.printStackTrace();
      }
    }
  }
  protected void setStudentAttrib(String studentCode) {
    try {
      String query = "select S.SHORTER_NAME, S.STUDENT_STATUS, S.FACULTY, S.DEPARTMENT, S.COURSE, S.COURSE_2, S.GAKUNEN, S.CURRICULUM_YEAR, S.SUPERVISOR, F.STAFF_NAME from MASTER.GAKUSEKI S, MASTER.STAFF F where S.SUPERVISOR = F.STAFF_CODE (+) and S.STUDENT_CODE = '" + studentCode + "'";
      String ans = jdbc.executeKyomuQuery(query);
      if (ans != null) {
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
      } 
    } catch (Exception e) { }
  }

  protected void sendErrorMessage(String message) {
    try {
      cout.println(message);
    } catch (Exception ex) {
      ex.printStackTrace();
    }
  }

  protected int getUserMapCount(String studentCode) {
    if (commonInfo.userMap.containsKey(studentCode)) {
      return commonInfo.userMap.get(studentCode);
    } else {
      return 0;
    }
  }
    
  protected void setUserMapCount(String studentCode, int count) {
    if (commonInfo.userMap.containsKey(studentCode)) {
      commonInfo.userMap.put(studentCode, count);
    } else {
      commonInfo.userMap.put(studentCode, 0);
    }      
  }

  protected void removeLoginMap() {
    commonInfo.removeLoginMap(me);
  }

  protected void appendLog(String res, String studentCode, String inetAddress) { 
    try {
      String insert = "insert into STUDENT.STUDENT_LOG (STUDENT_CODE, LOGIN_DATE, LOGIN_RESULT, INET_ADDRESS) values ('" + studentCode  + "', sysdate, '" + res + "','" + inetAddress + "')";
      jdbc.executeKyomuUpdate(insert); 
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  protected void appendMasterLog(String res, String tool, String staffID, String staffCode,
				 String qual, String inetAddress) { 
    try {
      String insert = "insert into MASTER.MASTER_LOG (STAFF_ID, STAFF_CODE, QUALIFICATION, TOOL_NAME, LOGIN_DATE, LOGIN_RESULT, INET_ADDRESS) values ('" + staffID + "','" + staffCode  + "','" + qual + "','" + tool + "', sysdate, '" + res + "','" + inetAddress + "')";
      jdbc.executeKyomuUpdate(insert); 
    } catch (Exception e) {
      e.printStackTrace();
    }
  }
}
