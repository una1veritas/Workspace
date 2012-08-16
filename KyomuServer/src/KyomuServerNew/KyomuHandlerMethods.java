package KyomuServerNew;
import common.*;
import java.net.*;
import java.io.*;
import java.util.*;
import java.awt.*;
import java.sql.*;

public class KyomuHandlerMethods {
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

  public KyomuHandlerMethods(CommonInfo commonInfo,
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
  }


  protected void sendPasswordCheckResult(String paramValues) {
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
	cout.println("利用不可: 学務職員以外はこのツールにログインすることができません。");
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

  protected void sendPasswordChangeResult(String paramValues) {	
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

  protected void saveKimatsuData(String paramValues) { 
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
    } catch (Exception e) {
      cout.println("0");    
    }
  }

  protected void saveSaishiData(String paramValues) { 
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
    } catch (Exception e) {
      cout.println("0");    
    }
  }

  protected void reportNinteiData(String paramValues) { 
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
 
  protected void ninteiToSeiseki(String paramValues) {
    String[] tokens2 = paramValues.split("\\|");
    String studentCode = tokens2[0].trim();
    String teacherCode = tokens2[1].trim();
    String schoolYear  = tokens2[2].trim();

    try {
      String update = "insert into SEISEKI (STUDENT_CODE, SCHOOL_YEAR, SUBJECT_CODE, CLASS_CODE, TEACHER_CODE, MARKS, KUBUN_CODE, REQ_CODE, UNIT, REMARK, YOMIKAE_SUBJ, REGISTR_DATE, REGISTR_BY, REPORTED_DATE, REPORTED_BY)  select Q.STUDENT_CODE, '"+schoolYear+"', Q.SUBJECT_CODE, '00', '"+teacherCode+"', 200, C.KUBUN_CODE, C.REQ_CODE, C.UNIT, Q.EVIDENCE, null, null, null, Q.QUALIFIED_DATE, Q.QUALIFIED_BY from TEACHER.QUALIFY_INPUT Q, MASTER.CURRICULUM C, MASTER.STUDENT S where Q.STUDENT_CODE = '" + studentCode + "' and Q.QUALIFIED = 'Y' and Q.STUDENT_CODE = S.STUDENT_CODE and S.CURRICULUM_YEAR = C.CURRICULUM_YEAR and S.FACULTY = C.FACULTY and ((S.DEPARTMENT = C.DEPARTMENT) or (C.DEPARTMENT = '30')) and S.COURSE = C.COURSE and Q.SUBJECT_CODE = C.SUBJECT_CODE";
      int count = jdbc.executeKyomuUpdate(update);
      cout.println("" + count); 
    } catch (Exception e) {
      cout.println("ERROR$"+e.toString());
    }
  }
  
  protected void kimatsuReportToSeiseki(String paramValues) {
    String[] tokens2 = paramValues.split("\\|");
    String schoolYear  = tokens2[0].trim();
    String subjectCode = tokens2[1].trim();
    String classCode   = tokens2[2].trim();

    String update;
    int ins_seiseki_count;
    int ins_saishi_count;
    int update_rtoroku_count;

    try {
      update = "insert into SEISEKI (STUDENT_CODE, SCHOOL_YEAR, SUBJECT_CODE, CLASS_CODE, TEACHER_CODE, MARKS, KUBUN_CODE, REQ_CODE, UNIT, REMARK, YOMIKAE_SUBJ, REGISTR_DATE, REGISTR_BY, REPORTED_DATE, REPORTED_BY) select R.STUDENT_CODE, R.SCHOOL_YEAR, R.SUBJECT_CODE, R.CLASS_CODE, C.TEACHER_CODE, R.MARKS, R.KUBUN_CODE, R.REQ_CODE, R.UNIT, R.REMARK, R.YOMIKAE_SUBJ, R.REGISTR_DATE, R.REGISTR_BY, R.HOKOKU_DATE, R.HOKOKU_BY from KYOMU.RTOROKU R, MASTER.CLASS_INFO C where R.SCHOOL_YEAR = C.SCHOOL_YEAR and R.SUBJECT_CODE = C.SUBJECT_CODE and R.CLASS_CODE = C.CLASS_CODE and R.SCHOOL_YEAR = '"+schoolYear+"' and R.SUBJECT_CODE = '"+subjectCode+"' and R.CLASS_CODE = '"+classCode+"' and R.FIXED = 'Y' and (R.MARKS >= 60 or R.MARKS = 0)";
      ins_seiseki_count = jdbc.executeKyomuUpdate(update);

      String msg = "select distinct FACULTY from MASTER.JIKANWARI where SCHOOL_YEAR = '"+schoolYear+"' and SUBJECT_CODE = '"+subjectCode+"' and CLASS_CODE = '"+classCode+"'";      
      String res = jdbc.executeKyomuQuery(msg);      
      String[] lines = res.split("\\$");
      String[] tokens = lines[0].split("\\|");	
      String faculty = tokens[0];

      if (faculty.equals("32")) {

	update = "insert into SEISEKI (STUDENT_CODE, SCHOOL_YEAR, SUBJECT_CODE, CLASS_CODE, TEACHER_CODE, MARKS, KUBUN_CODE, REQ_CODE, UNIT, REMARK, YOMIKAE_SUBJ, REGISTR_DATE, REGISTR_BY, REPORTED_DATE, REPORTED_BY) select R.STUDENT_CODE, R.SCHOOL_YEAR, R.SUBJECT_CODE, R.CLASS_CODE, C.TEACHER_CODE, 0, R.KUBUN_CODE, R.REQ_CODE, R.UNIT, R.REMARK, R.YOMIKAE_SUBJ, R.REGISTR_DATE, R.REGISTR_BY, R.HOKOKU_DATE, R.HOKOKU_BY from KYOMU.RTOROKU R, MASTER.CLASS_INFO C where R.SCHOOL_YEAR = C.SCHOOL_YEAR and R.SUBJECT_CODE = C.SUBJECT_CODE and R.CLASS_CODE = C.CLASS_CODE and R.SCHOOL_YEAR = '"+schoolYear+"' and R.SUBJECT_CODE = '"+subjectCode+"' and R.CLASS_CODE = '"+classCode+"' and R.FIXED = 'Y' and (R.MARKS < 60 and R.MARKS != 0)";
	ins_seiseki_count = jdbc.executeKyomuUpdate(update);

      } else {

	update = "insert into SAISHI (STUDENT_CODE, SCHOOL_YEAR, SUBJECT_CODE, CLASS_CODE, TEACHER_CODE, MARKS, KUBUN_CODE, REQ_CODE, UNIT, REMARK, YOMIKAE_SUBJ, REGISTR_DATE, REGISTR_BY, HOKOKU_DATE, HOKOKU_BY, FIXED) select R.STUDENT_CODE, R.SCHOOL_YEAR, R.SUBJECT_CODE, R.CLASS_CODE, C.TEACHER_CODE, null, R.KUBUN_CODE, R.REQ_CODE, R.UNIT, R.REMARK, R.YOMIKAE_SUBJ, R.REGISTR_DATE, R.REGISTR_BY, R.HOKOKU_DATE, R.HOKOKU_BY, 'N' from KYOMU.RTOROKU R, MASTER.CLASS_INFO C where R.SCHOOL_YEAR = C.SCHOOL_YEAR and R.SUBJECT_CODE = C.SUBJECT_CODE and R.CLASS_CODE = C.CLASS_CODE and R.SCHOOL_YEAR = '"+schoolYear+"' and R.SUBJECT_CODE = '"+subjectCode+"' and R.CLASS_CODE = '"+classCode+"' and R.FIXED = 'Y' and (MARKS < 60 and MARKS != 0)";
	ins_saishi_count = jdbc.executeKyomuUpdate(update);

      }
      
      update = "update RTOROKU set FIXED = 'F' where SCHOOL_YEAR = '"+schoolYear+"' and SUBJECT_CODE = '"+subjectCode+"' and CLASS_CODE = '"+classCode+"' and FIXED = 'Y'";
      update_rtoroku_count = jdbc.executeKyomuUpdate(update);

      cout.println("" + update_rtoroku_count); 
    } catch (Exception e) {
      cout.println("ERROR$"+e.toString());
    }
  }

  protected void saishiReportToSeiseki(String paramValues) {
    String[] tokens2 = paramValues.split("\\|");
    String schoolYear  = tokens2[0].trim();
    String subjectCode = tokens2[1].trim();
    String classCode   = tokens2[2].trim();
    String teacherCode = tokens2[3].trim();

    String update;
    int ins_seiseki_count;
    try {
      update = "insert into SEISEKI (STUDENT_CODE, SCHOOL_YEAR, SUBJECT_CODE, CLASS_CODE, TEACHER_CODE, MARKS, KUBUN_CODE, REQ_CODE, UNIT, REMARK, YOMIKAE_SUBJ, REGISTR_DATE, REGISTR_BY, REPORTED_DATE, REPORTED_BY) select STUDENT_CODE, SCHOOL_YEAR, SUBJECT_CODE, CLASS_CODE, TEACHER_CODE, MARKS, KUBUN_CODE, REQ_CODE, UNIT, REMARK, YOMIKAE_SUBJ, REGISTR_DATE, REGISTR_BY, HOKOKU_DATE, HOKOKU_BY from KYOMU.SAISHI where SCHOOL_YEAR = '"+schoolYear+"' and SUBJECT_CODE = '"+subjectCode+"' and CLASS_CODE = '"+classCode+"' and TEACHER_CODE = '"+teacherCode+"' and FIXED = 'Y' and (MARKS >= 60 or MARKS = 0)";
      ins_seiseki_count = jdbc.executeKyomuUpdate(update);
 
      update = "update SAISHI set FIXED = 'N', MARKS = null where SCHOOL_YEAR = '"+schoolYear+"' and SUBJECT_CODE = '"+subjectCode+"' and CLASS_CODE = '"+classCode+"' and TEACHER_CODE = '"+teacherCode+"' and FIXED = 'Y' and (MARKS < 60 and MARKS != 0)";
      int ins_saishi_count = jdbc.executeKyomuUpdate(update);
            
      update = "update SAISHI set FIXED = 'F' where SCHOOL_YEAR = '"+schoolYear+"' and SUBJECT_CODE = '"+subjectCode+"' and CLASS_CODE = '"+classCode+"' and TEACHER_CODE = '"+teacherCode+"' and FIXED = 'Y'";
      int update_saishi_count = jdbc.executeKyomuUpdate(update);

      cout.println("" + update_saishi_count); 
    } catch (Exception e) {
      cout.println("ERROR$"+e.toString());
    }
  }

  protected void sendQueryStaffAttribResult(String param) {	
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
		sbuf.append(key).append("|").append(val).append("$");
	      }
	      cout.println(sbuf.toString());
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

  protected void sendErrorMessage(String message) {
    try {
      cout.println(message);
    } catch (Exception ex) {
      ex.printStackTrace();
    }
  }

  protected int getUserMapCount(String userID) {
    if (commonInfo.userMap.containsKey(userID)) {
      return commonInfo.userMap.get(userID);
    } else {
      return 0;
    }
  }
    
  protected void setUserMapCount(String userID, int count) {
    if (commonInfo.userMap.containsKey(userID)) {
      commonInfo.userMap.put(userID, count);
    } else {
      commonInfo.userMap.put(userID, 0);
    }      
  }

  protected void removeLoginMap() {
    commonInfo.removeLoginMap(me);
  }
  
  protected void appendLog(String res, String userID, String STAFF_CODE, String QUALIFICATION,  String inetAddress) { 
    try {
      String insert = "insert into KYOMU.KYOMU_LOG (STAFF_ID, STAFF_CODE, QUALIFICATION, TOOL_NAME, LOGIN_DATE, LOGIN_RESULT, INET_ADDRESS) values ('" + userID + "','" + STAFF_CODE  + "','" + QUALIFICATION + "','KyomuTool', sysdate, '" + res + "','" + inetAddress + "')";
      jdbc.executeKyomuUpdate(insert);     
    } catch (Exception e) {
      e.printStackTrace();
    }
  } 
}
