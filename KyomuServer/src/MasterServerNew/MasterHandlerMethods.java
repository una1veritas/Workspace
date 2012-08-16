package MasterServerNew;
import common.*;
import java.net.*;
import java.io.*;
import java.util.*;
import java.awt.*;
import java.sql.*;

public class MasterHandlerMethods {
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

  public MasterHandlerMethods(CommonInfo commonInfo,
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

  protected void updateGakuseki(String paramValues) {
    try {
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
	String str = jdbc.getGakusekiInfo(studentCode);
	if (str != null) {
	  cout.println(str);
	} else {
	  cout.println("null");
	}
      } else {
	cout.println("ERROR: ユーザ認証に合格していません。"); 
      }
    } catch (Exception e) {	
      cout.println("ERROR$" + e.toString().trim()); 
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

  protected void initializeAndSendStaffInitialPassword(String staffID) {  
    String update, INIT_PASSWORD, salt, cpassword;
    try {
      INIT_PASSWORD = commonInfo.getInitialPassword();
      salt = commonInfo.getSalt();
      cpassword = commonInfo.crypt(INIT_PASSWORD, salt);	  
      update = "update TEACHER.STAFF_PASSWD set PASSWORD = '"+cpassword+"', PASSWORD_STATUS = '0', REVISED_DATE = sysdate where STAFF_ID = '" + staffID + "'";
      int res = jdbc.executeKyomuUpdate(update); 
      if (res != 1) {
	cout.println("ERROR$ " + staffID + " のパスワード初期化に失敗しました。");
	return;
      } else {
	cout.println(INIT_PASSWORD);
	return;
      }
    } catch (Exception e) {
      cout.println("ERROR： " + e.toString().trim());
    }
  }

  protected void registrateAndSendStaffInitialPassword(String paramValues) { 
    String update, INIT_PASSWORD, salt, cpassword;
    try {
      String[] tokens = paramValues.split("\\|");
      String staffID     = tokens[0];
      String staffCode   = tokens[1];
      String qualification = tokens[2];

      if (!me.QUALIFICATION.equals("8")) {
	int qual = Integer.parseInt(qualification);
	if (qual >= 3) {
	  cout.println("ERROR$ あなたにはその資格のユーザを登録する権限がありません。");
	  return;
	}
      }	
      INIT_PASSWORD = commonInfo.getInitialPassword();
      salt = commonInfo.getSalt();
      cpassword = commonInfo.crypt(INIT_PASSWORD, salt);
      String ins = "insert into TEACHER.STAFF_PASSWD (STAFF_CODE, STAFF_ID, QUALIFICATION, PASSWORD, PASSWORD_STATUS, REVISED_DATE) values ('"+staffCode+"','"+staffID+"','"+qualification+"','"+cpassword+"','0', sysdate)";
      int res = jdbc.executeKyomuUpdate(ins); 
      if (res != 1) {
	cout.println("ERROR$ " + staffID + " のユーザ登録には失敗しました。");
	return;
      } else {
	cout.println(INIT_PASSWORD);
	return;
      }
    } catch (Exception e) {
      cout.println("ERROR： " + e.toString().trim());
    }
  }

  protected void sendQueryStaffAttribResult(String paramValues) {	
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

  protected void updateCalendar(String serviceName, String commandCode, 
				String paramValues) {
    try {
      String[] tokens = paramValues.split("\\|");
      if (serviceName.equals("SCHOOLEVENT")) {
	if (commandCode.equals("ADD")) {
	  String year  = tokens[0].trim();
	  String month = tokens[1].trim();
	  String day   = tokens[2].trim();
	  String schoolEvent = tokens[3].trim();
	  String upd = "update MASTER.CALENDAR set SCHOOL_EVENTS = SCHOOL_EVENTS || '  ' || '" + schoolEvent + "' where YEAR = '" + year + "' and MONTH = " + month + " and DAY = " + day;	
	  int res = jdbc.executeKyomuUpdate(upd); 
	  if (res > 0) {  
	    cout.println("1");
	    return;
	  } else {
	    String ins = "insert into MASTER.CALENDAR (YEAR, MONTH, DAY, HOLIDAY_INFO, SCHOOL_EVENTS, CLASS_WEEK) values ('"+ year +"',"+ month +","+ day +", null, '"+ schoolEvent +"', null)";
	    int res2 = jdbc.executeKyomuUpdate(ins); 
	    if (res2 > 0) {  
	      cout.println("1");
	      return;
	    } else {
	      cout.println("0");
	      return;
	    }
	  }	  
	} else if (commandCode.equals("DELETE")) {
	  String year  = tokens[0].trim();
	  String month = tokens[1].trim();
	  String day   = tokens[2].trim();
	  String del = "update MASTER.CALENDAR set SCHOOL_EVENTS = null where YEAR = '" + year + "' and MONTH = " + month + " and DAY = " + day;
	  int res = jdbc.executeKyomuUpdate(del); 
	  if (res <= 0) {
	    cout.println("0");
	    return;
	  } else {
	    cout.println("1");
	    return;
	  }	  
	}
      } else if (serviceName.equals("HOLIDAYINFO")) {
	if (commandCode.equals("ADD")) {
	  String year  = tokens[0].trim();
	  String month = tokens[1].trim();
	  String day   = tokens[2].trim();
	  String holidayInfo = tokens[3].trim();	  
	  String upd = "update MASTER.CALENDAR set HOLIDAY_INFO = HOLIDAY_INFO || '  ' || '" + holidayInfo + "' where YEAR = '" + year + "' and MONTH = " + month + " and DAY = " + day;	
	  int res = jdbc.executeKyomuUpdate(upd); 
	  if (res > 0) {  
	    cout.println("1");
	    return;
	  } else {
	    String ins = "insert into MASTER.CALENDAR (YEAR, MONTH, DAY, HOLIDAY_INFO, SCHOOL_EVENTS, CLASS_WEEK) values ('"+ year +"',"+ month +","+ day +",'"+ holidayInfo +"', null, null)";
	    int res2 = jdbc.executeKyomuUpdate(ins); 
	    if (res2 > 0) {  
	      cout.println("1");
	      return;
	    } else {
	      cout.println("0");
	      return;
	    }
	  }	
	} else if (commandCode.equals("DELETE")) {
	  String year  = tokens[0].trim();
	  String month = tokens[1].trim();
	  String day   = tokens[2].trim();
	  String del = "update MASTER.CALENDAR set HOLIDAY_INFO = null where YEAR = '" + year + "' and MONTH = " + month + " and DAY = " + day;
	  int res = jdbc.executeKyomuUpdate(del); 
	  if (res <= 0) {
	    cout.println("0");
	    return;
	  } else {
	    cout.println("1");
	    return;
	  }		  
	}
      } else if (serviceName.equals("CLASSWEEK")) {
	if (commandCode.equals("ADD")) {
	  String year  = tokens[0].trim();
	  String month = tokens[1].trim();
	  String day   = tokens[2].trim();
	  String week  = tokens[3].trim();	  
	  String upd = "update MASTER.CALENDAR set CLASS_WEEK = '" + week + "' where YEAR = '" + year + "' and MONTH = " + month + " and DAY = " + day;	
	  int res = jdbc.executeKyomuUpdate(upd); 
	  if (res > 0) {  
	    cout.println("1");
	    return;
	  } else {
	    String ins = "insert into MASTER.CALENDAR (YEAR, MONTH, DAY, HOLIDAY_INFO, SCHOOL_EVENTS, CLASS_WEEK) values ('"+ year +"',"+ month +","+ day +", null, null, '"+week+"')";
	    int res2 = jdbc.executeKyomuUpdate(ins); 
	    if (res2 > 0) {  
	      cout.println("1");
	      return;
	    } else {
	      cout.println("0");
	      return;
	    }
	  }	
	} else if (commandCode.equals("DELETE")) {
	  String year  = tokens[0].trim();
	  String month = tokens[1].trim();
	  String day   = tokens[2].trim();
	  String del = "update MASTER.CALENDAR set CLASS_WEEK = null where YEAR = '" + year + "' and MONTH = " + month + " and DAY = " + day;
	  int res = jdbc.executeKyomuUpdate(del); 
	  if (res <= 0) {
	    cout.println("0");
	    return;
	  } else {
	    cout.println("1");
	    return;
	  }		  
	}
      }
      cout.println("0");      
    } catch (Exception e) {
      cout.println("ERROR$ " + e.toString().trim());
    }
  }

  protected void jikanwariControl(String commandCode, String paramValues) {
    if (commandCode.equals("ADDSUBJECTCLASSTOJIKANWARI")) {
      // SCHOOL_YEAR|FACULTY|SEMESTER|GAKUNEN|WEEK|DEPARTMENT|HOUR|SUBJECT_CODE|CLASS_CODE| 
      try {
	String[] tokens = paramValues.split("\\|");
	String schoolYear  = tokens[0].trim();
	String faculty     = tokens[1].trim();
	String semester    = tokens[2].trim();
	String gakunen     = tokens[3].trim();
	String week        = tokens[4].trim();
	String dept        = tokens[5].trim();
	String hour        = tokens[6].trim();
	String subjectCode = tokens[7].trim();
	String classCode   = tokens[8].trim();	
	String query = "select count(*) from MASTER2.CURRICULUM_T where ((GAKUNEN = '"+gakunen+"' and CURRICULUM_YEAR = ('"+schoolYear+"' - GAKUNEN + 1)) or (GAKUNEN = ('"+gakunen+"' - 1) and CURRICULUM_YEAR = ('"+schoolYear+"' - GAKUNEN))) and FACULTY = '"+faculty+"' and DEPARTMENT = '"+dept+"' and COURSE = '10' and SUBJECT_CODE = '"+subjectCode+"'";
	String ans = jdbc.executeKyomuQuery(query);
	StringTokenizer stk2 = new StringTokenizer(ans, "|"); 
	String cntStr  = stk2.nextToken().trim();
	if (cntStr.equals("0")) {
	  cout.println("0");
	  return;
	} else {
	  String insert = "insert into JIKANWARI (SCHOOL_YEAR, FACULTY, DEPARTMENT, GAKUNEN, SEMESTER, WEEK, HOUR, SUBJECT_CODE, CLASS_CODE, ROOM) values('"+schoolYear+"','"+faculty+"','"+dept+"','"+gakunen+"','"+semester+"','"+week+"','"+hour+"','"+subjectCode+"','"+classCode+"',null)";
	  int res = jdbc.executeKyomuUpdate(insert); 
	  if (res == 0) {
	    cout.println("0");
	  } else {
	    cout.println("1");
	  }
	  return;
	}
      } catch (Exception e) {
	cout.println("ERROR$ " + e.toString().trim());
      }
    } else if (commandCode.equals("DELETESUBJECTCLASSFROMJIKANWARI")) {
      // SCHOOL_YEAR|FACULTY|SEMESTER|GAKUNEN|WEEK|SUBJECT_CODE|CLASS_CODE|DEPARTMENT|HOUR| 
      try {
	String[] tokens = paramValues.split("\\|");
	String schoolYear  = tokens[0].trim();
	String faculty     = tokens[1].trim();
	String semester    = tokens[2].trim();
	String gakunen     = tokens[3].trim();
	String week        = tokens[4].trim();
	String subjectCode = tokens[5].trim();
	String classCode   = tokens[6].trim();
	String dept        = tokens[7].trim();
	String hour        = tokens[8].trim();		
	String delete = "delete from JIKANWARI where SCHOOL_YEAR = '"+schoolYear+"' and FACULTY = '"+faculty+"' and DEPARTMENT = '"+dept+"' and GAKUNEN = '"+gakunen+"' and SEMESTER = '"+semester+"' and WEEK = '"+week+"' and HOUR = '"+hour+"' and SUBJECT_CODE = '"+subjectCode+"' and CLASS_CODE = '"+classCode+"'";
	int res = jdbc.executeKyomuUpdate(delete); 
	if (res <= 0) {
	  cout.println("0");
	} else {
	  cout.println("1");
	}		
	return; 
      } catch (Exception e) {
	cout.println("ERROR$ " + e.toString().trim());
      }
    } else if (commandCode.equals("DELETESUBJECTCLASSFROMJIKANWARI2")) {
      // SCHOOL_YEAR|FACULTY|SEMESTER|GAKUNEN|WEEK|SUBJECT_CODE|CLASS_CODE|DEPARTMENT|HOUR| 
      try {
	String[] tokens = paramValues.split("\\|");
	String schoolYear  = tokens[0].trim();
	String faculty     = tokens[1].trim();
	String semester    = tokens[2].trim();
	String gakunen     = tokens[3].trim();
	String week        = tokens[4].trim();
	String subjectCode = tokens[5].trim();
	String classCode   = tokens[6].trim();
	String dept        = tokens[7].trim();
	String hour        = tokens[8].trim();
	String delete = "delete from JIKANWARI where SCHOOL_YEAR = '"+schoolYear+"' and FACULTY = '"+faculty+"' and SEMESTER = '"+semester+"' and WEEK = '"+week+"' and SUBJECT_CODE = '"+subjectCode+"' and CLASS_CODE = '"+classCode+"'";
	int res = jdbc.executeKyomuUpdate(delete); 
	if (res <= 0) {
	  cout.println("0");
	} else {
	  cout.println("1");
	}		
	return; 
      } catch (Exception e) {
	cout.println("ERROR$ " + e.toString().trim());
      }
    } else if (commandCode.equals("SETROOMTOSUBJECTCLASSOFJIKANWARI")) {
      // SCHOOL_YEAR|FACULTY|SEMESTER|GAKUNEN|WEEK|SUBJECT_CODE|CLASS_CODE|DEPARTMENT|HOUR|ROOM| 
      try {
	String[] tokens = paramValues.split("\\|");
	String schoolYear  = tokens[0].trim();
	String faculty     = tokens[1].trim();
	String semester    = tokens[2].trim();
	String gakunen     = tokens[3].trim();
	String week        = tokens[4].trim();
	String subjectCode = tokens[5].trim();
	String classCode   = tokens[6].trim();
	String dept        = tokens[7].trim();
	String hour        = tokens[8].trim();
	String room        = tokens[9].trim();
	String update;
	if (room.trim().equals("")) {
	  update = "update JIKANWARI set ROOM = null where SCHOOL_YEAR = '"+schoolYear+"' and FACULTY = '"+faculty+"' and DEPARTMENT = '"+dept+"' and GAKUNEN = '"+gakunen+"' and SEMESTER = '"+semester+"' and WEEK = '"+week+"' and HOUR = '"+hour+"' and SUBJECT_CODE = '"+subjectCode+"' and CLASS_CODE = '"+classCode+"'";
	} else {	
	  update = "update JIKANWARI set ROOM = '"+room+"' where SCHOOL_YEAR = '"+schoolYear+"' and FACULTY = '"+faculty+"' and DEPARTMENT = '"+dept+"' and GAKUNEN = '"+gakunen+"' and SEMESTER = '"+semester+"' and WEEK = '"+week+"' and HOUR = '"+hour+"' and SUBJECT_CODE = '"+subjectCode+"' and CLASS_CODE = '"+classCode+"'";
	}
	int res = jdbc.executeKyomuUpdate(update); 
	if (res <= 0) {
	  cout.println("0");
	} else {
	  cout.println("1");
	}		
	return; 
      } catch (Exception e) {
	cout.println("ERROR$ " + e.toString().trim());
      }
    } else if (commandCode.equals("SETROOMTOSUBJECTCLASSOFJIKANWARI2")) {
      // SCHOOL_YEAR|FACULTY|SEMESTER|GAKUNEN|WEEK|SUBJECT_CODE|CLASS_CODE|DEPARTMENT|HOUR|ROOM| 
      try {
	String[] tokens = paramValues.split("\\|");
	String schoolYear  = tokens[0].trim();
	String faculty     = tokens[1].trim();
	String semester    = tokens[2].trim();
	String gakunen     = tokens[3].trim();
	String week        = tokens[4].trim();
	String subjectCode = tokens[5].trim();
	String classCode   = tokens[6].trim();
	String dept        = tokens[7].trim();
	String hour        = tokens[8].trim();
	String room        = tokens[9].trim();
	String update;
	if (room.trim().equals("")) {
	  update = "update JIKANWARI set ROOM = null where SCHOOL_YEAR = '"+schoolYear+"' and FACULTY = '"+faculty+"' and SEMESTER = '"+semester+"' and WEEK = '"+week+"' and SUBJECT_CODE = '"+subjectCode+"' and CLASS_CODE = '"+classCode+"'";
	} else {	
	  update = "update JIKANWARI set ROOM = '"+room+"' where SCHOOL_YEAR = '"+schoolYear+"' and FACULTY = '"+faculty+"' and SEMESTER = '"+semester+"' and WEEK = '"+week+"' and SUBJECT_CODE = '"+subjectCode+"' and CLASS_CODE = '"+classCode+"'";
	}
	int res = jdbc.executeKyomuUpdate(update); 
	if (res <= 0) {
	  cout.println("0");
	} else {
	  cout.println("1");
	}		
	return; 
      } catch (Exception e) {
	cout.println("ERROR$ " + e.toString().trim());
      }
    } else if (commandCode.equals("ADDSUBJECTCLASSTOSHUCHULIST")) {
      // SCHOOL_YEAR|FACULTY|SEMESTER|DEPARTMENT|GAKUNEN|SUBJECT_CODE|CLASS_CODE|
      try {
	String[] tokens = paramValues.split("\\|");
	String schoolYear  = tokens[0].trim();
	String faculty     = tokens[1].trim();
	String semester    = tokens[2].trim();
	String dept        = tokens[3].trim();
	String gakunen     = tokens[4].trim();
	String subjectCode = tokens[5].trim();
	String classCode   = tokens[6].trim();		
	String query = "select count(*) from MASTER2.CURRICULUM_T where ((GAKUNEN = '"+gakunen+"' and CURRICULUM_YEAR = ('"+schoolYear+"' - GAKUNEN + 1)) or (GAKUNEN = ('"+gakunen+"' - 1) and CURRICULUM_YEAR = ('"+schoolYear+"' - GAKUNEN))) and FACULTY = '"+faculty+"' and DEPARTMENT = '"+dept+"' and COURSE = '10' and SUBJECT_CODE = '"+subjectCode+"'";
	String ans = jdbc.executeKyomuQuery(query);
	StringTokenizer stk2 = new StringTokenizer(ans, "|"); 
	String cntStr  = stk2.nextToken().trim();
	if (cntStr.equals("0")) {
	  cout.println("0");
	  return;
	} else {
	  String insert = "insert into JIKANWARI (SCHOOL_YEAR, FACULTY, DEPARTMENT, GAKUNEN, SEMESTER, WEEK, HOUR, SUBJECT_CODE, CLASS_CODE, ROOM) values('"+schoolYear+"','"+faculty+"','"+dept+"','"+gakunen+"','"+semester+"','0','0','"+subjectCode+"','"+classCode+"',null)";
	  int res = jdbc.executeKyomuUpdate(insert); 
	  if (res == 0) {
	    cout.println("0");
	  } else {
	    cout.println("1");
	  }
	  return;
	}
      } catch (Exception e) {
	cout.println("ERROR$ " + e.toString().trim());
      }
    } else if (commandCode.equals("ADDSUBJECTCLASSTOGRADUATEJIKANWARI")) {
      // SCHOOL_YEAR|FACULTY|SEMESTER|GAKUNEN|WEEK|DEPARTMENT|HOUR|SUBJECT_CODE|CLASS_CODE|
      try {
	String[] tokens = paramValues.split("\\|");
	String schoolYear   = tokens[0].trim();
	String faculty      = tokens[1].trim();
	String semester     = tokens[2].trim();
	String gakunen      = tokens[3].trim();
	String week         = tokens[4].trim();
	String dept         = tokens[5].trim();
	String hour         = tokens[6].trim();
	String subjectCode  = tokens[7].trim();
	String classCode    = tokens[8].trim();		
	String query = "select count(*) from CURRICULUM where ((GAKUNEN = '"+gakunen+"' and CURRICULUM_YEAR = ('"+schoolYear+"' - GAKUNEN + 1)) or (GAKUNEN = ('"+gakunen+"' - 1) and CURRICULUM_YEAR = ('"+schoolYear+"' - GAKUNEN))) and FACULTY = '"+faculty+"' and DEPARTMENT = '"+dept+"' and COURSE != '30' and SUBJECT_CODE = '"+subjectCode+"'";
	String ans = jdbc.executeKyomuQuery(query);
	StringTokenizer stk2 = new StringTokenizer(ans, "|"); 
	String cntStr  = stk2.nextToken().trim();
	if (cntStr.equals("0")) {
	  cout.println("0");
	  return;
	} else {
	  String insert = "insert into JIKANWARI (SCHOOL_YEAR, FACULTY, DEPARTMENT, GAKUNEN, SEMESTER, WEEK, HOUR, SUBJECT_CODE, CLASS_CODE, ROOM) values('"+schoolYear+"','"+faculty+"','"+dept+"','"+gakunen+"','"+semester+"','"+week+"','"+hour+"','"+subjectCode+"','"+classCode+"',null)";
	  int res = jdbc.executeKyomuUpdate(insert); 
	  if (res == 0) {
	    cout.println("0");
	  } else {
	    cout.println("1");
	  }
	  return;
	}
      } catch (Exception e) {
	cout.println("ERROR$ " + e.toString().trim());
      }

    } else if (commandCode.equals("ADDSUBJECTCLASSTOGRADUATESHUCHULIST")) {
      // SCHOOL_YEAR|FACULTY|SEMESTER|DEPARTMENT|GAKUNEN|SUBJECT_CODE|CLASS_CODE|
      try {
	String[] tokens = paramValues.split("\\|");
	String schoolYear   = tokens[0].trim();
	String faculty      = tokens[1].trim();
	String semester     = tokens[2].trim();
	String dept         = tokens[3].trim();
	String gakunen      = tokens[4].trim();
	String subjectCode  = tokens[5].trim();
	String classCode    = tokens[6].trim();		
	String query = "select count(*) from CURRICULUM where ((GAKUNEN = '"+gakunen+"' and CURRICULUM_YEAR = ('"+schoolYear+"' - GAKUNEN + 1)) or (GAKUNEN = ('"+gakunen+"' - 1) and CURRICULUM_YEAR = ('"+schoolYear+"' - GAKUNEN))) and FACULTY = '"+faculty+"' and DEPARTMENT = '"+dept+"' and COURSE != '30' and SUBJECT_CODE = '"+subjectCode+"'";
	String ans = jdbc.executeKyomuQuery(query);
	StringTokenizer stk2 = new StringTokenizer(ans, "|"); 
	String cntStr  = stk2.nextToken().trim();
	if (cntStr.equals("0")) {
	  cout.println("0");
	  return;
	} else {
	  String insert = "insert into JIKANWARI (SCHOOL_YEAR, FACULTY, DEPARTMENT, GAKUNEN, SEMESTER, WEEK, HOUR, SUBJECT_CODE, CLASS_CODE, ROOM) values('"+schoolYear+"','"+faculty+"','"+dept+"','"+gakunen+"','"+semester+"','0','0','"+subjectCode+"','"+classCode+"',null)";
	  int res = jdbc.executeKyomuUpdate(insert); 
	  if (res == 0) {
	    cout.println("0");
	  } else {
	    cout.println("1");
	  }
	  return;
	}
      } catch (Exception e) {
	cout.println("ERROR$ " + e.toString().trim());
      }
    } else if (commandCode.equals("DELETESUBJECTCLASSFROMSHUCHULIST")) {
      // SCHOOL_YEAR|FACULTY|SEMESTER|DEPARTMENT|GAKUNEN|SUBJECT_CODE|CLASS_CODE|
      try {
	String[] tokens = paramValues.split("\\|");
	String schoolYear   = tokens[0].trim();
	String faculty      = tokens[1].trim();
	String semester     = tokens[2].trim();
	String dept         = tokens[3].trim();
	String gakunen      = tokens[4].trim();
	String subjectCode  = tokens[5].trim();
	String classCode    = tokens[6].trim();			
	String delete = "delete from JIKANWARI where SCHOOL_YEAR = '"+schoolYear+"' and FACULTY = '"+faculty+"' and DEPARTMENT = '"+dept+"' and GAKUNEN = '"+gakunen+"' and SEMESTER = '"+semester+"' and WEEK = '0' and SUBJECT_CODE = '"+subjectCode+"' and CLASS_CODE = '"+classCode+"'";
	int res = jdbc.executeKyomuUpdate(delete); 
	if (res <= 0) {
	  cout.println("0");
	} else {
	  cout.println("1");
	}		
	return; 
      } catch (Exception e) {
	cout.println("ERROR$ " + e.toString().trim());
      }
    } else if (commandCode.equals("DELETESUBJECTCLASSFROMSHUCHULIST2")) {
      // SCHOOL_YEAR|FACULTY|SEMESTER|DEPARTMENT|GAKUNEN|SUBJECT_CODE|CLASS_CODE|
      try {
	String[] tokens = paramValues.split("\\|");
	String schoolYear   = tokens[0].trim();
	String faculty      = tokens[1].trim();
	String semester     = tokens[2].trim();
	String dept         = tokens[3].trim();
	String gakunen      = tokens[4].trim();
	String subjectCode  = tokens[5].trim();
	String classCode    = tokens[6].trim();			
	String delete = "delete from JIKANWARI where SCHOOL_YEAR = '"+schoolYear+"' and FACULTY = '"+faculty+"' and SEMESTER = '"+semester+"' and WEEK = '0' and SUBJECT_CODE = '"+subjectCode+"' and CLASS_CODE = '"+classCode+"'";
	int res = jdbc.executeKyomuUpdate(delete); 
	if (res <= 0) {
	  cout.println("0");
	} else {
	  cout.println("1");
	}		
	return; 
      } catch (Exception e) {
	cout.println("ERROR$ " + e.toString().trim());
      }
    } else {
      System.out.println("jikanwariControl: " + commandCode + " : " + paramValues); //
      cout.println("0");
    }
  }

  protected void copyCurriculumToNextYear(String paramValues) {
    try {
      String[] tokens = paramValues.split("\\|");
      String fromYear  = tokens[0].trim();
      String toYear    = tokens[1].trim();

      String del = "delete from MASTER.CURRICULUM where CURRICULUM_YEAR = "+toYear;
      String ins = "insert into MASTER.CURRICULUM ( CURRICULUM_YEAR, FACULTY, DEPARTMENT, COURSE, GAKUNEN, SUBJECT_CODE, KUBUN_CODE, REQ_CODE, UNIT) select '"+toYear+"', FACULTY, DEPARTMENT, COURSE, GAKUNEN, SUBJECT_CODE, KUBUN_CODE, REQ_CODE, UNIT from MASTER.CURRICULUM where CURRICULUM_YEAR = '"+fromYear+"'";

      int res = jdbc.executeKyomuUpdate(del); 
      int res2 = jdbc.executeKyomuUpdate(ins); 
      cout.println("ok");
    } catch (Exception e) {
      cout.println("ERROR： " + e.toString().trim());
    }
  }

  protected void copyEduCurriculumToNextYear(String paramValues) {
    try {
      String[] tokens = paramValues.split("\\|");
      String fromYear  = tokens[0].trim();
      String toYear    = tokens[1].trim();

      String del = "delete from MASTER.ED_CURRICULUM where CURRICULUM_YEAR = "+toYear;
      String ins = "insert into MASTER.ED_CURRICULUM ( CURRICULUM_YEAR, FACULTY, DEPARTMENT, GAKUNEN, SUBJECT_CODE, KUBUN_CODE, KUBUN_CODE_2, REQ_CODE, UNIT ) select '"+toYear+"', FACULTY, DEPARTMENT, GAKUNEN, SUBJECT_CODE, KUBUN_CODE, KUBUN_CODE_2, REQ_CODE, UNIT from MASTER.ED_CURRICULUM where CURRICULUM_YEAR = '"+fromYear+"'";

      int res = jdbc.executeKyomuUpdate(del); 
      int res2 = jdbc.executeKyomuUpdate(ins); 

      cout.println("ok");
    } catch (Exception e) {
      cout.println("ERROR： " + e.toString().trim());
    }
  }
  
  protected void copyClassInfoToNextYear(String paramValues) {
    try {
      String[] tokens = paramValues.split("\\|");
      String fromYear  = tokens[0].trim();
      String toYear    = tokens[1].trim();

      String del = "delete from MASTER.CLASS_INFO where SCHOOL_YEAR = "+toYear;
      String ins = "insert into MASTER.CLASS_INFO ( SCHOOL_YEAR, SUBJECT_CODE, CLASS_CODE, TEACHER_CODE, CO_TEACHERS, COMMENT_ON_CLASS) select '"+toYear+"', SUBJECT_CODE, CLASS_CODE, TEACHER_CODE, CO_TEACHERS, COMMENT_ON_CLASS from MASTER.CLASS_INFO where SCHOOL_YEAR = '"+fromYear+"'";
      
      int res = jdbc.executeKyomuUpdate(del); 
      int res2 = jdbc.executeKyomuUpdate(ins); 

      cout.println("ok");
    } catch (Exception e) {
      cout.println("ERROR： " + e.toString().trim());
    }
  }

  protected void copyJikanwariToNextYear(String paramValues) {
    try {
      String[] tokens = paramValues.split("\\|");
      String fromYear  = tokens[0].trim();
      String toYear    = tokens[1].trim();

      String del = "delete from MASTER.JIKANWARI where SCHOOL_YEAR = "+toYear;
      String ins1 = "insert into MASTER.JIKANWARI ( SCHOOL_YEAR, FACULTY, DEPARTMENT, GAKUNEN, SEMESTER, WEEK, HOUR, SUBJECT_CODE, CLASS_CODE, ROOM) select distinct '"+toYear+"', J.FACULTY, J.DEPARTMENT, J.GAKUNEN, J.SEMESTER, J.WEEK, J.HOUR, J.SUBJECT_CODE, J.CLASS_CODE, J.ROOM from MASTER.JIKANWARI J, MASTER.CURRICULUM C where J.FACULTY = '11' and J.SCHOOL_YEAR = '"+fromYear+"' and C.COURSE = '10' and (C.CURRICULUM_YEAR+2) >= "+fromYear+" and (C.CURRICULUM_YEAR + C.GAKUNEN - 2) = J.SCHOOL_YEAR and J.FACULTY = C.FACULTY and ((J.DEPARTMENT = C.DEPARTMENT) or (C.DEPARTMENT = '30')) and J.GAKUNEN = C.GAKUNEN and J.SUBJECT_CODE = C.SUBJECT_CODE";
      String ins2 = "insert into MASTER.JIKANWARI ( SCHOOL_YEAR, FACULTY, DEPARTMENT, GAKUNEN, SEMESTER, WEEK, HOUR, SUBJECT_CODE, CLASS_CODE, ROOM) select distinct '"+toYear+"', J.FACULTY, J.DEPARTMENT, J.GAKUNEN, J.SEMESTER, J.WEEK, J.HOUR, J.SUBJECT_CODE, J.CLASS_CODE, J.ROOM from MASTER.JIKANWARI J where J.FACULTY = '32' and J.SCHOOL_YEAR = '"+fromYear+"'";
      
      int res = jdbc.executeKyomuUpdate(del); 
      int res2 = jdbc.executeKyomuUpdate(ins1);  
      int res3 = jdbc.executeKyomuUpdate(ins2); 
      cout.println("ok");
    } catch (Exception e) {
      cout.println("ERROR： " + e.toString().trim());
    }
  }
  
  protected void copyJikanwariOverlapToNextYear(String paramValues) {
    try {
      String[] tokens = paramValues.split("\\|");
      String fromYear  = tokens[0].trim();
      String toYear    = tokens[1].trim();
      int SCHOOL_YEAR = Integer.parseInt(toYear);

      String del = "delete from MASTER.JIKAN_OVERLAP where SCHOOL_YEAR = "+toYear;
      String ins = "insert into MASTER.JIKAN_OVERLAP (SCHOOL_YEAR, SEMESTER, SUBJECT_CODE, CLASS_CODE) select '"+toYear+"', SEMESTER, SUBJECT_CODE, CLASS_CODE from MASTER.JIKAN_OVERLAP where SCHOOL_YEAR = '"+fromYear+"'";
      int res = jdbc.executeKyomuUpdate(del); 
      int res2 = jdbc.executeKyomuUpdate(ins); 
      cout.println("ok");
    } catch (Exception e) {
      cout.println("ERROR： " + e.toString().trim());
    }
  }

  protected void copyYomikaeToNextYear(String paramValues) {
    String query;
    String ans, del, ins;
    HashSet<String> jikanwariSet = new HashSet<String>();
    HashMap<String, String> yomikaeMap = new HashMap<String, String>();
    HashSet<String> studentYearDeptSet = new HashSet<String>();
    HashSet<String> studentYearSet = new HashSet<String>();

    try {
      String[] ttokens = paramValues.split("\\|");
      String fromYear  = ttokens[0].trim();
      String toYear    = ttokens[1].trim();
      int SCHOOL_YEAR = Integer.parseInt(toYear);

      del = "delete from MASTER.CURR_YOMIKAE where SCHOOL_YEAR = "+toYear;
      int res = jdbc.executeKyomuUpdate(del); 

      query = "select CURRICULUM_YEAR, DEPARTMENT, COURSE, SUBJECT_CODE, YOMIKAE_CODE from MASTER.CURR_YOMIKAE where FACULTY = '11' and SCHOOL_YEAR = '" + fromYear + "'";
      ans = jdbc.executeKyomuQuery(query);
      if (ans != null) {
	String[] lines = ans.split("\\$");
	for (String line : lines) {
	  String[] tokens = line.split("\\|");
	  String cyear       = tokens[0];
	  String department  = tokens[1];
	  String course      = tokens[2];
	  String subjectCode = tokens[3];
	  String yomikaeCode = tokens[4];
	  if (!yomikaeCode.equals(" ")) {
	    String key = cyear + "|" + department + "|" + course + "|" + subjectCode; 
	    yomikaeMap.put(key, yomikaeCode);
	  }
	}
      }

      query = "select distinct DEPARTMENT, SUBJECT_CODE from MASTER2.JIKANWARI_INFO where SCHOOL_YEAR = '" + toYear + "' and FACULTY = '11'";
      ans = jdbc.executeKyomuQuery(query);
      if (ans != null) {
	String[] lines = ans.split("\\$");
	for (String line : lines) {
	  String[] tokens = line.split("\\|");
	  String department  = tokens[0];
	  String subjectCode = tokens[1];
	  String key = department + "|" + subjectCode; 
	  jikanwariSet.add(key);
	}
      }

      query = "select distinct CURRICULUM_YEAR, DEPARTMENT, COURSE from MASTER.STUDENT where FACULTY = '11' and STUDENT_STATUS <= 4";
      ans = jdbc.executeKyomuQuery(query);
      if (ans != null) {
	String[] lines = ans.split("\\$");
	for (String line : lines) {
	  String[] tokens = line.split("\\|");
	  String cyear      = tokens[0];
	  String department = tokens[1];
	  String course     = tokens[2];
	  String key = cyear + "|" + department + "|" + course; 
	  studentYearDeptSet.add(key);
	}
      }
	
      query = "select distinct CURRICULUM_YEAR from MASTER.STUDENT where FACULTY = '11' and STUDENT_STATUS <= 4 and COURSE = '10'";
      ans = jdbc.executeKyomuQuery(query);
      if (ans != null) {
	String[] lines = ans.split("\\$");
	for (String line : lines) {
	  String[] tokens = line.split("\\|");
	  String cyear = tokens[0];
	  studentYearSet.add(cyear);
	}
      }
      
      // 自然科学、情報科目、対象分野で０単位でない科目の読み替え表を作成

      for (String elem : studentYearDeptSet) {
	String[] tttokens = elem.split("\\|");
	String CURRICULUM_YEAR = tttokens[0];
	String DEPARTMENT      = tttokens[1];
	String COURSE          = tttokens[2];
	int cyear = Integer.parseInt(CURRICULUM_YEAR);
	int gakunen;
	if (COURSE.equals("10")) {
	  gakunen = SCHOOL_YEAR - cyear + 1;
	} else {
	  gakunen = SCHOOL_YEAR - cyear + 3;
	}
	query = "select distinct SUBJECT_CODE, GAKUNEN, KUBUN_CODE, REQ_CODE, UNIT from MASTER2.CURRICULUM_T where CURRICULUM_YEAR = '" + CURRICULUM_YEAR + "' and FACULTY = '11' and DEPARTMENT = '" + DEPARTMENT + "' and COURSE = '" + COURSE + "' and GAKUNEN <= " + gakunen + " and UNIT != 0 and ((KUBUN_CODE = '635') or (KUBUN_CODE = '636') or (KUBUN_CODE = '637'))";	
	ans = jdbc.executeKyomuQuery(query);
	if (ans != null) {
	  String[] lines = ans.split("\\$");
	  for (String line : lines) {
	    String[] tokens = line.split("\\|");
	    String SUBJECT_CODE = tokens[0];
	    String GAKUNEN      = tokens[1];
	    String KUBUN        = tokens[2];
	    String REQ          = tokens[3];
	    String UNIT         = tokens[4];	    
	    String key  = DEPARTMENT + "|" + SUBJECT_CODE;	    
	    if (!jikanwariSet.contains(key)) {	
	      String key2 = CURRICULUM_YEAR + "|" + DEPARTMENT + "|" + COURSE + "|" + SUBJECT_CODE; 
	      if (yomikaeMap.containsKey(key2)) {
		String YOMIKAE_CODE = yomikaeMap.get(key2);     
		ins = "insert into MASTER.CURR_YOMIKAE (SCHOOL_YEAR, CURRICULUM_YEAR, FACULTY, DEPARTMENT, COURSE, GAKUNEN, KUBUN_CODE, REQ_CODE, UNIT, SUBJECT_CODE, YOMIKAE_CODE) values ('"+toYear+"','"+CURRICULUM_YEAR+"','11','"+DEPARTMENT+"','"+COURSE+"','"+GAKUNEN+"','"+KUBUN+"','"+REQ+"',"+UNIT+",'"+SUBJECT_CODE+"','"+YOMIKAE_CODE+"')";
	      } else {  
		ins = "insert into MASTER.CURR_YOMIKAE (SCHOOL_YEAR, CURRICULUM_YEAR, FACULTY, DEPARTMENT, COURSE, GAKUNEN, KUBUN_CODE, REQ_CODE, UNIT, SUBJECT_CODE, YOMIKAE_CODE) values ('"+toYear+"','"+CURRICULUM_YEAR+"','11','"+DEPARTMENT+"','"+COURSE+"','"+GAKUNEN+"','"+KUBUN+"','"+REQ+"',"+UNIT+",'"+SUBJECT_CODE+"', null)";
	      }
	      int res2 = jdbc.executeKyomuUpdate(ins); 
	    }
	  }
	}
      }
	      
      // 自然科学、情報科目、対象分野以外の必修科目の読み替え表を作成

      for (String CURRICULUM_YEAR : studentYearSet) {
	int cyear = Integer.parseInt(CURRICULUM_YEAR);
	int gakunen = SCHOOL_YEAR - cyear + 1;
	query = "select distinct SUBJECT_CODE, GAKUNEN, KUBUN_CODE, REQ_CODE, UNIT from MASTER2.CURRICULUM_T where CURRICULUM_YEAR = '" + CURRICULUM_YEAR + "' and FACULTY = '11' and DEPARTMENT = '31' and COURSE = '10' and GAKUNEN <= " + gakunen + " and REQ_CODE = '1' and ((KUBUN_CODE = '440') or (KUBUN_CODE = '444') or (KUBUN_CODE = '445') or (KUBUN_CODE = '446') or (KUBUN_CODE = '447') or (KUBUN_CODE = '448') or (KUBUN_CODE = '991') or (KUBUN_CODE = '992'))";	
	ans = jdbc.executeKyomuQuery(query);
	if (ans != null) {
	  String[] lines = ans.split("\\$");
	  for (String line : lines) {
	    String[] tokens = line.split("\\|");
	    String SUBJECT_CODE = tokens[0];
	    String GAKUNEN      = tokens[1];
	    String KUBUN        = tokens[2];
	    String REQ          = tokens[3];
	    String UNIT         = tokens[4];	    
	    String key  = "31|" + SUBJECT_CODE;	     
	    if (!jikanwariSet.contains(key)) {	
	      String key2 = CURRICULUM_YEAR + "|30|10|" + SUBJECT_CODE; 
	      if (yomikaeMap.containsKey(key2)) {
		String YOMIKAE_CODE = yomikaeMap.get(key2);     
		ins = "insert into MASTER.CURR_YOMIKAE (SCHOOL_YEAR, CURRICULUM_YEAR, FACULTY, DEPARTMENT, COURSE, GAKUNEN, KUBUN_CODE, REQ_CODE, UNIT, SUBJECT_CODE, YOMIKAE_CODE) values ('"+toYear+"','"+CURRICULUM_YEAR+"','11','30','10','"+GAKUNEN+"','"+KUBUN+"','"+REQ+"',"+UNIT+",'"+SUBJECT_CODE+"','"+YOMIKAE_CODE+"')";
	      } else {  
		ins = "insert into MASTER.CURR_YOMIKAE (SCHOOL_YEAR, CURRICULUM_YEAR, FACULTY, DEPARTMENT, COURSE, GAKUNEN, KUBUN_CODE, REQ_CODE, UNIT, SUBJECT_CODE, YOMIKAE_CODE) values ('"+toYear+"','"+CURRICULUM_YEAR+"','11','30','10','"+GAKUNEN+"','"+KUBUN+"','"+REQ+"',"+UNIT+",'"+SUBJECT_CODE+"', null)";
	      }
	      int res2 = jdbc.executeKyomuUpdate(ins); 
	    }
	  }
	}
      }
      cout.println("ok");
    } catch (Exception e) {
      cout.println("ERROR： " + e.toString().trim());
    }
  }
  
  protected void copyGradYokenToNextYear(String paramValues) {
    try {
      String[] tokens = paramValues.split("\\|");
      String fromYear  = tokens[0].trim();
      String toYear    = tokens[1].trim();

      String del1 = "delete from MASTER.YOKEN_TABLE where CURRICULUM_YEAR = "+toYear;
      String del2 = "delete from MASTER.YOKEN_SUBJECT_LIST_ORIG where CURRICULUM_YEAR = "+toYear;
      String ins1 = "insert into MASTER.YOKEN_TABLE ( CURRICULUM_YEAR, FACULTY, DEPARTMENT, COURSE, YOKEN_TYPE, YOKEN_CODE, LOWER_BOUND, UPPER_BOUND ) select '"+toYear+"', FACULTY, DEPARTMENT, COURSE, YOKEN_TYPE, YOKEN_CODE, LOWER_BOUND, UPPER_BOUND from MASTER.YOKEN_TABLE where CURRICULUM_YEAR = '"+fromYear+"'";
      String ins2 = "insert into MASTER.YOKEN_SUBJECT_LIST_ORIG ( CURRICULUM_YEAR, FACULTY, DEPARTMENT, COURSE, YOKEN_CODE, SUBJECT_CODE ) select '"+toYear+"', FACULTY, DEPARTMENT, COURSE, YOKEN_CODE, SUBJECT_CODE from MASTER.YOKEN_SUBJECT_LIST_ORIG where CURRICULUM_YEAR = '"+fromYear+"'";
      
      int res = jdbc.executeKyomuUpdate(del1); 
      int res2 = jdbc.executeKyomuUpdate(del2); 
      int res3 = jdbc.executeKyomuUpdate(ins1); 
      int res4 = jdbc.executeKyomuUpdate(ins2); 
      cout.println("ok");
    } catch (Exception e) {
      cout.println("ERROR： " + e.toString().trim());
    }
  }
  
  protected void copyEduYokenToNextYear(String paramValues) {
    try {
      String[] tokens = paramValues.split("\\|");
      String fromYear  = tokens[0].trim();
      String toYear    = tokens[1].trim();

      String del = "delete from MASTER.ED_YOKEN_TABLE where CURRICULUM_YEAR = "+toYear;
      String ins = "insert into MASTER.ED_YOKEN_TABLE ( CURRICULUM_YEAR, FACULTY, DEPARTMENT, YOKEN_TYPE, YOKEN_CODE, LOWER_BOUND ) select '"+toYear+"', FACULTY, DEPARTMENT, YOKEN_TYPE, YOKEN_CODE, LOWER_BOUND from MASTER.ED_YOKEN_TABLE where CURRICULUM_YEAR = '"+fromYear+"'";
      
      int res = jdbc.executeKyomuUpdate(del); 
      int res2 = jdbc.executeKyomuUpdate(ins); 
      cout.println("ok");
    } catch (Exception e) {
      cout.println("ERROR： " + e.toString().trim());
    }
  }
    
  protected void copyModuleDefToNextYear(String paramValues) {
    try {
      String[] tokens = paramValues.split("\\|");
      String fromYear  = tokens[0].trim();
      String toYear    = tokens[1].trim();

      String del1 = "delete from MASTER.MODULE_DEF where CURRICULUM_YEAR = "+toYear;
      String del2 = "delete from MASTER.MODULE_SUBJECT_LIST where CURRICULUM_YEAR = "+toYear;
      String del3 = "delete from MASTER.MODULE_COURSE_DEF where CURRICULUM_YEAR = "+toYear;
      String ins1 = "insert into MASTER.MODULE_DEF ( CURRICULUM_YEAR, MODULE_CODE, LOWER_BOUND ) select '"+toYear+"', MODULE_CODE, LOWER_BOUND from MASTER.MODULE_DEF where CURRICULUM_YEAR = '"+fromYear+"'";
      String ins2 = "insert into MASTER.MODULE_SUBJECT_LIST ( CURRICULUM_YEAR, MODULE_CODE, SUBJECT_CODE ) select '"+toYear+"', MODULE_CODE, SUBJECT_CODE from MASTER.MODULE_SUBJECT_LIST where CURRICULUM_YEAR = '"+fromYear+"'";
      String ins3 = "insert into MASTER.MODULE_COURSE_DEF ( CURRICULUM_YEAR, MODULE_COURSE_CODE, MODULE_CODE ) select '"+toYear+"', MODULE_COURSE_CODE, MODULE_CODE from MASTER.MODULE_COURSE_DEF where CURRICULUM_YEAR = '"+fromYear+"'";
      
      int res = jdbc.executeAttendUpdate(del1); 
      int res2= jdbc.executeAttendUpdate(del2); 
      int res3= jdbc.executeAttendUpdate(del3); 
      int res4 = jdbc.executeAttendUpdate(ins1); 
      int res5 = jdbc.executeAttendUpdate(ins2); 
      int res6 = jdbc.executeAttendUpdate(ins3); 
      cout.println("ok");
    } catch (Exception e) {
      cout.println("ERROR： " + e.toString().trim());
    }
  }
  

  protected void copyIIFCurriculumToNextYear(String paramValues) {
    try {
      String[] tokens = paramValues.split("\\|");
      String fromYear  = tokens[0].trim();
      String toYear    = tokens[1].trim();

      String del = "delete from MASTER.IIF_CURRICULUM where CURRICULUM_YEAR = "+toYear;
      String ins = "insert into MASTER.IIF_CURRICULUM ( CURRICULUM_YEAR, FACULTY, DEPARTMENT, GAKUNEN, SUBJECT_CODE, KUBUN_CODE, REQ_CODE, UNIT ) select '"+toYear+"', FACULTY, DEPARTMENT, GAKUNEN, SUBJECT_CODE, KUBUN_CODE, REQ_CODE, UNIT from MASTER.IIF_CURRICULUM where CURRICULUM_YEAR = '"+fromYear+"'";
      
      int res = jdbc.executeKyomuUpdate(del); 
      int res2 = jdbc.executeKyomuUpdate(ins); 
      cout.println("ok");
    } catch (Exception e) {
      cout.println("ERROR： " + e.toString().trim());
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
      String insert = "insert into MASTER.MASTER_LOG (STAFF_ID, STAFF_CODE, QUALIFICATION, TOOL_NAME, LOGIN_DATE, LOGIN_RESULT, INET_ADDRESS) values ('" + userID + "','" + STAFF_CODE  + "','" + QUALIFICATION + "','MasterTool', sysdate, '" + res + "','" + inetAddress + "')";
      jdbc.executeKyomuUpdate(insert);     
    } catch (Exception e) {
      e.printStackTrace();
    }
  } 
}
