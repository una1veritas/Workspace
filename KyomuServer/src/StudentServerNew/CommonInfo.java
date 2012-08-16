package StudentServerNew;
import common.*;
import java.util.*;
import java.net.*;
import java.sql.*;
import syllabusNew.*;

public class CommonInfo extends CommonInfoBase  {  
  public SyllabusControl syllabus;
    
  public CommonInfo() { 
    serverHost = "131.206.103.7";
    serverPort = 3402;
    
    kyomuDBSchema = "STUDENT";
    kyomuDBPasswd = "(»ÎÃ©æ Û)";
    
    attendDBSchema = "STUDENT";
    attendDBPasswd = "(»ÎÃ©æ Û)";
    
    SecretKillerCode =  "KenjiroMaginuKiller";
    SecretKillerPassword = "(»ÎÃ©æ Û)";

    try {
      jdbc = new JDBCConnection(this);
      int thisSchoolYear = jdbc.thisSchoolYear;
      syllabus = new SyllabusControl(thisSchoolYear);
    } catch (Exception e) {
      System.out.println(e);
    }
  }

  public void addLoginMap(Person me) {
    String key = me.STUDENT_CODE;
    String val = me.STUDENT_CODE+"|"+me.STUDENT_NAME+"|"+me.STUDENT_DEPARTMENT+"|"+me.STUDENT_GAKUNEN+"|"+me.inetAddress;
    loginMap.put(key, val);
  }

  public void removeLoginMap(Person me) {
    String key = me.STUDENT_CODE;
    loginMap.remove(key);
  }  
}
