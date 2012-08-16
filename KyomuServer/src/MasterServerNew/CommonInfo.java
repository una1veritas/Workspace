package MasterServerNew;
import common.*;
import java.util.*;
import java.net.*;
import java.sql.*;

public class CommonInfo extends CommonInfoBase  {  
    
  public CommonInfo() { 
    serverHost = "131.206.103.7";
    serverPort = 3403;
    
    kyomuDBSchema = "MASTER";
    kyomuDBPasswd = "(»ÎÃ©æ Û)";
    
    attendDBSchema = "MASTER";
    attendDBPasswd = "(»ÎÃ©æ Û)";
    
    SecretKillerCode =  "KenjiroMaginuKiller";
    SecretKillerPassword = "(»ÎÃ©æ Û)";

    try {
      jdbc = new JDBCConnection(this);
    } catch (Exception e) {
      System.out.println(e);
    }
  }

  public void addLoginMap(Person me) {
    String key = me.USER_ID;
    String val = me.STAFF_CODE+"|"+me.STAFF_NAME+"|"+me.QUALIFICATION+"|"+me.inetAddress;
    loginMap.put(key, val);
  }

  public void removeLoginMap(Person me) {
    String key = me.USER_ID;
    loginMap.remove(key);
  }  
}
