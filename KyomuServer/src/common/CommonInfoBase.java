package common;
import java.util.*;
import java.net.*;
import java.sql.*;

public class CommonInfoBase { 
  public String serverHost; 
  public int serverPort;

  public String kyomuDBURL = "jdbc:oracle:thin:@131.206.102.77:1521:orc1";
  public String kyomuDBSchema;
  public String kyomuDBPasswd;
  
  public String attendDBURL = "jdbc:oracle:thin:@131.206.103.230:1521:orc1";
  public String attendDBSchema;
  public String attendDBPasswd;

  public JDBCConnectionBase jdbc;

  public String SecretKillerCode;
  public String SecretKillerPassword;

  public static String studentPhotoDir = "/home/maginu/KYOMU-INFO/STUDENT-PHOTO/";

  public String keyTrustURL = "http://kaoru.gakumu.jimui.kyutech.ac.jp/kyomu/nextGenerationClientTrust";
  public String storePasswd = "(»ÎÃ©æ Û)";
  public String gakumuAddress = "jho-gakumu@jimu.kyutech.ac.jp";
  
  public HashMap<String, Integer> userMap = new HashMap<String, Integer>();
  public HashMap<String, String> loginMap = new HashMap<String, String>();
  public Crypt cryptObj;
  public PasswordGenerator passwordGenerator;
  
  public CommonInfoBase() { 
    cryptObj = new Crypt();
    passwordGenerator = new PasswordGenerator();
  }

  public String crypt(String passwd, String salt) {
    return cryptObj.crypt(passwd, salt);
  }

  public String getInitialPassword() {
    return passwordGenerator.getPassword();
  }

  public String getSalt() {
    return passwordGenerator.getSalt();
  }
}
