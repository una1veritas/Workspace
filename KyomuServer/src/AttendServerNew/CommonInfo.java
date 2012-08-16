package AttendServerNew;

public class CommonInfo {     
  public String keyTrustURL = "http://kaoru.gakumu.jimui.kyutech.ac.jp/kyomu/nextGenerationClientTrust";
  public String storePasswd = "NextGenerationKyomuInfo";
  public String serverHost = "131.206.103.7";
  public int serverPort = 3289; 
  public JDBCConnection conn;

  public CommonInfo() {           
    try {
      conn = new JDBCConnection();
    } catch (Exception e) {
      System.out.println(e);
    }
  }
}
