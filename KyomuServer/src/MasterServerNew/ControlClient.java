package MasterServerNew;
import java.awt.*;
import java.net.*;
import java.io.*;
import java.util.*;
import javax.net.ssl.SSLSocketFactory;

public class ControlClient {

  private CommonInfo commonInfo;
  private BufferedReader cin;
  private PrintWriter cout;

  public ControlClient() {   
    commonInfo = new CommonInfo();
    try {
      URL conn = new URL(commonInfo.keyTrustURL);
      InputStream is = conn.openStream();
      File tempfile = File.createTempFile("keyTrust", ".key");
      tempfile.deleteOnExit();    
      OutputStream os = new FileOutputStream(tempfile);
      int ch;
      while ((ch = is.read()) != -1) {
	os.write(ch);
      }
      is.close();
      os.close(); 
      
      String keyTrustFile = tempfile.getPath();
      System.setProperty("javax.net.ssl.trustStore", keyTrustFile);
      System.setProperty("javax.net.ssl.trustStorePassword", commonInfo.storePasswd);
      Socket socket = SSLSocketFactory.getDefault().createSocket(commonInfo.serverHost, commonInfo.serverPort);	
      
      InputStream istream = socket.getInputStream();
      OutputStream ostream = socket.getOutputStream();
      cin = new BufferedReader(new InputStreamReader(istream));
      cout = new PrintWriter(ostream, true);
    } catch (Exception e) {
      System.out.println("MasterServer に対する ControlClient が暗号通信を設定できません。");
      System.exit(0);
    }
    
    String id = commonInfo.SecretKillerCode;
    String password = commonInfo.SecretKillerPassword;
    String message = "SERVER_CONTROLLER|ControlClient|MASTER_SERVER:" + id + "|" + password;
    try {
      cout.println(message);
      String answer = cin.readLine();
      if (!answer.equals("accept")) {
	System.exit(0);
      }
    } catch (Exception e) {
      e.printStackTrace();
    } 
  }
  
  public void loop() {
    BufferedReader keyin;
    try {
      keyin = new BufferedReader(new InputStreamReader(System.in));
      
      while (true) {
	System.out.println("");
	System.out.println(" 1: LOGIN リスト  ");
	System.out.println(" 2: 稼働中の Thread 数  ");
	System.out.println(" 9: Control 終了  ");
	System.out.println("");
	System.out.print(" code = ");
      
	String str = keyin.readLine().trim();
	if (str.equals("1")) {
	    printLoginList();
	} else if (str.equals("2")) {
	    printThreadCount();
	} else if (str.equals("9")) {
	    break;
	}
      }
    } catch (Exception e) {
      e.printStackTrace();
    }
  }
    
  public void printLoginList() {
    String message = "PRINT_LOGIN_LIST";
    String line;
    try {
      cout.println(message);
      String answer = cin.readLine();
      StringTokenizer stk = new StringTokenizer(answer, "$");
      int count = 0;
      while (stk.hasMoreTokens()) {
	  line = stk.nextToken();
	  System.out.println(line);
	  count++;
      }
      System.out.println("loginCount = " + count);
    } catch (Exception e) {
      e.printStackTrace();
    } 
  }
   
  public void printThreadCount() {
    String message = "PRINT_THREAD_COUNT";
    try {
      cout.println(message);
      String answer = cin.readLine();
      System.out.println(answer);
    } catch (Exception e) {
      e.printStackTrace();
    } 
  }
               
  public static void main(String args[]){
    ControlClient tcl = new ControlClient();
    tcl.loop();
  }
}

