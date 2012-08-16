package TeacherServerNew;
import java.awt.*;
import java.net.*;
import java.io.*;
import java.util.*;
import javax.net.ssl.SSLSocketFactory;

public class KillerClient {
  private CommonInfo commonInfo;
  private BufferedReader cin;
  private PrintWriter cout;
  
  public KillerClient() {    
    commonInfo = new CommonInfo();
    
    try {
      Process process = Runtime.getRuntime().exec( "/bin/ps ux" );
      InputStream i = process.getInputStream();
      InputStreamReader r = new InputStreamReader(i);
      BufferedReader in = new BufferedReader(r);
      ArrayList<String> lines = new ArrayList<String>();
      String line;
      while ((line = in.readLine()) != null) {
	lines.add(line);
      }
      process.waitFor();
      process.destroy();
      for (String s : lines) {
	if ((s.indexOf("TeacherServerNew/TeacherServer")) >= 0) {	  
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
	  
	  String id = commonInfo.SecretKillerCode;
	  String password = commonInfo.SecretKillerPassword;
	  String message = "SERVER_KILLER|KillerClient|TEACHER_SERVER:" + id + "|" + password;
	  cout.println(message);
	  String answer = cin.readLine();
	}
      }
    } catch (Exception e) {
      e.printStackTrace();
    }	
  }
  
  public static void main(String args[]){
    KillerClient tcl = new KillerClient();
  }
}
