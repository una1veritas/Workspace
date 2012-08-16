//import java.util.*;
import java.io.*;
//import java.awt.*;
//import java.awt.event.*;
import java.net.*;
//import java.io.*;
import javax.net.ssl.SSLSocketFactory;

public class ServerConnection { 

  private String serverHost = "131.206.103.7";
  private int serverPort = 3289;
  
  private String keyTrustURL = "http://kaoru.gakumu.jimui.kyutech.ac.jp/kyomu/nextGenerationClientTrust";
  private String storePasswd = "NextGenerationKyomuInfo";
  
  private Socket socket;
  private InputStream istream;
  private OutputStream ostream;    
  private BufferedReader cin;
  private PrintWriter cout;
  private PrintWriter fout;
  
  public ServerConnection(PrintWriter fout) { 
    this.fout = fout;
    
    try {
      URL conn = new URL(keyTrustURL);
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
      System.setProperty("javax.net.ssl.trustStorePassword", storePasswd);
      socket = SSLSocketFactory.getDefault().createSocket(serverHost, serverPort);
      
      istream = socket.getInputStream();
      ostream = socket.getOutputStream();
      cin = new BufferedReader(new InputStreamReader(istream, "EUC-JP"));
      cout = new PrintWriter(new OutputStreamWriter(ostream, "EUC-JP"), true);
    } catch (Exception e) {
      System.out.println("CollectCardData: " + e.toString());
      System.exit(0);
    }
  }
  
  public void close() throws IOException {
    socket.close();
  }
  
  public int insertError(String err, String id) {
    try {
      String msg = "10010KenjiroMaginu:" + err + "|" + id;
      cout.println(msg);       
      String answer = cin.readLine();
      return Integer.parseInt(answer);
    } catch (Exception e) {
      return 0;
    }
  }
  
  
  public String query(String query) {	
    String msg = "10006KenjiroMaginu:" + query;
    try { 
      cout.println(msg);     
      String answer = cin.readLine();
      if (answer.equals("null")) {
	return null;
      } else if (answer.startsWith("ERROR")) {  
	fout.println("// " + answer);
	return null;      
      } else {  
	return answer;
      }
    } catch (Exception e) {
      fout.println("// " + e.toString());
      return null;
    }
  }
  
  public void checkConnectionToServer() {	
    try { 
      cout.println("10007KenjiroMaginu:Are you OK?");    
      String answer = cin.readLine();
      if (!answer.startsWith("OK")) { 
	fout.println("// Attend Server is not alive");
	System.exit(0);
      } 
    } catch (SocketException e1) {	    
      fout.println("// " + e1.toString());
      String msg = e1.toString().trim();
      if (msg.equals("java.net.SocketException: Broken pipe")) {
	e1.printStackTrace();
	System.exit(0);
      }
    } catch (Exception e2) {	    
      fout.println("// " + e2.toString());
    }
  }
  
  public int insertAttendData(String ins) {	
    try { 
      String msg = "10008KenjiroMaginu:" + ins;
      cout.println(msg);
      String answer = cin.readLine();
      if (answer.startsWith("ERROR")) { 
	int idx = answer.indexOf("$");
	fout.println("// " + answer.substring(idx+1));
	return 0;       
      } else { 
	return Integer.parseInt(answer);
      }
    } catch (Exception e) {
      fout.println("// " + e.toString());
      return 0;
    }
  }
}
