package common;
import clients.*;
//import java.util.*;
import java.net.*;
import java.io.*;
import javax.net.ssl.SSLSocketFactory;
//import javax.swing.*;
//import javax.swing.border.*;

public class ServerConnectionBase { 
  protected CommonInfo commonInfo;
  private String serverHost;
  private int    serverPort;
  public ServerConnectionMethods serverConnectionMethods;

  /*
  protected JPasswordField oldPasswdField;
  protected JPasswordField newPasswdField1;
  protected JPasswordField newPasswdField2;
*/
//  protected InputStream istream;
//  protected OutputStream ostream;

  protected BufferedReader cin;
  protected PrintWriter cout;

  private boolean qualified = false;
  
  public ServerConnectionBase(String serverHost, 
			      int serverPort,
			      CommonInfo commonInfo) { 
    this.serverHost = serverHost;
    this.serverPort = serverPort; 
    this.commonInfo = commonInfo;

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
      Socket socket = SSLSocketFactory.getDefault().createSocket(serverHost, serverPort);
    
      InputStream istream = socket.getInputStream();
      OutputStream ostream = socket.getOutputStream();
      cin = new BufferedReader(new InputStreamReader(istream, "EUC-JP"));
      cout = new PrintWriter(new OutputStreamWriter(ostream, "EUC-JP"), true);
    } catch (Exception e) {
	  System.err.println(e.toString());
      commonInfo.showMessage("ServerConnectionBase: " + e.toString());
      System.exit(0);
    }
 //   makeQualificationComponents(); 
    serverConnectionMethods = new ServerConnectionMethods(commonInfo, 
							 // istream, ostream, 
							  cin, cout);
  } 

  protected void setQualification(boolean res) {
    qualified = res;
    serverConnectionMethods.setQualification(qualified);
  }

//  private void makeQualificationComponents() {
	  /*
    quitButton = new JButton("Quit/終了");
//    quitButton.setBackground(Color.pink);
    quitButton.addActionListener(new ActionListener() {     
      public void actionPerformed(ActionEvent e) {
        System.exit(0);
      }} );      
    idField = new JTextField();
    idField.setFont(new Font("DialogInput", Font.PLAIN, 14));
    idField.setBorder(new EmptyBorder(2, 5, 2, 2));
    passwdField = new JPasswordField();
    passwdField.setBorder(new EmptyBorder(2, 5, 2, 2));
    passwdField.setBorder( new TitledBorder( " Password " ) );
    passwdField.setEchoChar('*');
    passwdField.setFont(new Font("DialogInput", Font.PLAIN, 14));
    */
    /*
    oldPasswdField = new JPasswordField();
    oldPasswdField.setBorder( new TitledBorder( " old password " ) );
    oldPasswdField.setEchoChar('?');
    oldPasswdField.setFont(new Font("DialogInput", Font.PLAIN, 14));
    newPasswdField1 = new JPasswordField();
    newPasswdField1.setBorder( new TitledBorder( " new password " ) );
    newPasswdField1.setEchoChar('?');
    newPasswdField1.setFont(new Font("DialogInput", Font.PLAIN, 14));
    newPasswdField2 = new JPasswordField();
    newPasswdField2.setBorder( new TitledBorder( " new password (confirm) " ) );
    newPasswdField2.setEchoChar('?');
    newPasswdField2.setFont(new Font("DialogInput", Font.PLAIN, 14));
    */
//  }

/*
  public String queryStruct(String key, String paramValues) {
    String param = paramValues.trim();
    if (param.equals("")) {
      param = "empty";
    }
    String msg = key + ":" + param;
    try { 
      commonInfo.timerRestart();
      cout.println(msg);
      String answer = cin.readLine();
      if (answer.equals("null")) {
        return null;
      } else if (answer.startsWith("ERROR")) {  
        commonInfo.showMessage( answer );
        return null;      
      } else {  
        return answer;
      }
    } catch (Exception e) {
      commonInfo.showMessage( e.toString() );
      return null;
    }
  }
*/

  public String queryCommon(String key, String paramValues) {
    String param = paramValues.trim();
    if (param.equals("")) {
      param = "empty";
    }
    String msg = key + ":" + param;
    System.out.println("query = "+msg);
    try { 
      commonInfo.timerRestart();
      cout.println(msg);
      String answer = cin.readLine();
      if (answer.equals("null")) {
        return null;
      } else if (answer.startsWith("ERROR")) {  
        commonInfo.showMessage( answer );
        return null;      
      } else { 
    	  System.out.println("answer = " + answer);
        return answer;
      }
    } catch (Exception e) {
      commonInfo.showMessage( e.toString() );
      return null;
    }
  }

  public String query(String key, String params) {
    if (!qualified) {
      commonInfo.showMessage("User is not qualified to get data from DB.");
      System.exit(0);
    } 
    return queryCommon(key, params);
  }


  public int update(String key, String paramValues) {
    if (!qualified) {
      commonInfo.showMessage("User is not qualified to update data of DB.");	
      System.exit(0);
    } 

    String param = paramValues.trim();
    if (param.equals("")) {
      param = "empty";
    }
    String msg = key + ":" + param;

    try { 
      commonInfo.timerRestart();
      System.out.println("update q = " + msg);
      cout.println(msg);
      String answer = cin.readLine();
      if (answer.startsWith("ERROR")) { 
        int idx = answer.indexOf("$");
        commonInfo.showMessageLong(answer.substring(idx+1));
        return 0;       
      } else { 
        return 1;
      }
    } catch (Exception e) {
      commonInfo.showMessage(e.toString());
      return 0;
    }
  }

  public void checkUserQualification() {
    //  this method should be implemented in StaffServerConnection 
  }

  public void makeBackupOfStructTable(String tableName) {   
    //  this method should be implemented in Struct2ServerConnection 
  }

  public void readBackupOfStructTable(String tableName) {   
    //  this method should be implemented in Struct2ServerConnection 
  }

  public void setStaffDebugMode(String staffID) {   
    //  this method should be implemented in Struct2ServerConnection 
  }

  public void setStudentDebugMode(String studentCode) {   
    //  this method should be implemented in Struct2ServerConnection 
  }

  public void resetServerDebugMode() {   
    //  this method should be implemented in Struct2ServerConnection 
  }

  public String initializeStaffPassword(String userID) { 
    // this method should be installed in MasterServerConnection 
    return null;
  }
    
  public String registrateStaffPassword(String param) { 
    // this method should be installed in MasterServerConnection 
    return null;
  }    
}
