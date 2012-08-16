package AttendServerNew;
import java.net.*;
import java.io.*;
import java.util.*;
import java.awt.*;
import java.sql.*;

public class AttendHandler extends Thread {
  private  Socket socket;  
  private InputStream istream;
  private OutputStream ostream;    
  private BufferedReader cin;
  private PrintWriter cout;
  private  CommonInfo commonInfo;  
  private JDBCConnection conn;
    
  public AttendHandler(Socket socket, CommonInfo commonInfo) {
    this.socket = socket;
    this.commonInfo = commonInfo;
    this.conn = commonInfo.conn;
    try{
      istream = socket.getInputStream();
      ostream = socket.getOutputStream();
      cin = new BufferedReader(new InputStreamReader(istream));
      cout = new PrintWriter(ostream, true);
    } catch (IOException e) {
      e.printStackTrace();
    }  
  }
    
  public void run() { 
    String line;

    try{
      while ((line = cin.readLine()) != null) {
	int ind = line.indexOf(":");
	String command = line.substring(0, ind);
	String param = line.substring(ind+1);

	if (command.equals("10008KenjiroMaginu")) {
	  try {
	    if (param.startsWith("insert")) {
	      int res = conn.executeAttendUpdate(param); 
	      if (res != 0) {
		cout.println("" + res);
	      } else {
		cout.println("ERROR$ " + param + " is not done.");
	      }
	    } else {
	      cout.println("ERROR$ not an insert command");
	    }
	  } catch (Exception e) {
	    System.out.println(e); 
	  }			
	} else if (command.equals("10010KenjiroMaginu")) {
	  try {
	    int res = conn.executeAttendErrorInsert(param); 
	    if (res != 0) {
	      cout.println("" + res);
	    } else {
	      cout.println("ERROR$ " + param + " is not inserted.");
	    }
	  } catch (Exception e) {
	    System.out.println(e); 
	  }	
	} else if (command.equals("10009KenjiroMaginu")) {
	  if (param.equals("hello")) {
	    try {
	      cout.println("AttendServer is going to die ... ");
	    } catch (Exception ex) { }
	    System.exit(0);
	  }
	} else if (command.equals("10007KenjiroMaginu")) {
	  try {
	    if (param.equals("Are you OK?")) {
	      cout.println("OK");
	    } else {
	      cout.println("ERROR$ not an OK command");
	    }
	  } catch (Exception e) {
	    System.out.println(e); 
	  }
	} else if (command.equals("10006KenjiroMaginu")) {
	  try {
	    if (param.startsWith("select")) {
	      String ans = conn.executeAttendQuery(param);
	      if (ans != null) {
		cout.println(ans);
	      } else {
		cout.println("null");
	      }
	    } else {
	      cout.println("ERROR$ not an query command");
	    }
	  } catch (Exception e) {
	    System.out.println(e); 
	  }	
	}
      }      
      socket.close();      
    } catch (Exception e) { }
  }
}
