import java.net.*;
import java.io.*;
//import java.util.*;

public class StopCollection {

  private int serverPort = 8000;                      
  private String serverHost = "localhost";
  
  public static void main (String[] args) {
    new StopCollection();
  }
  
  public StopCollection () {
    
    try {     
      Socket socket = new Socket(serverHost, serverPort);  
      OutputStream ostream = socket.getOutputStream(); 
      PrintWriter out = new PrintWriter(ostream, true);            
      
      out.println("KenjiroMaginu");
      out.println("quit");
    } catch (Exception e) { }
  }
}
