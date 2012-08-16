package StudentServerNew;
import common.*;
import java.net.*;
import java.io.*;
import java.util.*;
import javax.net.ssl.SSLServerSocketFactory;

public class StudentServer  {
  CommonInfo commonInfo; 
 
  public StudentServer() {
    commonInfo = new CommonInfo();

    try {
      System.setProperty("javax.net.ssl.keyStore", "(秘密情報)");
      System.setProperty("javax.net.ssl.keyStorePassword", "(秘密情報)");
      ServerSocket serverSocket = SSLServerSocketFactory.getDefault().createServerSocket(commonInfo.serverPort);

      while(true){
	Socket socket = serverSocket.accept();
	Thread thread = new StudentHandler(socket, commonInfo);
        thread.setDaemon(true);
        thread.start();
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  public static void main(String[] args) {
    new StudentServer();
  }
}
