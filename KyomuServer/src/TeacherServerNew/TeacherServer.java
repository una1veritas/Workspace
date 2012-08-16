package TeacherServerNew;
import common.*;
import java.net.*;
import java.io.*;
//import java.util.*;
import javax.net.ssl.SSLServerSocketFactory;

public class TeacherServer  {
  CommonInfo commonInfo; 
 
  public TeacherServer() {
    commonInfo = new CommonInfo();

    try {
      System.setProperty("javax.net.ssl.keyStore", "http://kaoru.gakumu.jimui.kyutech.ac.jp/kyomu/nextGenerationClientTrust");
      System.setProperty("javax.net.ssl.keyStorePassword", "NextGenerationKyomuInfo");

      ServerSocket serverSocket = SSLServerSocketFactory.getDefault().createServerSocket(commonInfo.serverPort);

      while(true){
	Socket socket = serverSocket.accept();
	Thread thread = new TeacherHandler(socket, commonInfo);
        thread.setDaemon(true);
        thread.start();
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  public static void main(String[] args) {
    new TeacherServer();
  }
}
