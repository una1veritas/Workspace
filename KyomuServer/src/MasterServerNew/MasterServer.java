package MasterServerNew;
import common.*;
import java.net.*;
import java.io.*;
import java.util.*;
import javax.net.ssl.SSLServerSocketFactory;

public class MasterServer  {
  CommonInfo commonInfo; 
 
  public MasterServer() {
    commonInfo = new CommonInfo();

    try {
      System.setProperty("javax.net.ssl.keyStore", "(»ÎÃ©æ Û)");
      System.setProperty("javax.net.ssl.keyStorePassword", "(»ÎÃ©æ Û)");
      ServerSocket serverSocket = SSLServerSocketFactory.getDefault().createServerSocket(commonInfo.serverPort);

      while(true){
	Socket socket = serverSocket.accept();
	Thread thread = new MasterHandler(socket, commonInfo);
        thread.setDaemon(true);
        thread.start();
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  public static void main(String[] args) {
    new MasterServer();
  }
}
