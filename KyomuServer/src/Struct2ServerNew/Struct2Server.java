package Struct2ServerNew;
import common.*;
import java.net.*;
import java.io.*;
import java.util.*;
import javax.net.ssl.SSLServerSocketFactory;

public class Struct2Server  {
  CommonInfo commonInfo; 
 
  public Struct2Server() {
    commonInfo = new CommonInfo();

    try {
      System.setProperty("javax.net.ssl.keyStore", "(»ÎÃ©æ Û)");
      System.setProperty("javax.net.ssl.keyStorePassword", "(»ÎÃ©æ Û)");

      ServerSocket serverSocket = SSLServerSocketFactory.getDefault().createServerSocket(commonInfo.serverPort);

      while(true){
	Socket socket = serverSocket.accept();
	Thread thread = new Struct2Handler(socket, commonInfo);
        thread.setDaemon(true);
        thread.start();
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  public static void main(String[] args) {
    new Struct2Server();
  }
}
