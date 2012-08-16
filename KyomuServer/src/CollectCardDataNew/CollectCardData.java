import java.net.*;
import java.io.*;
import java.util.*;

public class CollectCardData {  

  private int port = 1234;
  private PrintWriter fout;
  private PrintWriter efout;
  private ServerConnection conn;
  
  private Socket cardSocket;
  private BufferedReader netIn;
  private PrintWriter netOut;
  
  private ArrayList<ArrayList<String>> cardReaderInfo = new ArrayList<ArrayList<String>>();
  private HashMap<String, String> cardReaderMap1 = new HashMap<String, String>();
  private HashMap<String, String> cardReaderMap2 = new HashMap<String, String>();
  
  private boolean check_mode = true;
  private static boolean check_mode2 = false;
  
  public static void main( String[] args ) {
    CollectCardData ccd = new CollectCardData();   
    //    ccd.printCardReaderInfo();  // for debug
    ccd.setCorrectTime();
    ccd.loop();
  }
  
  public CollectCardData() {        
    try {
      Thread thread = new StopCollectionThread();                  
      thread.setDaemon( true ); 
      thread.start();
      
      Calendar rightNow = Calendar.getInstance();
      int year = rightNow.get(Calendar.YEAR);
      int month = rightNow.get(Calendar.MONTH)+1;
      String monthString;
      if (month < 10) {
        monthString = "0"+month;
      } else {
        monthString = ""+month;
      }            
      int day = rightNow.get(Calendar.DAY_OF_MONTH);
      String dayString;
      if (day < 10) {
        dayString = "0"+day;
      } else {
        dayString = ""+day;
      }
      
      String efname = "./maginu/ERROR-NEW/"+year+"-"+monthString+"-"+dayString;
      FileWriter efw = new FileWriter(efname, true);
      String fname = "./maginu/BACKUP-NEW/"+year+"-"+monthString+"-"+dayString;
      FileWriter fw = new FileWriter(fname, true);
      efout = new PrintWriter(efw, true);
      fout = new PrintWriter(fw, true);
      
      conn = new ServerConnection(efout);
      getCardReaderInfo();
      
    } catch (Exception ex) {
      ex.printStackTrace();
      System.exit(0);
    }
  }  
  
  private void getCardReaderInfo() {
    String query = "select CARD_READER, CARD_READER_IP from CARD_READER_INFO where ALIVE_FLAG = 1 and CARD_READER_IP is not null order by CARD_READER";
    String ans = conn.query(query);
    if (ans != null) {  
      String[] lines = ans.split("\\$");
      for (String line : lines) {
        String[] tokens = line.split("\\|");
        ArrayList<String> list = new ArrayList<String>();
        String id = tokens[0];
        String ip = tokens[1];
        list.add(id);
        list.add(ip);
        cardReaderInfo.add(list);
        cardReaderMap1.put(id, "-1");
        cardReaderMap2.put(id, "0");
      }
    }
  }
  /*
  private void printCardReaderInfo() {
    for (ArrayList<String> list : cardReaderInfo) {
      String readerID = list.get(0);
      String readerIP = list.get(1);
      
      System.out.println(readerID + " ==> " + readerIP);
    }
  }
  */
  private void setCorrectTime() {
    for (ArrayList<String> list : cardReaderInfo) {
      String readerID = list.get(0);
      String readerIP = list.get(1);
      
      setCardReaderClock(readerIP, readerID, port);
    }
  }
  
  private void setCardReaderClock(String readerIP, String readerID, int port) {
    try {
      cardSocket = new Socket(readerIP, port);
      cardSocket.setSoTimeout(5000);                  
      
      InputStream istream = cardSocket.getInputStream();
      InputStreamReader reader = new InputStreamReader(istream);
      netIn = new BufferedReader(reader);            
      OutputStream ostream = cardSocket.getOutputStream();
      netOut = new PrintWriter(ostream, true);
      
      try {        
        /* String okString = */ netIn.readLine(); 
        Thread.currentThread();
		Thread.sleep(10); 
        if (check_mode2) {
          System.out.println("setCardReaderClock: "+readerIP +" : "+readerID); //
        }
        
        String timeStr = getTimeString();
        if (check_mode2) {
          System.out.println("send data: " + timeStr);  //
        }
        
        netOut.print(timeStr+"\r\n"); 
        netOut.flush();                
        String lineText = netIn.readLine();
        if (check_mode2) {
          System.out.println("received data: " + lineText); //
        }
        
      } catch (Exception ex) {
        if (check_mode2) {
          System.out.println("//(1) "+ readerIP +" : "+readerID +" : "+ex.toString());
        }
      }
      
      //            Thread.currentThread().sleep(10);  
      //            netOut.print("x\r\n"); 
      //            netOut.flush();  
      //            Thread.currentThread().sleep(10);   
      cardSocket.close();
      
      Thread.currentThread();
      Thread.sleep(200);
      // sleep 200 msec
    } catch (Exception e) {         
      if (check_mode2) {
        System.out.println("//(2) "+ readerIP +" : "+readerID +" : "+e.toString());
      }
    }   
  }
  
  private String getTimeString() {
    Calendar rightNow = Calendar.getInstance();
    int year = rightNow.get(Calendar.YEAR);
    int month = rightNow.get(Calendar.MONTH)+1;
    int day = rightNow.get(Calendar.DAY_OF_MONTH);
    int week = rightNow.get(Calendar.DAY_OF_WEEK)-1;
    int hour = rightNow.get(Calendar.HOUR_OF_DAY);
    int min = rightNow.get(Calendar.MINUTE);
    int sec = rightNow.get(Calendar.SECOND);
    
    StringBuffer sbuf = new StringBuffer();
    sbuf.append("t "+year+" ");
    if (month < 10) {
      sbuf.append("0"+month+" ");
    } else {
      sbuf.append(""+month+" ");
    }
    if (day < 10) {
      sbuf.append("0"+day+" ");
    } else {
      sbuf.append(""+day+" ");
    }
    sbuf.append("0"+week+" ");
    if (hour < 10) {
      sbuf.append("0"+hour+" ");
    } else {
      sbuf.append(""+hour+" ");
    }
    if (min < 10) {
      sbuf.append("0"+min+" ");
    } else {
      sbuf.append(""+min+" ");
    }
    if (sec < 10) {
      sbuf.append("0"+sec);
    } else {
      sbuf.append(""+sec);
    }
    return sbuf.toString();
  }
  
  private String getPresentTime() {
    Calendar rightNow = Calendar.getInstance();
    int year = rightNow.get(Calendar.YEAR);
    int month = rightNow.get(Calendar.MONTH)+1;
    int day = rightNow.get(Calendar.DAY_OF_MONTH);
    int hour = rightNow.get(Calendar.HOUR_OF_DAY);
    int min = rightNow.get(Calendar.MINUTE);
    int sec = rightNow.get(Calendar.SECOND);
    
    StringBuffer sbuf = new StringBuffer();
    sbuf.append(""+year+":");
    if (month < 10) {
      sbuf.append("0"+month+":");
    } else {
      sbuf.append(""+month+":");
    }
    if (day < 10) {
      sbuf.append("0"+day+" ");
    } else {
      sbuf.append(""+day+" ");
    }
    if (hour < 10) {
      sbuf.append("0"+hour+":");
    } else {
      sbuf.append(""+hour+":");
    }
    if (min < 10) {
      sbuf.append("0"+min+":");
    } else {
      sbuf.append(""+min+":");
    }
    if (sec < 10) {
      sbuf.append("0"+sec);
    } else {
      sbuf.append(""+sec);
    }
    return sbuf.toString();
  }
  
  
  private void loop() {        
    long startTime = 0;                     
    int cnt = 0;                            
    
    conn.checkConnectionToServer();
    
    while (true) {
      cnt++;
      startTime = System.currentTimeMillis();    
      int number = 0;
      
      for (ArrayList<String> list : cardReaderInfo) {
        String readerID = list.get(0);
        String readerIP = list.get(1);
        
        long time1 = System.currentTimeMillis();     
        int num = readCardDataAndStore(readerIP, readerID, port, cnt);
        long time2 = System.currentTimeMillis();     
        int sec =  (int) (time2 - time1)/1000;  
        if (sec > 60) { 
          String ptime = getPresentTime();
          conn.insertError("" + sec +" seconds to fetch", readerID); 
          efout.println("// id = " + readerID + ", ip = " + readerIP + ": " + sec + "seconds to fetch: " + ptime);
        }
        number = number + num;
      }
      long endTime = System.currentTimeMillis();             
      int time = (int) (endTime - startTime)/1000;           
      if (check_mode) {        
        if (number != 0) {
          String dots;
          if (number < 10) {
            dots = "   ...  ";
          } else if (number < 100) {
            dots = "  ...  ";
          } else {
            dots = " ...  ";
          }
          String ptime = getPresentTime();
          efout.println("stored data = " + number + dots + ptime);        
        }
      }
      
      if (time < 60) {
        long length = (60 - time) * 1000;
        try {
          Thread.currentThread();
          Thread.sleep(length);   // wait 60 sec to start next scan.
        } catch (Exception e) { }
      } 
      long endTime2 = System.currentTimeMillis();             
      int time2 = (int) (endTime2 - startTime)/1000;  
      if (check_mode) {
        if (time2 > 180) {
          String ptime = getPresentTime();
          efout.println("// scan period = " + time2 + ": " + ptime); 
        }
      }            
      conn.checkConnectionToServer(); // check whether server is alive or not.
    }
  } 
  
  public void closeCardReader() {
    if (cardSocket != null) {
      try {
        //                Thread.currentThread().sleep(10);   // sleep 10 msec
        //                netOut.print("x\r\n"); 
        //                netOut.flush();  
        //                Thread.currentThread().sleep(10);   // sleep 10 msec
        cardSocket.close();
      } catch (Exception e) {
        e.printStackTrace();
      }
    }
  }
  
  public int readCardDataAndStore(String ip, String id, int port, int cnt) {
    String line;
    String cardData[] = new String[11];
    int num = 0;
    
    try {
      cardSocket = new Socket(ip, port);
      cardSocket.setSoTimeout(5000);                   // SoTimeout = 5 sec
      
      InputStream istream = cardSocket.getInputStream();
      InputStreamReader reader = new InputStreamReader(istream);
      netIn = new BufferedReader(reader);            
      OutputStream ostream = cardSocket.getOutputStream();
      netOut = new PrintWriter(ostream, true);
      
      try {        
        //String okString = 
    	  netIn.readLine();      // initial OK from card reader
        
        Thread.currentThread();
		Thread.sleep(10);       // sleep 10 msec
        
        netOut.print("c\r\n"); 
        netOut.flush();
        
        String lineText = netIn.readLine();
        if (lineText == null) {
          Thread.currentThread().sleep(10);   // sleep 10 msec  
          cardSocket.close();
          return num;
        }
        
        line = lineText.trim();                 // get number of stored data
        
        if (!line.startsWith("0000")) {
          
          Thread.currentThread().sleep(10);   // sleep 10 msec
          
          netOut.print("o\r\n"); 
          netOut.flush();
          
          String reply = netIn.readLine();
          if (reply == null) {                        
            //                        netOut.print("x\r\n");  
            //                        netOut.flush(); 
            //                        Thread.currentThread().sleep(10);   // sleep 10 msec 
            cardSocket.close();
            return num;
          }
          
          line = reply.trim();
          if (line.length() < 3) {
            efout.println("// Error data from o command: " + line);        
            //                        netOut.print("x\r\n");  
            //                        netOut.flush();
            //                        Thread.currentThread().sleep(10);   // sleep 10 msec  
            cardSocket.close();
            return num;
          }
          
          while (!line.startsWith("0000")) { 
            
            StringTokenizer stk = new StringTokenizer(line, ",");
            for (int i = 0; i < 11; i++) {
              cardData[i] = stk.nextToken();
            }         
            
            int c = addToDatabase(id, cardData, line);
            num = num + c;
            
            netOut.print(cardData[0] + "\r\n"); 
            netOut.flush();
            
            reply = netIn.readLine();
            if (reply == null) { 
              //                            netOut.print("x\r\n");  
              //                            netOut.flush(); 
              //                            Thread.currentThread().sleep(10);   // sleep 10 msec 
              cardSocket.close();
              return num;
            }                        
            line = reply.trim();
          }
          
        } else {
          Thread.currentThread().sleep(300);   // sleep 300 msec
        }
      } catch (Exception ex) {
        reportError(ex, id, ip, cnt);
      }
      
      //            Thread.currentThread().sleep(10);   // sleep 10 msec
      //            netOut.print("x\r\n"); 
      //            netOut.flush();  
      //            Thread.currentThread().sleep(10);   // sleep 10 msec
      cardSocket.close();
      
    } catch (Exception ex2) {
      reportError(ex2, id, ip, cnt);
    }
    
    resetErrorMap(cnt, id);
    return num;
  }
  
  public void resetErrorMap(int cnt, String id) {
    int cnt2 = Integer.parseInt((String)cardReaderMap1.get(id));
    if (cnt2 != cnt) {
      cardReaderMap1.put(id, "-1");
      cardReaderMap2.put(id, "0");
    }
  }
  
  public void reportError(Exception ex, String id, String ip, int cnt) {
    if (ex instanceof java.net.SocketTimeoutException) return;
    
    String ptime = getPresentTime();
    int cnt2 = Integer.parseInt((String)cardReaderMap1.get(id));
    if (cnt - cnt2 == 1) {
      int times = Integer.parseInt((String)cardReaderMap2.get(id));
      if (times >= 2) {
        conn.insertError(ex.toString(), id);
        efout.println("// id = " + id + ", ip = " + ip + ": " + ex.toString() + " ... " + ptime);
      } 
      times = times + 1;
      cardReaderMap1.put(id, ""+cnt);                    
      cardReaderMap2.put(id, ""+times);
    } else {
      cardReaderMap1.put(id, ""+cnt);                    
      cardReaderMap2.put(id, "1");
    }
  }
  
  public int addToDatabase(String id, String[] cardData, String line) {    
    String school_year;
    String semester;
    int result = 0;
    
    if (!cardData[2].equals("G")) {
      efout.println("// non-student: " + line);   // non student data is not stored
      return 0;
    }
    
    int month = Integer.parseInt(cardData[6]);
    
    if ((month >= 1) && (month <= 3)) {
      int year = Integer.parseInt(cardData[5]) - 1;
      school_year = "" + year; 
    } else {
      school_year = cardData[5];
    }
    
    if(month >= 4 && month <= 9) {
      semester = "1";
    } else {
      semester = "2";
    }
    
    String student_code = cardData[3].substring(0,8);
    String date = cardData[5]+":"+cardData[6]+":"+cardData[7];
    String time = cardData[8]+":"+cardData[9]+":"+cardData[10];
  //  String read_flag = "0";
    
    StringBuffer sbuf = new StringBuffer();
    sbuf.append("insert into CARD_READER_DATA (SCHOOL_YEAR, SEMESTER, STUDENT_CODE, CARD_READER, ATTEND_DATE, READ_DATE_TIME, READ_FLAG, REMARK) values ("); 
    sbuf.append("'"+school_year+"',");
    sbuf.append("'"+semester+"',");
    sbuf.append("'"+student_code+"',");
    sbuf.append("'"+id+"',");
    sbuf.append("to_date('"+date+"','YYYY:MM:DD'),");
    sbuf.append("to_date('"+date+":"+time+"','YYYY:MM:DD:HH24:MI:SS'),");
    sbuf.append("'0', null)");
    
    try{
      fout.println(line);
      
      String ins = sbuf.toString();
      result = conn.insertAttendData(ins);
      if (result != 1) {
        //                conn.insertError("INSERT ERROR: "+line, id); 
      }
    } catch (Exception e) {
      efout.println("// exception (3) " + e.toString() + " ... " + line);
      conn.insertError(e.toString()+":"+line, id); //
    }
    return result;
  } 
  
  
  class  StopCollectionThread extends Thread {
    private int serverPort = 8000;
    private ServerSocket serverSocket;
    
    public StopCollectionThread() {
      try {
        serverSocket = new ServerSocket(serverPort);
      } catch (BindException e) {
        System.exit(0);
      } catch (Exception ex) {
        ex.printStackTrace();
      }
    }
    
    public void run() {    
      try {
        while (true) {
          Socket socket = serverSocket.accept();                       
          
          InputStream istream = socket.getInputStream();               
          InputStreamReader reader = new InputStreamReader(istream);
          BufferedReader in = new BufferedReader(reader);   
          
          String line1 = in.readLine();
          String line2 = in.readLine();
          
          if ((line1.equals("KenjiroMaginu")) && (line2.equals("quit"))) {
            fout.close();
            efout.close();
            conn.close();                
            closeCardReader();
            break;
          }
        } 
      } catch (IOException e) {
        e.printStackTrace();
      }
      System.exit(0);
    } 
  }  
}
