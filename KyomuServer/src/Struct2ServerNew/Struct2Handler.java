package Struct2ServerNew;
import common.*;
import java.net.*;
import java.io.*;
import java.util.*;
import java.awt.*;
import java.sql.*;
import syllabusNew.*;

public class Struct2Handler extends Thread {
  private  CommonInfo commonInfo;  
  private  Socket socket;  
  private InputStream istream;
  private OutputStream ostream;    
  private BufferedReader cin;
  private PrintWriter cout;
  private Struct2HandlerMethods handlerMethods;
  
  public Struct2Handler(Socket socket, CommonInfo commonInfo) {
    this.socket = socket;
    this.commonInfo = commonInfo;
    try{
      istream = socket.getInputStream();
      ostream = socket.getOutputStream();
      cin = new BufferedReader(new InputStreamReader(istream));
      cout = new PrintWriter(ostream, true);
    } catch (IOException e) {
      e.printStackTrace();
    }  
    handlerMethods = new Struct2HandlerMethods(commonInfo, socket, 
					       istream, ostream, cin, cout);
  }
    
  public void run() {  
    String line;
    try{
      while ((line = cin.readLine()) != null) {
	int ind = line.indexOf(":");
	String keyValue = line.substring(0, ind);
	String paramValues = line.substring(ind+1);
	String[] tokens = keyValue.split("\\|");
	String commandType = tokens[0];
	String serviceName = tokens[1];
	String commandCode = tokens[2];

	if ((commandType.equals("QUERY")) && (serviceName.equals("COMMON_QUERY"))) {
	  handlerMethods.sendCommonQueryResult(commandCode, paramValues);
	} else if (commandType.equals("QUERY")) {	    
	  handlerMethods.sendQueryResult(serviceName, commandCode, paramValues);
	  //  "QUERY|"+serviceName+"|"+panelID+"#"+switchCode;
	} else if (commandType.equals("QUERY_SINGLE_STRUCT")) {
	  handlerMethods.sendSingleStructQueryResult(serviceName, commandCode, paramValues);
	  //  "QUERY_SINGLE_STRUCT|"+serviceName+"|"+panelID+"#"+switchCode;
	} else if (commandType.equals("STRUCTBACKUP")) {
	  //  "STRUCTBACKUP|STRUCTCONTROL|MAKEBACKUP:"+tableName;
	  //  "STRUCTBACKUP|STRUCTCONTROL|READBACKUP:"+tableName;
	  if (commandCode.equals("MAKEBACKUP")) {
	    handlerMethods.sendMakeBackupResult(paramValues);
	  } else if (commandCode.equals("READBACKUP")) {
	    handlerMethods.sendReadBackupResult(paramValues);
	  }
	} else if (commandType.equals("SETDEBUGMODE")) {
	  //  "SETDEBUGMODE|STRUCTCONTROL|STAFFUSER:"+userID;
	  //  "SETDEBUGMODE|STRUCTCONTROL|STUDENTUSER:"+studentCode;
	  if (commandCode.equals("STAFFUSER")) {
	    handlerMethods.setStaffUserForDebug(paramValues);
	  } else if (commandCode.equals("STUDENTUSER")) {
	    handlerMethods.setStudentUserForDebug(paramValues);
	  }
	} else if (commandType.equals("RESETDEBUGMODE")) {
	  //  "RESETDEBUGMODE|STRUCTCONTROL|STAFFUSER";
	  handlerMethods.resetUserForDebug(paramValues);	  
	} else if (commandType.equals("DELETE")) {
	  handlerMethods.sendDeleteResult(serviceName, commandCode, paramValues);
	} else if (commandType.equals("UPDATE")) {
	  handlerMethods.sendUpdateResult(serviceName, commandCode, paramValues);
	} else if (commandType.equals("INSERT")) {
	  handlerMethods.sendInsertResult(serviceName, commandCode, paramValues);
	} else if (commandType.equals("CHECKPASSWORD")) {
	  handlerMethods.sendPasswordCheckResult(paramValues);
	} else if (commandType.equals("CHANGEPASSWORD")) {
	  handlerMethods.sendPasswordChangeResult(paramValues);
	} else if ((commandType.equals("QUERYATTRIB")) && (commandCode.equals("STAFF"))) {
	  handlerMethods.sendQueryStaffAttribResult(paramValues);
	} else if ((commandType.equals("QUERYATTRIB")) && (commandCode.equals("STUDENT"))) {
	  handlerMethods.sendQueryStudentAttribResult(paramValues); 
	} else if (commandType.equals("SENDSYLLABUS")) {
	  handlerMethods.sendSyllabusXml(paramValues); 
	} else if (commandType.equals("SAVESYLLABUS")) {
	  handlerMethods.saveSyllabusXml(paramValues);
	} else if (commandType.equals("SYLLABUSTOOL")) {
	  handlerMethods.syllabusTool(commandCode, paramValues);
	} else if (commandType.equals("SAVEKIMATSU")) {
	  handlerMethods.saveKimatsuData(paramValues);  
	} else if (commandType.equals("REPORTKIMATSU")) {
	  handlerMethods.saveKimatsuData(paramValues);
	} else if (commandType.equals("SAVESAISHI")) {
	  handlerMethods.saveSaishiData(paramValues);
	} else if (commandType.equals("REPORTSAISHI")) {
	  handlerMethods.saveSaishiData(paramValues);
	} else if (commandType.equals("REPORTNINTEI")) {
	  handlerMethods.reportNinteiData(paramValues); 
	} else if ((commandType.equals("PHOTO")) && (serviceName.equals("STUDENT"))) {
	  handlerMethods.sendStudentPhoto(paramValues);
	} else if (commandType.equals("CANCELREGISTR")) {
	  handlerMethods.cancelRegistration(paramValues);
	} else if (commandType.equals("ADDREGISTR")) {
	  handlerMethods.addRegistration(paramValues);
	} else if (commandType.equals("UPDATEGAKUSEKI")) {
	  handlerMethods.updateGakuseki(paramValues); 
	} else if ((commandType.equals("GAKUSEKI")) && (serviceName.equals("STUDENT"))) {
	  handlerMethods.sendStudentGakuseki(paramValues); 
	} else if (commandType.equals("SERVER_KILLER")) {
	  if ((commandCode.equals("STRUCT2_SERVER")) && 
	      (paramValues.equals(commonInfo.SecretKillerCode+"|"+commonInfo.SecretKillerPassword))) {
	    System.exit(0);
	  } else {
	    handlerMethods.sendErrorMessage("ERROR: format error: " + line);
	  }
	} else {
	  handlerMethods.sendErrorMessage("ERROR: format error: " + line);
	} 
      }
      handlerMethods.removeLoginMap();
      socket.close();      
    } catch (Exception e) { }
  }	
}
