package TeacherServerNew;

import common.*;
import java.net.*;
import java.io.*;
import java.util.*;
import java.awt.*;
import java.sql.*;
import syllabusNew.*;

public class TeacherHandler extends Thread {
	private CommonInfo commonInfo;
	private Socket socket;
	private InputStream istream;
	private OutputStream ostream;
	private BufferedReader cin;
	private PrintWriter cout;
	private TeacherHandlerMethods handlerMethods;

	public TeacherHandler(Socket socket, CommonInfo commonInfo) {
		this.socket = socket;
		this.commonInfo = commonInfo;
		try {
			istream = socket.getInputStream();
			ostream = socket.getOutputStream();
			cin = new BufferedReader(new InputStreamReader(istream));
			cout = new PrintWriter(ostream, true);
		} catch (IOException e) {
			e.printStackTrace();
		}
		handlerMethods = new TeacherHandlerMethods(commonInfo, socket, istream,
				ostream, cin, cout);
	}

	public void run() {
		String line;
		try {
			while ((line = cin.readLine()) != null) {
				int ind = line.indexOf(":");
				String keyValue = line.substring(0, ind);
				String paramValues = line.substring(ind + 1);
				String[] tokens = keyValue.split("\\|");
				String commandType = tokens[0];
				String serviceName = tokens[1];
				String commandCode = tokens[2];

				if ((commandType.equals("QUERY"))
						&& (serviceName.equals("COMMON_QUERY"))) {
					handlerMethods.sendCommonQueryResult(commandCode,
							paramValues);
				} else if (commandType.equals("QUERY")) {
					handlerMethods.sendQueryResult(serviceName, commandCode,
							paramValues);
					// "QUERY|"+serviceName+"|"+panelID+"#"+switchCode;
				} else if (commandType.equals("DELETE")) {
					handlerMethods.sendDeleteResult(serviceName, commandCode,
							paramValues);
				} else if (commandType.equals("UPDATE")) {
					handlerMethods.sendUpdateResult(serviceName, commandCode,
							paramValues);
				} else if (commandType.equals("INSERT")) {
					handlerMethods.sendInsertResult(serviceName, commandCode,
							paramValues);
				} else if (commandType.equals("SPECIAL")) {
					handlerMethods.sendSpecialResult(serviceName, commandCode,
							paramValues);
				} else if (commandType.equals("CHECKPASSWORD")) {
					handlerMethods.sendPasswordCheckResult(paramValues);
				} else if (commandType.equals("CHANGEPASSWORD")) {
					handlerMethods.sendPasswordChangeResult(paramValues);
				} else if ((commandType.equals("QUERYATTRIB"))
						&& (commandCode.equals("STAFF"))) {
					handlerMethods.sendQueryStaffAttribResult(paramValues);
				} else if (commandType.equals("SENDSYLLABUS")) {
					handlerMethods.sendSyllabusXml(paramValues);
				} else if (commandType.equals("SAVESYLLABUS")) {
					handlerMethods.saveSyllabusXml(paramValues);
				} else if (commandType.equals("SYLLABUSTOOL")) {
					handlerMethods.syllabusTool(commandCode, paramValues);
				} else if (commandType.equals("COPYSYLLABUSTONEXTYEAR")) {
					handlerMethods.copySyllabusToNextYear(paramValues);
				} else if (commandType.equals("COPYCLASSTIMEZONETONEXTYEAR")) {
					handlerMethods.copyClassTimeZoneToNextYear(paramValues);
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
				} else if (commandType.equals("RESETATTENDCONTROLPARAMS")) {
					if (commandCode.equals("ALLSUBJECTS")) {
						handlerMethods
								.resetAttendControlParamsOfAll(paramValues);
					} else {
						handlerMethods.resetAttendControlParams(paramValues);
					}
				} else if ((commandType.equals("PHOTO"))
						&& (serviceName.equals("STUDENT"))) {
					handlerMethods.sendStudentPhoto(paramValues);
				} else if (commandType.equals("SERVER_KILLER")) {
					if ((commandCode.equals("TEACHER_SERVER"))
							&& (paramValues.equals(commonInfo.SecretKillerCode
									+ "|" + commonInfo.SecretKillerPassword))) {
						System.exit(0);
					} else {
						handlerMethods.sendErrorMessage("ERROR: format error: "
								+ line);
					}
				} else if (commandType.equals("SERVER_CONTROLLER")) {
					if ((commandCode.equals("TEACHER_SERVER"))
							&& (paramValues.equals(commonInfo.SecretKillerCode
									+ "|" + commonInfo.SecretKillerPassword))) {
						try {
							cout.println("accept");
							handlerMethods.executeControlLoop();
						} catch (Exception ex) {
							ex.printStackTrace();
						}
					}
				} else {
					handlerMethods.sendErrorMessage("ERROR: format error: "
							+ line);
				}
			}
			handlerMethods.removeLoginMap();
			socket.close();
		} catch (Exception e) {
		}
	}
}
