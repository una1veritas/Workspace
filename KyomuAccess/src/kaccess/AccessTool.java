package kaccess;

import java.io.*;
import java.net.*;
import java.util.*;
//import javax.net.ssl.*;
import javax.net.ssl.SSLSocketFactory;

public class AccessTool {
	static String TEACHERTOOL_SERVER = "kyomuinfo://131.206.103.7:3401/TeacherTool/";
	static String STUDENTTOOL_SERVER = "kyomuinfo://131.206.103.7:3402/StudentTool/";
	static String[] TOOLS = { TEACHERTOOL_SERVER, STUDENTTOOL_SERVER };
	static String KEYTRUST_URL = "http://kaoru.gakumu.jimui.kyutech.ac.jp/kyomu/nextGenerationClientTrust";
	static String STORE_PASSWORD = "NextGenerationKyomuInfo";
	static int ATTENDDB_HOST_PORT = 3289;

	String host;
	int port;
	String toolClass;
	Socket socket;
	BufferedReader sockin;
	PrintWriter sockout;

	AccessTool(String toolname) throws Exception {
		String tool;
		for (String uriString : TOOLS) {
			URI uri = new URI(uriString);
			tool = uri.getPath();
			if (tool.substring(1, tool.length() - 1).equals(toolname)) {
				host = uri.getHost();
				port = uri.getPort();
				toolClass = tool.substring(1, tool.length() - 1);
				break;
			}
		}
		if (toolClass.isEmpty())
			throw new Exception();
	}

	AccessTool(String hostip, int portNo) {
		host = hostip;
		port = portNo;
	}

	void makeSSLConnection(String trustStorePath, String storePassword)
			throws Exception {
		System.setProperty("javax.net.ssl.trustStore", trustStorePath);
		System.setProperty("javax.net.ssl.trustStorePassword", storePassword);
		socket = SSLSocketFactory.getDefault().createSocket(host, port);
		sockin = new BufferedReader(new InputStreamReader(
				socket.getInputStream(), "EUC-JP"));
		sockout = new PrintWriter(new OutputStreamWriter(
				socket.getOutputStream(), "EUC-JP"), true);
	}

	void closeConnection() throws IOException {
		socket.close();
	}

	public boolean checkConnectionToAttendServer() {
		try {
			sockout.println("10007KenjiroMaginu:Are you OK?");
			String answer = sockin.readLine();
			System.out.println(answer);
			if (answer.startsWith("OK")) {
				return true;
			}
			System.out.println("// Attend Server is not alive");
			return false;
		} catch (SocketException e1) {
			System.out.println("// " + e1.toString());
			String msg = e1.toString().trim();
			if (msg.equals("java.net.SocketException: Broken pipe")) {
				e1.printStackTrace();
			}
		} catch (Exception e2) {
			System.out.println("// " + e2.toString());
		}
		return false;
	}

	public String queryToAttendServer(String query) {
		String msg = "10006KenjiroMaginu:" + query;
		try {
			sockout.println(msg);
			String answer = sockin.readLine();
			if (answer.equals("null") || answer.startsWith("ERROR")) {
				System.err.println("// " + answer);
			} else {
				return answer;
			}
		} catch (Exception e) {
			System.out.println("// " + e.toString());
		}
		return null;
	}

	public String[] query(String key, String paramValues) throws IOException {
		String param = paramValues.trim();
		if (param.equals("")) {
			param = "empty";
		}
		String msg = key + ":" + param;
		String lines[] = msg.split("\r\n");
		for (String line : lines) {
			sockout.println(line);
		}
		String answer = sockin.readLine();
		while (sockin.ready()) {
			answer += "\r\n" + sockin.readLine();
		}
		System.err.println("answer to the query: " + key + ", " + paramValues);
		return answer.split("\\$");
	}

	static String[] TEACHER_PROPERTY_NAMES = { "STAFF_CODE", "STAFF_NAME",
			"STAFF_ATTRIB", "STAFF_QUALIFICATION", "MAIL_ADDRESS",
			"LOCAL_ATTRIB", "STAFF_DEPARTMENT" };
	static String[] STUDENT_PROPERTY_NAMES = { "STUDENT_CODE", "STUDENT_NAME",
			"STUDENT_STATUS", "STUDENT_FACULTY", "STUDENT_DEPARTMENT",
			"STUDENT_COURSE", "STUDENT_COURSE_2", "STUDENT_GAKUNEN",
			"STUDENT_CURRICULUM_YEAR", "STUDENT_SUPERVISOR",
			"STUDENT_SUPERVISOR_NAME", };

	public HashMap<String, String> getAttribute(String accid) {
		HashMap<String, String> propmap = new HashMap<String, String>();
		String[] result;
		try {
			if (toolClass.equals("TeacherTool")) {
				result = query("QUERYATTRIB|" + toolClass + "|STAFF",
						accid)[0].split("\\|");
				for (int i = 0; i < TEACHER_PROPERTY_NAMES.length; i++) {
					propmap.put(TEACHER_PROPERTY_NAMES[i], result[i]);
				}
				propmap.put("STAFF_FACULTY", result[6]);
//				String dep = propmap.get("STAFF_DEPARTMENT");
			} else if (toolClass.equals("StudentTool")) {
				result = query("QUERYATTRIB|" + toolClass + "|STUDENT", accid)[0]
						.split("\\|");
				for (int i = 0; i < STUDENT_PROPERTY_NAMES.length; i++) {
					propmap.put(STUDENT_PROPERTY_NAMES[i], result[i]);
				}
				String fac = propmap.get("STUDENT_FACULTY");
				if(fac.equals("32")) {
					propmap.put("STUDENT_FACULTY_NAME", "大学院");
					String course = propmap.get("STUDENT_COURSE");
					if(course.equals("85")) propmap.put("STUDENT_DEPARTMENT_NAME", "知能情報");
					else if(course.equals("88")) propmap.put("STUDENT_DEPARTMENT_NAME", "電子情報");
					else if(course.equals("86")) propmap.put("STUDENT_DEPARTMENT_NAME", "ｼｽﾃﾑ創成情報");
					else if(course.equals("89")) propmap.put("STUDENT_DEPARTMENT_NAME", "機械情報");
					else if(course.equals("87")) propmap.put("STUDENT_DEPARTMENT_NAME", "生命情報");
					else if(course.equals("90")) propmap.put("STUDENT_DEPARTMENT_NAME", "情報創成");
					else propmap.put("STUDENT_DEPARTMENT_NAME", "error");
				} else if(fac.equals("11")) {
					propmap.put("STUDENT_FACULTY_NAME", "学部");
					String dep = propmap.get("STUDENT_DEPARTMENT");
					if(dep.equals("31")) propmap.put("STUDENT_DEPARTMENT_NAME", "知能情報");
					else if(dep.equals("32")) propmap.put("STUDENT_DEPARTMENT_NAME", "電子情報");
					else if(dep.equals("33")) propmap.put("STUDENT_DEPARTMENT_NAME", "ｼｽﾃﾑ創成情報");
					else if(dep.equals("34")) propmap.put("STUDENT_DEPARTMENT_NAME", "機械情報");
					else if(dep.equals("35")) propmap.put("STUDENT_DEPARTMENT_NAME", "生命情報");
					else propmap.put("STUDENT_DEPARTMENT_NAME", "error");
				}
			}
			return propmap;
		} catch (IOException ex) {
			System.err.println("attribute query failed. " + ex);
		}
		return propmap;
	}

	public String authenticate(String name, String pwd) {
		if ((name.length() > 0) && (pwd.length() > 5)) {
			String key = "CHECKPASSWORD|" + toolClass + "|PASSWORD";
			String value = name + "|" + pwd;
			try {
				if (toolClass.equals("StudentTool")) {
					// System.out.println("Querying.. key: "+ key +" value: " +
					// value);
					String[] answer = query(key, value)[0].split("\\|");
					if (answer != null && answer[0].equals("success")) {
						return name;
					}
				} else if (toolClass.equals("TeacherTool")) {
					// System.out.println("Querying.. key: "+ key +" value: " +
					// value);
					String[] answer = query(key, value)[0].split("\\|");
					if (answer != null && answer[0].equals("success")) {
						return answer[1];
					}
				}
				System.err.println("query failed.");
			} catch (Exception ex) {
				System.err.println("サーバとの通信に失敗: " + ex);
			}
		}
		return "";
	}

	void copyURLContentsToFile(URL srcURL, File dstFile) throws IOException {
		InputStream in = srcURL.openStream();
		OutputStream out = new FileOutputStream(dstFile);
		for (int c; (c = in.read()) != -1;) {
			out.write(c);
		}
		out.close();
		in.close();
	}
}
