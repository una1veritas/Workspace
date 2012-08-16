package kaccess;

import java.io.*;
import java.net.*;
import java.util.*;
//import javax.net.ssl.*;
//import javax.net.ssl.SSLSocketFactory;

public class KyomuAccess implements Runnable {
	public static int servicePort = 3409;
	public static HashMap<String, KyomuAccess> accessors = new HashMap<String, KyomuAccess>();

	BufferedWriter output;
	BufferedReader input;
	HashMap<String, String> statdict;
	AccessTool tool;
	Thread myThread;

	public String readToken(long millis) throws IOException {
		int ch = -1;
		long timer = System.currentTimeMillis();
		StringBuffer tmp = new StringBuffer();
		do {
			while (input.ready()) { // ストリームが読み込み可能だった場合
				if ((ch = input.read()) == -1)
					break;
				// System.err.println(ch + ": " + (char)ch);
				if (ch == 0)
					continue;
				if (Character.isWhitespace((char) ch))
					break;
				tmp.append((char) ch);
				timer = System.currentTimeMillis();
			}
			if (millis > 0 && System.currentTimeMillis() > timer + millis)
				break;
		} while ((tmp.length() == 0)
				|| (ch != -1 && !Character.isWhitespace((char) ch)));
		return tmp.toString();
	}

	public String readToken() throws IOException {
		return readToken(5000);
	}

	public KyomuAccess(Socket toclient) throws Exception {
		input = new BufferedReader(new InputStreamReader(
				toclient.getInputStream(), "UTF-8"));
		output = new BufferedWriter(new OutputStreamWriter(
				toclient.getOutputStream(), "UTF-8"));
		statdict = new HashMap<String, String>();
		tool = null; // will be assigned when opened.
		myThread = null;
	}

	public synchronized void reconnect(BufferedReader in, BufferedWriter out)
			throws Exception {
		input = in;
		output = out;
		writeln("reconnected.");
	}

	public void open(String toolname) throws Exception {
		tool = new AccessTool(toolname);
		File tmpfile = File.createTempFile("keyTrust", ".key");
		tmpfile.deleteOnExit();
		tool.copyURLContentsToFile(new URL(AccessTool.KEYTRUST_URL), tmpfile);
		tool.makeSSLConnection(tmpfile.getPath(), AccessTool.STORE_PASSWORD);
	}

	public void closeBuffer() {
		try {
			output.close();
			input.close();
		} catch (IOException ex) {
			System.err.println("Ignoring an exception on closing. " + ex);
		}
	}

	public void close() {
		try {
			output.close();
			input.close();
			tool.closeConnection();
		} catch (IOException ex) {
			System.err.println("Ignoring an exception on closing. " + ex);
		} catch (NullPointerException ex) {
			System.err.println("Ignoring an exception on closing. " + ex);
		}
	}

	Thread thread(Thread thre) {
		return myThread = thre;
	}

	Thread thread() {
		return myThread;
	}

	void write(String message) throws Exception {
		try {
			output.write(message);
		} catch (Exception ex) {
			System.err.println("Error on writeln: " + ex);
			accessors.remove(thread().getName());
			close();
		}
		System.out.print(message);
	}

	void writeln(String message) {
		try {
			write(message);
			output.newLine();
			output.flush();
		} catch (Exception ex) {
			System.err.println("Error on writeln: " + ex);
			accessors.remove(thread().getName());
			close();
		}
		System.out.println();
	}

	void writeln() {
		try {
			output.newLine();
			output.flush();
		} catch (Exception ex) {
			System.err.println("Error on writeln: " + ex);
			accessors.remove(thread().getName());
			close();
		}
		System.out.println();
	}

	public void run() {
		//
		String cmd = "";
		writeln("READY.");
		for (;;) {
			try {
				if ((cmd = readToken()).isEmpty()) {
					Thread.sleep(5);
					continue;
				}
				// System.err.println("cmd = " + cmd);
				if (!processComand(cmd)) {
					break;
				}
			} catch (Exception ex) {
				System.err.println("Error on command " + cmd + ", " + ex);
				if (!cmd.isEmpty()) {
					writeln("Error");
					break;
				}
			}
			if (cmd.equalsIgnoreCase("suspend")) { // suspendコマンドがきたら休む
				closeBuffer();
				synchronized (this) {
					try {
						this.wait();
					} catch (InterruptedException ex) {
					}
				}
			}
			writeln("READY.");
		}
		accessors.remove(thread().getName());
		System.err.println("Closing...");
		close();
	}

	public static void main(String args[]) throws Exception {
		try {
			ServerSocket servsocket = new ServerSocket(servicePort);

			while (true) {
				KyomuAccess temp = new KyomuAccess(servsocket.accept());
				temp.writeln("SESSIONID?");
				String session = temp.readToken(); // セッションIDをとってくる
				System.out.println("sessionID = " + session);
				KyomuAccess acc;
				if ((acc = KyomuAccess.accessors.get(session)) != null) { // 既に登録されたセッションIDの場合
					System.err.println("Thread state "
							+ acc.thread().getState());
					if (acc.thread().getState() == Thread.State.TERMINATED) { // スレッド再開
						KyomuAccess.accessors.remove(session); // スレッドが終了していたらハッシュマップから削除
						acc = null;
					}
				}
				if (acc == null) {
					(temp.thread(new Thread(temp, session))).start(); // 新規スレッドなら作成してスタート
				} else {
					acc.reconnect(temp.input, temp.output);
					acc.thread().interrupt();
				}
			}
			//
		} catch (IOException ex) {
			System.err.println("Error in main: " + ex);
			System.exit(0);
		}
	}

	//
	public boolean processComand(String cmd) throws Exception {
		if (cmd.isEmpty())
			return false;

		System.out.println("command: " + cmd);

		if (cmd.equalsIgnoreCase("CONNECT")) {
			String toolname = readToken();
			try {
				System.err.println("CONNECT \"" + toolname + "\", \""
						+ thread().getName() + "\".");
				open(toolname);
				System.err.println("CONNECTED.");
				accessors.put(thread().getName(), this);
				processComand("TODAY");
				writeln("SUCCEEDED SSL CONNECTION for " + toolname);
			} catch (Exception e) {
				writeln("FAILED SSL CONNECTION for " + toolname);
				System.err.println("Error in establishing connection: " + e);
				return false;
			}
		} else if (cmd.equalsIgnoreCase("PROPERTY")) {
			String propname = readToken();
			if (!propname.isEmpty()) {
				if (statdict.containsKey(propname)) {
					write(statdict.get(propname));
				} else {
					write("INVALID PROPERTY NAME " + propname);
				}
				writeln();
			}
		} else if (cmd.equalsIgnoreCase("STATUS")) {
			String fac = statdict.get("STUDENT_FACULTY");
			write("所属:　");
			if (fac.equals("32")) {
				write("情報工学府");
			}
			String dep = statdict.get("STUDENT_DEPARTMENT_NAME");
			write(dep);
			if (fac.equals("32")) {
				write("　");
			} else {
				write("工学科");
			}
			write(statdict.get("STUDENT_GAKUNEN"));
			write("年<br>学籍番号:　");
			write(statdict.get("ACCESSID"));
			write("<br>名前:　");
			write(statdict.get("STUDENT_NAME"));
			write("<br>指導教員:　");
			writeln(statdict.get("STUDENT_SUPERVISOR_NAME"));
			/*
			 * for (String name : statdict.keySet()) { write(name + " = ");
			 * writeln(statdict.get(name)); } writeln("DONE.");
			 */

		} else if (cmd.equalsIgnoreCase("LOGIN")) {
			// writeln("id, password ?>");
			String id = readToken();
			String pwd = readToken();
			String accessid = null;
			statdict.remove("ACCESSID");
			if (!id.isEmpty() && !pwd.isEmpty()
					&& !(accessid = tool.authenticate(id, pwd)).isEmpty()) {
				statdict.putAll(tool.getAttribute(accessid));
				statdict.put("ACCESSID", accessid);
				//System.err.println(statdict);
				writeln("SUCCEEDED LOGIN");
			} else {
				writeln("FAILED LOGIN");
				System.err.println("login failed.");
				return false;
			}

		} else if (cmd.equalsIgnoreCase("TODAY")) {
			String k = "QUERY|COMMON_QUERY|queryPresentDate";
			String p = "";
			statdict.remove("DATETODAY");
			String res = tool.query(k, p)[0];
			res = res.substring(0, res.length() - 1).replace('|', '/');
			statdict.put("DATETODAY", res);
			writeln(res);
			writeln("DONE.");

		} else if (cmd.equalsIgnoreCase("common")) {
			writeln("Tool|Number|UnknownCode > ");
			String tmp = readToken();
			if (!tmp.isEmpty()) {
				String res = tool.query(
						"QUERY|COMMON_QUERY|queryServerQueryParams", tmp)[0];
				for (String s : res.split("\\$")) {
					writeln(s);
				}
			}
			writeln("DONE.");

		} else if (cmd.equalsIgnoreCase("DOWNLOADSYLLABUS")) {
			// query = SENDSYLLABUS|SYLLABUS|XML:2011|11316104|82541854
			String schlyear = readToken();
			String subjid = readToken();
			String teacherid = readToken();
			if (schlyear.isEmpty() || subjid.isEmpty() || teacherid.isEmpty()) {
				writeln("There is/are unspecified argment(s).");
			} else {
				String[] res = tool.query("SENDSYLLABUS|SYLLABUS|XML", schlyear
						+ '|' + subjid + '|' + teacherid);
				StringBuffer buf = new StringBuffer();
				for (String line : res) {
					writeln(line);
					buf.append(line);
				}
				statdict.put("SYLLABUS", buf.toString());
			}
			writeln("$");

		} else if (cmd.equalsIgnoreCase("UPLOADSYLLABUS")) {
			// String msg = "SAVESYLLABUS|SYLLABUS|XML:" + paramValues;
			String schlyear = readToken();
			String subjid = readToken();
			String teacherid = readToken();
			if (schlyear.isEmpty() || subjid.isEmpty() || teacherid.isEmpty()) {
				writeln("CANCELED.");
				return true;
			}
			String sylltext = statdict.get("SYLLABUS");
			tool.query("SAVESYLLABUS|SYLLABUS|XML", schlyear + '|' + subjid
					+ '|' + teacherid + "\r\n" + sylltext + "\r\n.");
			writeln("DONE.");

		} else if (cmd.equalsIgnoreCase("filein")) {
			String propname = readToken();
			String filename = readToken();
			FileInputStream fis = new FileInputStream(filename);
			BufferedReader infile = new BufferedReader(new InputStreamReader(
					fis, "EUC-JP"));
			StringBuffer buf = new StringBuffer(infile.readLine());
			while (infile.ready()) {
				buf.append("\r\n");
				buf.append(infile.readLine());
			}
			infile.close();
			statdict.put(propname, buf.toString());
			writeln("DONE.");

		} else if (cmd.equalsIgnoreCase("fileout")) {
			String propname = readToken();
			String filename = readToken();
			FileOutputStream fos = new FileOutputStream(filename);
			PrintWriter outfile = new PrintWriter(new OutputStreamWriter(fos,
					"EUC-JP"));
			outfile.write(statdict.get(propname));
			outfile.close();
			writeln("DONE.");

		} else if (cmd.equalsIgnoreCase("GET")) {
			String obj = readToken();
			String[] res = null;
			String accessid = statdict.get("ACCESSID");
			if (obj.equalsIgnoreCase("KYOMUINFO")) {
				int reqno = 200;
				res = tool.query("QUERY|UndergradKyomuInfo|" + reqno + "#0",
						accessid);
			} else if (obj.equalsIgnoreCase("STUDIED")) {
				Calendar cal = Calendar.getInstance();
				int year = cal.get(Calendar.YEAR);
				int month = cal.get(Calendar.MONTH);
				if (month < 4)
					year--;
				int reqno = 201;
				res = tool.query("QUERY|UndergradKyomuInfo|" + reqno + "#0",
						accessid + "|" + year);
			} else if (obj.equalsIgnoreCase("ATTENDANCE")) {
				int reqno = 203;
				String param1 = readToken();
				String param2 = readToken();
				String param3 = readToken();
				res = tool.query("QUERY|UndergradKyomuInfo|" + reqno + "#0",
						accessid + "|" + param1 + '|' + param2 + '|' + param3);
			} else if (obj.equalsIgnoreCase("COURSES")) {
				// QUERY|CurriculumTool|202#0:2012|11|31|10|637|
				int reqno = 204; // 202 get list
				String param1 = readToken();
				String param2 = readToken();
				String param3 = readToken();
				String param4 = readToken();
				String param5 = readToken();
				res = tool.query("QUERY|SyllabusTool|" + reqno + "#0", param1
						+ '|' + param2 + '|' + param3 + '|' + param4 + '|'
						+ param5 + '|');
			} else if (obj.equalsIgnoreCase("SYLLABUS")) {
				int reqno = 207;
				String param1 = readToken();
				String param2 = readToken();
				String param3 = readToken();
				res = tool.query("QUERY|SyllabusTool|" + reqno + "#0", param1
						+ '|' + param2 + '|' + param3);
			} else if (obj.equalsIgnoreCase("CALENDAR")) {
				Calendar cal = Calendar.getInstance();
				int year = cal.get(Calendar.YEAR);
				int month = cal.get(Calendar.MONTH) + 1;
				int reqno = 200;
				res = tool.query("QUERY|CalendarTool|" + reqno + "#0", year
						+ "|" + month);
			} else if (obj.equalsIgnoreCase("NAME")) {
				if (tool.toolClass.equals("TeacherTool")) {
					writeln(statdict.get("STAFF_NAME"));
				} else {
					writeln(statdict.get("STUDENT_NAME"));
				}
			} else if (obj.equalsIgnoreCase("FACULTY")) {
				if (tool.toolClass.equals("TeacherTool")) {
					writeln(statdict.get("STAFF_FACULTY"));
				} else {
					writeln(statdict.get("STUDENT_FACULTY_NAME"));
				}
			} else if (obj.equalsIgnoreCase("DEPARTMENT")) {
				if (tool.toolClass.equals("TeacherTool")) {
					writeln(statdict.get("STAFF_DEPARTMENT"));
				} else {
					writeln(statdict.get("STUDENT_DEPARTMENT_NAME"));
				}
			} else if (obj.equalsIgnoreCase("KUBUN_SUBJECTS")) {
				writeln("RESPONCE GET KUBUN_SUBJECTS");
				int reqno = 218;
				String param1 = readToken();
				String param2 = readToken();
				res = tool.query("QUERY|MeiboTool|" + reqno + "#0", param1
						+ '|' + param2);
				writeln();
			} else if (obj.equalsIgnoreCase("REGISTERED_STUDENTS")) {
				writeln("RESPONCE GET REGISTERED_STUDENTS");
				int reqno = 219;
				String param1 = readToken();
				String param2 = readToken();
				String param3 = readToken();
				if (param3.isEmpty())
					param3 = "01";
				res = tool.query("QUERY|MeiboTool|" + reqno + "#0", param1
						+ '|' + param2 + '|' + param3);
			} else if (obj.equalsIgnoreCase("JIKANWARI")) {
				String schlyear = readToken();
				String term = readToken();
				String dept = readToken();
				String ord = readToken();
				String faculty = "11";
				res = tool.query("QUERY|JikanwariTool|202#8", schlyear + "|"
						+ term + "|" + faculty + "|" + dept + "|" + ord);
				//
				//SCHOOL_YEAR:SEMESTER:FACULTY:DEPARTMENT:GAKUNEN|$
				//QUERY|JikanwariTool|202#8:2011|1|11|31|2|
				//
				//query = QUERY|JikanwariTool|201#0:2011|1|11|
				//answer = 2011|1|31|1|$2011|1|31|2|$2011|1|31|3|$2011|1|31|4|$2011|1|32|1|$2011|1|32|2|$2011|1|32|3|$2011|1|32|4|$2011|1|33|1|$2011|1|33|2|$2011|1|33|3|$2011|1|33|4|$2011|1|34|1|$2011|1|34|2|$2011|1|34|3|$2011|1|34|4|$2011|1|35|1|$2011|1|35|2|$2011|1|35|3|$2011|1|35|4|$
				//answer = SCHOOL_YEAR:SEMESTER:FACULTY|$
				//SCHOOL_YEAR:SEMESTER:FACULTY:DEPARTMENT|$
				//query = QUERY|JikanwariTool|214#0:2011|1|11|31
			}
			if (res != null) {
				for (String s : res) {
					// s = s.replace('|', '\t');
					if (s.isEmpty())
						continue;
					writeln(s + "$");
				}
				writeln("*********end|");
			}

		} else if (cmd.equalsIgnoreCase("QUERY")) {
			String toolreqno = readToken();
			String param = readToken();
			if (toolreqno.isEmpty()) {
				writeln("Tool name and query no expected. ");
				return true;
			}
			String[] res = tool.query("QUERY|" + toolreqno, param);
			int count = 0;
			for (String s : res) {
				//s.replace('|', '\t');
				if (s.isEmpty())
					continue;
				writeln(s);
				count++;
			}
			writeln("***END "+count+" RESULTS.");
			/*
			 * QUERY|MeiboTool|216#0:2011|205| ... みなし所属学科の教員の開講科目リスト
			 * QUERY|MeiboTool|219#0:2011|11316139|01| ... 科目の履修者リスト
			 * QUERY|MeiboTool|218#0:2011|636| ... 要件区分の科目リスト 区分は 635 自然科学科目
			 * 自然科学, 636 情報科目 情報科目, 637 対象分野科目 対象分野, 690 全科目（査定内） 査定内科目　（未使用？）
			 * 
			 * QUERY|ReportTool|221#0:2011|82541854| answer =
			 * 2011|11316139|計算量理論|01|82541854|下薗　真一|21|$
			 */
		} else if (cmd.equalsIgnoreCase("suspend")) {
			writeln("SUSPENDED.");
		} else if (cmd.equalsIgnoreCase("quit")) {
			writeln("QUIT WITH ENDING SESSION AND CLOSING CONNECTION.");
			return false;
		}
		return true;
	}
	
}
