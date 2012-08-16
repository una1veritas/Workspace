package TeacherServerNew;

import common.*;
import java.net.*;
import java.io.*;
import java.util.*;
import java.awt.*;
import java.sql.*;
import syllabusNew.*;

public class TeacherHandlerMethods {
	private CommonInfo commonInfo;
	private Socket socket;
	private InputStream istream;
	private OutputStream ostream;
	private BufferedReader cin;
	private PrintWriter cout;

	private JDBCConnection jdbc;
	private SyllabusControl syllabus;
	private Person me;
	private String PASSWORD = "";
	private boolean qualified = false;

	public TeacherHandlerMethods(CommonInfo commonInfo, Socket socket,
			InputStream istream, OutputStream ostream, BufferedReader cin,
			PrintWriter cout) {
		this.commonInfo = commonInfo;
		this.socket = socket;
		this.istream = istream;
		this.ostream = ostream;
		this.cin = cin;
		this.cout = cout;
		jdbc = (JDBCConnection) commonInfo.jdbc;
		syllabus = commonInfo.syllabus;
		me = new Person();
	}

	protected void sendPasswordCheckResult(String paramValues) {
		String userID;
		String rawPassword;
		String ans = "";
		String STAFF_CODE = "";
		String PASSWORD_STATUS = "";
		String QUALIFICATION = "";
		String inetAddress = socket.getInetAddress().getHostAddress();

		String[] tokens = paramValues.split("\\|");
		userID = tokens[0];
		rawPassword = tokens[1];

		try {
			String quest = "select STAFF_CODE, PASSWORD, PASSWORD_STATUS, QUALIFICATION from TEACHER.STAFF_PASSWD where STAFF_ID = '"
					+ userID + "'";
			ans = jdbc.executeKyomuQuery(quest);

			if (ans == null) {
				cout.println("たぶん USER_ID が間違っています。");
				qualified = false;
				return;
			} else {
				String[] tokens2 = ans.split("\\|");
				STAFF_CODE = tokens2[0].trim();
				PASSWORD = tokens2[1].trim();
				PASSWORD_STATUS = tokens2[2].trim();
				QUALIFICATION = tokens2[3].trim();
			}

			if (PASSWORD_STATUS.equals("9")) {
				cout.println("利用停止中: 学務係に申し出て利用停止を解除して下さい。");
				qualified = false;
				return;
			}

			int qual = Integer.parseInt(QUALIFICATION);
			/*
			 * if (qual < 2) { cout.println("利用不可:このツールを利用するには専任講師以上の権限が必要です。");
			 * qualified = false; return; }
			 */

			int cnt = getUserMapCount(userID);
			if (cnt > 4) {
				cout.println("パスワードの入力ミスが規定回数を超えたため本日中はログインできません。");
				qualified = false;
				return;
			}
			String salt = PASSWORD.substring(0, 2);
			String cryptPassword = commonInfo.crypt(rawPassword, salt);
			if (cryptPassword.equals(PASSWORD)) {
				appendLog("1", userID, STAFF_CODE, QUALIFICATION, inetAddress);
				setUserMapCount(userID, 0);
				me.USER_ID = userID;
				me.STAFF_CODE = STAFF_CODE;
				me.QUALIFICATION = QUALIFICATION;
				me.inetAddress = inetAddress;

				String query = "select STAFF_TYPE, STAFF_OCCUPATION, STAFF_STATUS, STAFF_ATTRIB, LOCAL_ATTRIB, STAFF_NAME, MAIL_ADDRESS from MASTER.STAFF where STAFF_CODE = '"
						+ me.STAFF_CODE + "'";
				ans = jdbc.executeKyomuQuery(query);
				if (ans != null) {
					String[] tokens3 = ans.split("\\|");
					me.STAFF_TYPE = tokens3[0].trim();
					me.STAFF_OCCUPATION = tokens3[1].trim();
					me.STAFF_STATUS = tokens3[2].trim();
					me.STAFF_ATTRIB = tokens3[3].trim();
					me.LOCAL_ATTRIB = tokens3[4].trim();
					me.STAFF_NAME = tokens3[5].trim();
					me.MAIL_ADDRESS = tokens3[6].trim();
				}

				if ((me.STAFF_ATTRIB.equals("081"))
						|| (me.STAFF_ATTRIB.equals("235"))) {
					if (me.LOCAL_ATTRIB.equals("知能情報")) {
						// me.STAFF_ATTRIB = "055";
						me.STAFF_ATTRIB = "205";
						me.STAFF_DEPARTMENT = "31";
					} else if (me.LOCAL_ATTRIB.equals("電子情報")) {
						// me.STAFF_ATTRIB = "057";
						me.STAFF_ATTRIB = "210";
						me.STAFF_DEPARTMENT = "32";
					} else if (me.LOCAL_ATTRIB.equals("創成情報")) {
						// me.STAFF_ATTRIB = "059";
						me.STAFF_ATTRIB = "215";
						me.STAFF_DEPARTMENT = "33";
					} else if (me.LOCAL_ATTRIB.equals("機械情報")) {
						// me.STAFF_ATTRIB = "061";
						me.STAFF_ATTRIB = "220";
						me.STAFF_DEPARTMENT = "34";
					} else if (me.LOCAL_ATTRIB.equals("生命情報")) {
						// me.STAFF_ATTRIB = "063";
						me.STAFF_ATTRIB = "225";
						me.STAFF_DEPARTMENT = "35";
					} else if (me.LOCAL_ATTRIB.equals("人間科学")) {
						// me.STAFF_ATTRIB = "065";
						me.STAFF_ATTRIB = "230";
						me.STAFF_DEPARTMENT = "30";
					}
				} else if ((me.STAFF_ATTRIB.equals("055"))
						|| (me.STAFF_ATTRIB.equals("205"))) {
					me.STAFF_DEPARTMENT = "31";
				} else if ((me.STAFF_ATTRIB.equals("057"))
						|| (me.STAFF_ATTRIB.equals("210"))) {
					me.STAFF_DEPARTMENT = "32";
				} else if ((me.STAFF_ATTRIB.equals("059"))
						|| (me.STAFF_ATTRIB.equals("215"))) {
					me.STAFF_DEPARTMENT = "33";
				} else if ((me.STAFF_ATTRIB.equals("061"))
						|| (me.STAFF_ATTRIB.equals("220"))) {
					me.STAFF_DEPARTMENT = "34";
				} else if ((me.STAFF_ATTRIB.equals("063"))
						|| (me.STAFF_ATTRIB.equals("225"))) {
					me.STAFF_DEPARTMENT = "35";
				} else if ((me.STAFF_ATTRIB.equals("065"))
						|| (me.STAFF_ATTRIB.equals("230"))) {
					me.STAFF_DEPARTMENT = "30";
				}
				commonInfo.addLoginMap(me);
				cout.println("success|" + STAFF_CODE);
				qualified = true;
				return;
			} else {
				cnt++;
				setUserMapCount(userID, cnt);
				cout.println("パスワードが正しくありません。");
				appendLog("0", userID, STAFF_CODE, QUALIFICATION, inetAddress);
				qualified = false;
				return;
			}
		} catch (Exception e) {
			cout.println("ERROR: " + e.toString().trim());
			qualified = false;
			return;
		}
	}

	protected void sendPasswordChangeResult(String paramValues) {
		try {
			if (!qualified) {
				cout.println("ERROR: ユーザ認証に合格していません。");
				return;
			}
			String[] tokens = paramValues.split("\\|");
			String userID = tokens[0];
			String oldPasswd = tokens[1];
			String newPasswd = tokens[2];

			String salt = PASSWORD.substring(0, 2);
			String cryptPassword = commonInfo.crypt(oldPasswd, salt);
			if ((cryptPassword.equals(PASSWORD)) && (userID.equals(me.USER_ID))) {
				salt = newPasswd.substring(0, 2);
				cryptPassword = commonInfo.crypt(newPasswd, salt);
				String update = "update TEACHER.STAFF_PASSWD set PASSWORD = '"
						+ cryptPassword
						+ "', PASSWORD_STATUS = '1', REVISED_DATE = sysdate where STAFF_ID = '"
						+ userID + "'";
				int res = jdbc.executeKyomuUpdate(update);
				if (res == 1) {
					cout.println("success");
				} else {
					cout.println("パスワードの変更に失敗しました。");
				}
			} else {
				cout.println("USER_ID または PASSWORD が間違っています。");
			}
		} catch (Exception e) {
			cout.println("ERROR： " + e.toString().trim());
		}
	}

	protected void sendCommonQueryResult(String commandCode, String paramValues) {
		try {
			if (paramValues.equals("empty")) {
				paramValues = null;
			}
			String result = jdbc.getCommonQueryResult(commandCode, paramValues);
			if (result != null) {
				cout.println(result);
			} else {
				cout.println("null");
			}
		} catch (Exception e) {
			cout.println("ERROR$" + e.toString().trim());
		}
	}

	protected void sendQueryResult(String serviceName, String commandCode,
			String paramValues) {
		try {
			if (qualified) {
				if (paramValues.equals("empty")) {
					paramValues = null;
				}
				// commandCode = panelID+"#"+switchCode;
				String[] tokens = commandCode.split("\\#");
				String panelID = tokens[0];
				String switchCode = tokens[1];
				String str = jdbc.getQueryResult(serviceName, panelID,
						switchCode, paramValues, me);
				if (str == null) {
					cout.println("null");
				} else {
					cout.println(str);
				}
			} else {
				cout.println("ERROR: ユーザ認証に合格していません。");
			}
		} catch (Exception e) {
			cout.println("ERROR$" + e.toString().trim());
		}
	}

	protected void sendDeleteResult(String serviceName, String commandCode,
			String paramValues) {
		try {
			if (qualified) {
				if (paramValues.equals("empty")) {
					paramValues = null;
				}
				int res = jdbc.deleteCommand(serviceName, commandCode,
						paramValues, me);
				cout.println("" + res);
			} else {
				cout.println("ERROR: ユーザ認証に合格していません。");
			}
		} catch (Exception e) {
			cout.println("ERROR$" + e.toString().trim());
		}
	}

	protected void sendUpdateResult(String serviceName, String commandCode,
			String paramValues) {
		try {
			if (qualified) {
				if (paramValues.equals("empty")) {
					paramValues = null;
				}
				int res = jdbc.updateCommand(serviceName, commandCode,
						paramValues, me);
				cout.println("" + res);
			} else {
				cout.println("ERROR: ユーザ認証に合格していません。");
			}
		} catch (Exception e) {
			cout.println("ERROR$" + e.toString().trim());
		}
	}

	protected void sendInsertResult(String serviceName, String commandCode,
			String paramValues) {
		try {
			if (qualified) {
				if (paramValues.equals("empty")) {
					paramValues = null;
				}
				int res = jdbc.insertCommand(serviceName, commandCode,
						paramValues, me);
				cout.println("" + res);
			} else {
				cout.println("ERROR: ユーザ認証に合格していません。");
			}
		} catch (Exception e) {
			cout.println("ERROR$" + e.toString().trim());
		}
	}

	protected void sendSpecialResult(String serviceName, String commandCode,
			String paramValues) {
		try {
			if (qualified) {
				if (paramValues.equals("empty")) {
					paramValues = null;
				}
				int res = jdbc.specialCommand(serviceName, commandCode,
						paramValues, me);
				cout.println("" + res);
			} else {
				cout.println("ERROR: ユーザ認証に合格していません。");
			}
		} catch (Exception e) {
			cout.println("ERROR$" + e.toString().trim());
		}
	}

	protected void sendQueryStaffAttribResult(String paramValues) {
		try {
			if (!qualified) {
				cout.println("ERROR: ユーザ認証に合格していません。");
				return;
			}
			String mailAddress = me.MAIL_ADDRESS;
			String localAttrib = me.LOCAL_ATTRIB;
			if ((mailAddress == null) || (mailAddress.equals(""))) {
				mailAddress = " ";
			}
			if ((localAttrib == null) || (localAttrib.equals(""))) {
				localAttrib = " ";
			}

			StringBuffer sbuf = new StringBuffer();
			sbuf.append(me.STAFF_CODE).append("|");
			sbuf.append(me.STAFF_NAME).append("|");
			sbuf.append(me.STAFF_ATTRIB).append("|");
			sbuf.append(me.QUALIFICATION).append("|");
			sbuf.append(mailAddress).append("|");
			sbuf.append(localAttrib).append("|");
			sbuf.append(me.STAFF_DEPARTMENT).append("|");
			cout.println(sbuf.toString());
		} catch (Exception e) {
			cout.println("ERROR： " + e.toString().trim());
		}
	}

	protected void sendSyllabusXml(String paramValues) {
		try {
			String[] tokens = paramValues.split("\\|");
			String schoolYear = tokens[0].trim();
			String subjectCode = tokens[1].trim();
			String teacherCode = tokens[2].trim();

			String str = syllabus.getElementXml(schoolYear, teacherCode,
					subjectCode);
			if (str != null) {
				cout.println(str);
				cout.println(".");
			} else {
				cout.println("null");
				cout.println(".");
			}
		} catch (Exception e) {
			cout.println("ERROR$" + e);
			cout.println(".");
		}
	}

	protected void saveSyllabusXml(String paramValues) {
		try {
			String[] tokens = paramValues.split("\\|");
			String schoolYear = tokens[0].trim();
			String subjectCode = tokens[1].trim();
			String teacherCode = tokens[2].trim();

			if (schoolYear.trim().equals("")) {
				cout.println("ERROR$ 開講年度が指定されていません。 ");
				return;
			}
			if (subjectCode.trim().equals("")) {
				cout.println("ERROR$ 科目が指定されていません。 ");
				return;
			}
			if (teacherCode.trim().equals("")) {
				cout.println("ERROR$ 担当教官が指定されていません。 ");
				return;
			}

			String line;
			StringBuffer sbuf = new StringBuffer();
			while ((line = cin.readLine()) != null) {
				if (line.equals("."))
					break;
				sbuf.append(line).append("\n");
			}
			String xmlText = sbuf.toString();

			if (qualifiedToUpdateSyllabus(teacherCode, me)) {
				syllabus.updateElement(schoolYear, xmlText);
				cout.println("1");
			} else {
				cout.println("ERROR$ あなたは教授要目を変更する資格がありません。");
			}
		} catch (Exception e) {
			cout.println("ERROR$" + e);
		}
	}

	protected void syllabusTool(String commandCode, String paramValues) {
		String update = "";
		if (commandCode.equals("SETTEACHERENGLISHNAME")) {
			try {
				String[] tokens = paramValues.split("\\|");
				String subjectCode = tokens[0].trim();
				String teacherCode = tokens[1].trim();
				String teacherEnglishName = tokens[2].trim();
				if (teacherEnglishName.equals("")) {
					update = "update MASTER.STAFF set ENGLISH_NAME = null where STAFF_CODE = '"
							+ teacherCode + "'";
				} else {
					update = "update MASTER.STAFF set ENGLISH_NAME = '"
							+ teacherEnglishName + "' where STAFF_CODE = '"
							+ teacherCode + "'";
				}
				int res = jdbc.executeKyomuUpdate(update);
				if (res <= 0) {
					cout.println("0");
				} else {
					cout.println("1");
				}
				return;
			} catch (Exception e) {
				cout.println("0");
			}
		} else if (commandCode.equals("SETSUBJECTENGLISHNAME")) {
			try {
				String[] tokens = paramValues.split("\\|");
				String subjectCode = tokens[0].trim();
				String teacherCode = tokens[1].trim();
				String subjectEnglishName = tokens[2].trim();
				if (subjectEnglishName.equals("")) {
					update = "update MASTER.SUBJECT set ENGLISH_NAME = null where SUBJECT_CODE = '"
							+ subjectCode + "'";
				} else {
					update = "update MASTER.SUBJECT set ENGLISH_NAME = '"
							+ subjectEnglishName + "' where SUBJECT_CODE = '"
							+ subjectCode + "'";
				}
				int res = jdbc.executeKyomuUpdate(update);
				if (res <= 0) {
					cout.println("0");
				} else {
					cout.println("1");
				}
				return;
			} catch (Exception e) {
				cout.println("0");
			}
		}
	}

	protected void copyClassTimeZoneToNextYear(String paramValues) {
		try {
			String[] tokens = paramValues.split("\\|");
			String fromYear = tokens[0].trim();
			String toYear = tokens[1].trim();

			String del = "delete from ATTEND.CLASS_TIME_ZONE where SCHOOL_YEAR = "
					+ toYear;
			String ins = "insert into ATTEND.CLASS_TIME_ZONE ( SCHOOL_YEAR, SEMESTER, HOUR, ATTEND_TIME_START, LATE_TIME_START, LATE_TIME_END, ATTEND_TIME_END ) select '"
					+ toYear
					+ "', SEMESTER, HOUR, ATTEND_TIME_START, LATE_TIME_START, LATE_TIME_END, ATTEND_TIME_END from ATTEND.CLASS_TIME_ZONE where SCHOOL_YEAR =  '"
					+ fromYear + "'";

			int res = jdbc.executeAttendUpdate(del);
			int res2 = jdbc.executeAttendUpdate(ins);
			cout.println("ok");
		} catch (Exception e) {
			cout.println("ERROR： " + e.toString().trim());
		}
	}

	protected void copySyllabusToNextYear(String paramValues) {
		String path = "/home/maginu/KYOMU-INFO/SYLLABUS/";
		try {
			String[] tokens = paramValues.split("\\|");
			String fromYear = tokens[0].trim();
			String toYear = tokens[1].trim();

			HashMap<String, Integer> fileMap = new HashMap<String, Integer>();
			HashMap<String, String> classMap = new HashMap<String, String>();

			File idir = new File(path, fromYear);
			File odir = new File(path, toYear);
			odir.mkdir();

			String[] flist = idir.list();
			for (String fname : flist) {
				File f = new File(idir, fname);
				int size = (int) f.length();
				fileMap.put(fname, size);
			}

			int startYear = Integer.parseInt(fromYear) - 3;
			String query = "select distinct TEACHER_CODE, SUBJECT_CODE from MASTER.CLASS_INFO where SCHOOL_YEAR >= "
					+ startYear;
			String ans = jdbc.executeKyomuQuery(query);
			String[] lines = ans.split("\\$");
			for (String line : lines) {
				String[] tokens2 = line.split("\\|");
				String teacherCode = tokens2[0];
				String subjectCode = tokens2[1];
				if (!teacherCode.equals(" ")) {
					String key = teacherCode + "_" + subjectCode + ".xml";
					classMap.put(key, "1");
				}
			}

			Set<String> keySet = classMap.keySet();
			for (String key : keySet) {
				String classCode = classMap.get(key);
				if (fileMap.containsKey(key)) {
					copyToNextYear(idir, odir, key);
				}
			}
			cout.println("ok");
		} catch (Exception e) {
			cout.println("ERROR： " + e.toString().trim());
		}
	}

	private void copyToNextYear(File idir, File odir, String fname)
			throws IOException {
		InputStream in = new FileInputStream(new File(idir, fname));
		OutputStream out = new FileOutputStream(new File(odir, fname));
		int ch;
		while ((ch = in.read()) != -1) {
			out.write(ch);
		}
		in.close();
		out.close();
	}

	protected void saveKimatsuData(String paramValues) {
		HashMap<String, String> rtorokuMap = new HashMap<String, String>();
		try {
			String[] tokens2 = paramValues.split("\\|");
			String schoolYear = tokens2[0].trim();
			String subjectCode = tokens2[1].trim();
			String classCode = tokens2[2].trim();
			String teacherCode = tokens2[3].trim();
			String msg = "select STUDENT_CODE, FIXED from KYOMU.RTOROKU where SCHOOL_YEAR = '"
					+ schoolYear
					+ "' and SUBJECT_CODE = '"
					+ subjectCode
					+ "' and CLASS_CODE = '" + classCode + "'";
			String res = jdbc.executeKyomuQuery(msg);
			String[] lines = res.split("\\$");
			for (String line : lines) {
				String[] tokens = line.split("\\|");
				String scode = tokens[0];
				String fixed = tokens[1];
				rtorokuMap.put(scode, fixed);
			}
			if (qualifiedToSeisekiReport(teacherCode, me)) {
				int count = 0;
				String line;
				while ((line = cin.readLine()) != null) {
					if (line.equals("."))
						break;
					String[] tokens = line.split("\\|");
					String studentCode = tokens[0];
					String marks = tokens[1];
					String fixed = tokens[2];
					String fixed2 = rtorokuMap.get(studentCode);
					if ((fixed2 == null) || (fixed2.equals("F"))
							|| (fixed2.equals("Y")))
						continue;

					String upd = "update KYOMU.RTOROKU set MARKS = " + marks
							+ ", HOKOKU_DATE = sysdate, FIXED = '" + fixed
							+ "', HOKOKU_BY = '" + me.STAFF_CODE
							+ "' where SCHOOL_YEAR = '" + schoolYear
							+ "' and STUDENT_CODE = '" + studentCode
							+ "' and SUBJECT_CODE = '" + subjectCode
							+ "' and CLASS_CODE = '" + classCode + "'";
					int n = jdbc.executeKyomuUpdate(upd);
					count += n;
				}
				cout.println("" + count);
			} else {
				cout.println("0");
			}
		} catch (Exception e) {
			cout.println("0");
		}
	}

	protected void saveSaishiData(String paramValues) {
		HashMap<String, String> saishiMap = new HashMap<String, String>();

		try {
			String[] tokens2 = paramValues.split("\\|");
			String schoolYear = tokens2[0].trim();
			String subjectCode = tokens2[1].trim();
			String classCode = tokens2[2].trim();
			String teacherCode = tokens2[3].trim();
			String msg = "select STUDENT_CODE, FIXED from KYOMU.SAISHI where SCHOOL_YEAR = '"
					+ schoolYear
					+ "' and SUBJECT_CODE = '"
					+ subjectCode
					+ "' and CLASS_CODE = '" + classCode + "'";
			String res = jdbc.executeKyomuQuery(msg);
			String[] lines = res.split("\\$");
			for (String line : lines) {
				String[] tokens = line.split("\\|");
				String scode = tokens[0];
				String fixed = tokens[1];
				saishiMap.put(scode, fixed);
			}
			if (qualifiedToSeisekiReport(teacherCode, me)) {
				int count = 0;
				String line;
				while ((line = cin.readLine()) != null) {
					if (line.equals("."))
						break;
					String[] tokens = line.split("\\|");
					String studentCode = tokens[0];
					String marks = tokens[1];
					String fixed = tokens[2];
					String fixed2 = saishiMap.get(studentCode);
					if ((fixed2 == null) || (fixed2.equals("F"))
							|| (fixed2.equals("Y")))
						continue;

					String upd = "update KYOMU.SAISHI set MARKS = " + marks
							+ ", HOKOKU_DATE = sysdate, FIXED = '" + fixed
							+ "', HOKOKU_BY = '" + me.STAFF_CODE
							+ "' where SCHOOL_YEAR = '" + schoolYear
							+ "' and STUDENT_CODE = '" + studentCode
							+ "' and SUBJECT_CODE = '" + subjectCode
							+ "' and CLASS_CODE = '" + classCode + "'";
					int n = jdbc.executeKyomuUpdate(upd);
					count += n;
				}
				cout.println("" + count);
			} else {
				cout.println("0");
			}
		} catch (Exception e) {
			cout.println("0");
		}
	}

	protected void reportNinteiData(String paramValues) {
		String studentCode = paramValues;
		try {
			String line;
			int count = 0;
			while ((line = cin.readLine()) != null) {
				if (line.equals("."))
					break;
				String[] tokens = line.split("\\|");
				String subjectCode = tokens[0];
				String qualify = tokens[1];
				String evidence = tokens[2];

				String upd = "";
				if (qualify.equals("N")) {
					upd = "update TEACHER.QUALIFY_INPUT set EVIDENCE = null, QUALIFIED_DATE = null, QUALIFIED_BY = null, QUALIFIED = 'N' where STUDENT_CODE = '"
							+ studentCode
							+ "' and SUBJECT_CODE = '"
							+ subjectCode + "'";
				} else if (qualify.equals("Y")) {
					int index = evidence.indexOf("\n");
					if (index > 0) {
						evidence = evidence.substring(0, index);
					}
					upd = "update TEACHER.QUALIFY_INPUT set EVIDENCE = '"
							+ evidence
							+ "', QUALIFIED_DATE = sysdate, QUALIFIED_BY = '"
							+ me.STAFF_CODE
							+ "', QUALIFIED =  'Y' where STUDENT_CODE = '"
							+ studentCode + "' and SUBJECT_CODE = '"
							+ subjectCode + "'";
				}
				int n = jdbc.executeKyomuUpdate(upd);
				count += n;
			}
			cout.println("" + count);
		} catch (Exception e) {
			cout.println("0");
		}
	}

	protected void resetAttendControlParams(String paramValues) {
		try {
			String[] tokens2 = paramValues.split("\\|");
			String schoolYear = tokens2[0].trim();
			String semester = tokens2[1].trim();
			String subjectCode = tokens2[2].trim();
			String classCode = tokens2[3].trim();

			String del = "delete from ATTEND.CLASS_DATE_CNTRL where SCHOOL_YEAR = '"
					+ schoolYear
					+ "' and SEMESTER = '"
					+ semester
					+ "' and SUBJECT_CODE = '"
					+ subjectCode
					+ "' and CLASS_CODE = '" + classCode + "'";
			int res = jdbc.executeAttendUpdate(del);

			int syear = Integer.parseInt(schoolYear);
			int secondSemesterStartMonth = 9;
			int secondSemesterStartDay = 25;

			String query = "select YEAR, MONTH, DAY, SCHOOL_EVENTS from MASTER.CALENDAR where SCHOOL_EVENTS is not null and ((YEAR = "
					+ syear
					+ " and MONTH >= 4) or (YEAR = ("
					+ syear
					+ " + 1) and MONTH <= 3)) order by 1, 2, 3";
			String ans = jdbc.executeKyomuQuery(query);
			String[] lines = ans.split("\\$");
			for (String line : lines) {
				String[] tokens = line.split("\\|");
				String year = tokens[0];
				String month = tokens[1];
				String day = tokens[2];
				String schoolEvent = tokens[3];
				if (schoolEvent.indexOf("後期履修申告") >= 0) {
					secondSemesterStartMonth = Integer.parseInt(month.trim());
					secondSemesterStartDay = Integer.parseInt(day.trim());
					break;
				} else if (schoolEvent.indexOf("後期授業開始") >= 0) {
					secondSemesterStartMonth = Integer.parseInt(month.trim());
					secondSemesterStartDay = Integer.parseInt(day.trim());
					break;
				}
			}

			if (semester.equals("1")) {
				String semesterStartDate = "" + schoolYear + ":04:01";
				String semesterEndDate;
				if (secondSemesterStartMonth < 10) {
					semesterEndDate = "" + schoolYear + ":0"
							+ secondSemesterStartMonth + ":"
							+ (secondSemesterStartDay - 1);
				} else {
					semesterEndDate = "" + schoolYear + ":"
							+ secondSemesterStartMonth + ":"
							+ (secondSemesterStartDay - 1);
				}
				String ins = "insert into ATTEND.CLASS_DATE_CNTRL (SCHOOL_YEAR, SEMESTER, SUBJECT_CODE, CLASS_CODE, CLASS_DATE, HOUR, ROOM, ATTEND_START_DATE, LATE_START_DATE, LATE_END_DATE, ATTEND_END_DATE, CHECK_FLAG ) select JC.SCHOOL_YEAR, JC.SEMESTER, JC.SUBJECT_CODE, JC.CLASS_CODE, to_date(C.YEAR||':'||C.MONTH||':'||C.DAY, 'YYYY:MM:DD'), JC.HOUR, JC.ROOM, to_date(C.YEAR||':'||C.MONTH||':'||C.DAY||':'||TZ.ATTEND_TIME_START, 'YYYY:MM:DD:HH24:MI'), to_date(C.YEAR||':'||C.MONTH||':'||C.DAY||':'||TZ.LATE_TIME_START, 'YYYY:MM:DD:HH24:MI'), to_date(C.YEAR||':'||C.MONTH||':'||C.DAY||':'||TZ.LATE_TIME_END, 'YYYY:MM:DD:HH24:MI'), to_date(C.YEAR||':'||C.MONTH||':'||C.DAY||':'||TZ.ATTEND_TIME_END, 'YYYY:MM:DD:HH24:MI'), JC.CHECK_FLAG from ATTEND.JIKANWARI_CHECK JC, MASTER.CALENDAR C, ATTEND.CLASS_TIME_ZONE TZ where JC.SCHOOL_YEAR = '"
						+ schoolYear
						+ "' and JC.SEMESTER = '"
						+ semester
						+ "' and JC.SUBJECT_CODE = '"
						+ subjectCode
						+ "' and JC.CLASS_CODE = '"
						+ classCode
						+ "' and JC.WEEK = C.CLASS_WEEK and (to_date(C.YEAR||':'||C.MONTH||':'||C.DAY, 'YYYY:MM:DD') >= to_date('"
						+ semesterStartDate
						+ "', 'YYYY:MM:DD')) and (to_date(C.YEAR||':'||C.MONTH||':'||C.DAY, 'YYYY:MM:DD') <= to_date('"
						+ semesterEndDate
						+ "', 'YYYY:MM:DD')) and TZ.SCHOOL_YEAR = JC.SCHOOL_YEAR and TZ.SEMESTER = JC.SEMESTER and TZ.HOUR = JC.HOUR";
				res = jdbc.executeAttendUpdate(ins);
				cout.println("" + res);
			} else {
				String semesterStartDate;
				if (secondSemesterStartMonth < 10) {
					semesterStartDate = "" + schoolYear + ":0"
							+ secondSemesterStartMonth + ":"
							+ secondSemesterStartDay;
				} else {
					semesterStartDate = "" + schoolYear + ":"
							+ secondSemesterStartMonth + ":"
							+ secondSemesterStartDay;
				}
				String semesterEndDate = "" + (syear + 1) + ":03:31";
				String ins = "insert into ATTEND.CLASS_DATE_CNTRL (SCHOOL_YEAR, SEMESTER, SUBJECT_CODE, CLASS_CODE, CLASS_DATE, HOUR, ROOM, ATTEND_START_DATE, LATE_START_DATE, LATE_END_DATE, ATTEND_END_DATE, CHECK_FLAG ) select JC.SCHOOL_YEAR, JC.SEMESTER, JC.SUBJECT_CODE, JC.CLASS_CODE, to_date(C.YEAR||':'||C.MONTH||':'||C.DAY, 'YYYY:MM:DD'), JC.HOUR, JC.ROOM, to_date(C.YEAR||':'||C.MONTH||':'||C.DAY||':'||TZ.ATTEND_TIME_START, 'YYYY:MM:DD:HH24:MI'), to_date(C.YEAR||':'||C.MONTH||':'||C.DAY||':'||TZ.LATE_TIME_START, 'YYYY:MM:DD:HH24:MI'), to_date(C.YEAR||':'||C.MONTH||':'||C.DAY||':'||TZ.LATE_TIME_END, 'YYYY:MM:DD:HH24:MI'), to_date(C.YEAR||':'||C.MONTH||':'||C.DAY||':'||TZ.ATTEND_TIME_END, 'YYYY:MM:DD:HH24:MI'), JC.CHECK_FLAG from ATTEND.JIKANWARI_CHECK JC, MASTER.CALENDAR C, ATTEND.CLASS_TIME_ZONE TZ where JC.SCHOOL_YEAR = '"
						+ schoolYear
						+ "' and JC.SEMESTER = '"
						+ semester
						+ "' and JC.SUBJECT_CODE = '"
						+ subjectCode
						+ "' and JC.CLASS_CODE = '"
						+ classCode
						+ "' and JC.WEEK = C.CLASS_WEEK and (to_date(C.YEAR||':'||C.MONTH||':'||C.DAY, 'YYYY:MM:DD') >= to_date('"
						+ semesterStartDate
						+ "', 'YYYY:MM:DD')) and (to_date(C.YEAR||':'||C.MONTH||':'||C.DAY, 'YYYY:MM:DD') <= to_date('"
						+ semesterEndDate
						+ "', 'YYYY:MM:DD')) and TZ.SCHOOL_YEAR = JC.SCHOOL_YEAR and TZ.SEMESTER = JC.SEMESTER and TZ.HOUR = JC.HOUR";
				res = jdbc.executeAttendUpdate(ins);
				cout.println("" + res);
			}
		} catch (Exception e) {
			cout.println("ERROR$" + e.toString().trim());
		}
	}

	protected void resetAttendControlParamsOfAll(String paramValues) {
		try {
			String[] tokens2 = paramValues.split("\\|");
			String schoolYear = tokens2[0].trim();
			String semester = tokens2[1].trim();

			String del = "delete from ATTEND.CLASS_DATE_CNTRL where SCHOOL_YEAR = '"
					+ schoolYear + "' and SEMESTER = '" + semester + "'";
			int res = jdbc.executeAttendUpdate(del);

			int syear = Integer.parseInt(schoolYear);
			int secondSemesterStartMonth = 9;
			int secondSemesterStartDay = 25;

			String query = "select YEAR, MONTH, DAY, SCHOOL_EVENTS from MASTER.CALENDAR where SCHOOL_EVENTS is not null and ((YEAR = "
					+ syear
					+ " and MONTH >= 4) or (YEAR = ("
					+ syear
					+ " + 1) and MONTH <= 3)) order by 1, 2, 3";
			String ans = jdbc.executeKyomuQuery(query);
			String[] lines = ans.split("\\$");
			for (String line : lines) {
				String[] tokens = line.split("\\|");
				String year = tokens[0];
				String month = tokens[1];
				String day = tokens[2];
				String schoolEvent = tokens[3];
				if (schoolEvent.indexOf("後期履修申告") >= 0) {
					secondSemesterStartMonth = Integer.parseInt(month.trim());
					secondSemesterStartDay = Integer.parseInt(day.trim());
					break;
				} else if (schoolEvent.indexOf("後期授業開始") >= 0) {
					secondSemesterStartMonth = Integer.parseInt(month.trim());
					secondSemesterStartDay = Integer.parseInt(day.trim());
					break;
				}
			}

			if (semester.equals("1")) {
				String semesterStartDate = "" + schoolYear + ":04:01";
				String semesterEndDate;
				if (secondSemesterStartMonth < 10) {
					semesterEndDate = "" + schoolYear + ":0"
							+ secondSemesterStartMonth + ":"
							+ (secondSemesterStartDay - 1);
				} else {
					semesterEndDate = "" + schoolYear + ":"
							+ secondSemesterStartMonth + ":"
							+ (secondSemesterStartDay - 1);
				}
				String ins = "insert into ATTEND.CLASS_DATE_CNTRL (SCHOOL_YEAR, SEMESTER, SUBJECT_CODE, CLASS_CODE, CLASS_DATE, HOUR, ROOM, ATTEND_START_DATE, LATE_START_DATE, LATE_END_DATE, ATTEND_END_DATE, CHECK_FLAG ) select JC.SCHOOL_YEAR, JC.SEMESTER, JC.SUBJECT_CODE, JC.CLASS_CODE, to_date(C.YEAR||':'||C.MONTH||':'||C.DAY, 'YYYY:MM:DD'), JC.HOUR, JC.ROOM, to_date(C.YEAR||':'||C.MONTH||':'||C.DAY||':'||TZ.ATTEND_TIME_START, 'YYYY:MM:DD:HH24:MI'), to_date(C.YEAR||':'||C.MONTH||':'||C.DAY||':'||TZ.LATE_TIME_START, 'YYYY:MM:DD:HH24:MI'), to_date(C.YEAR||':'||C.MONTH||':'||C.DAY||':'||TZ.LATE_TIME_END, 'YYYY:MM:DD:HH24:MI'), to_date(C.YEAR||':'||C.MONTH||':'||C.DAY||':'||TZ.ATTEND_TIME_END, 'YYYY:MM:DD:HH24:MI'), JC.CHECK_FLAG from ATTEND.JIKANWARI_CHECK JC, MASTER.CALENDAR C, ATTEND.CLASS_TIME_ZONE TZ where JC.SCHOOL_YEAR = '"
						+ schoolYear
						+ "' and JC.SEMESTER = '"
						+ semester
						+ "' and JC.WEEK = C.CLASS_WEEK and (to_date(C.YEAR||':'||C.MONTH||':'||C.DAY, 'YYYY:MM:DD') >= to_date('"
						+ semesterStartDate
						+ "', 'YYYY:MM:DD')) and (to_date(C.YEAR||':'||C.MONTH||':'||C.DAY, 'YYYY:MM:DD') <= to_date('"
						+ semesterEndDate
						+ "', 'YYYY:MM:DD')) and TZ.SCHOOL_YEAR = JC.SCHOOL_YEAR and TZ.SEMESTER = JC.SEMESTER and TZ.HOUR = JC.HOUR";
				res = jdbc.executeAttendUpdate(ins);
				cout.println("" + res);
			} else {
				String semesterStartDate;
				if (secondSemesterStartMonth < 10) {
					semesterStartDate = "" + schoolYear + ":0"
							+ secondSemesterStartMonth + ":"
							+ secondSemesterStartDay;
				} else {
					semesterStartDate = "" + schoolYear + ":"
							+ secondSemesterStartMonth + ":"
							+ secondSemesterStartDay;
				}
				String semesterEndDate = "" + (syear + 1) + ":03:31";
				String ins = "insert into ATTEND.CLASS_DATE_CNTRL (SCHOOL_YEAR, SEMESTER, SUBJECT_CODE, CLASS_CODE, CLASS_DATE, HOUR, ROOM, ATTEND_START_DATE, LATE_START_DATE, LATE_END_DATE, ATTEND_END_DATE, CHECK_FLAG ) select JC.SCHOOL_YEAR, JC.SEMESTER, JC.SUBJECT_CODE, JC.CLASS_CODE, to_date(C.YEAR||':'||C.MONTH||':'||C.DAY, 'YYYY:MM:DD'), JC.HOUR, JC.ROOM, to_date(C.YEAR||':'||C.MONTH||':'||C.DAY||':'||TZ.ATTEND_TIME_START, 'YYYY:MM:DD:HH24:MI'), to_date(C.YEAR||':'||C.MONTH||':'||C.DAY||':'||TZ.LATE_TIME_START, 'YYYY:MM:DD:HH24:MI'), to_date(C.YEAR||':'||C.MONTH||':'||C.DAY||':'||TZ.LATE_TIME_END, 'YYYY:MM:DD:HH24:MI'), to_date(C.YEAR||':'||C.MONTH||':'||C.DAY||':'||TZ.ATTEND_TIME_END, 'YYYY:MM:DD:HH24:MI'), JC.CHECK_FLAG from ATTEND.JIKANWARI_CHECK JC, MASTER.CALENDAR C, ATTEND.CLASS_TIME_ZONE TZ where JC.SCHOOL_YEAR = '"
						+ schoolYear
						+ "' and JC.SEMESTER = '"
						+ semester
						+ "' and JC.WEEK = C.CLASS_WEEK and (to_date(C.YEAR||':'||C.MONTH||':'||C.DAY, 'YYYY:MM:DD') >= to_date('"
						+ semesterStartDate
						+ "', 'YYYY:MM:DD')) and (to_date(C.YEAR||':'||C.MONTH||':'||C.DAY, 'YYYY:MM:DD') <= to_date('"
						+ semesterEndDate
						+ "', 'YYYY:MM:DD')) and TZ.SCHOOL_YEAR = JC.SCHOOL_YEAR and TZ.SEMESTER = JC.SEMESTER and TZ.HOUR = JC.HOUR";
				res = jdbc.executeAttendUpdate(ins);
				cout.println("" + res);
			}
		} catch (Exception e) {
			cout.println("ERROR$" + e.toString().trim());
		}
	}

	protected void sendStudentPhoto(String studentCode) {
		String path = commonInfo.studentPhotoDir + studentCode;
		byte[] data = new byte[60000];
		int len = 0;
		try {
			InputStream fin = new FileInputStream(path);
			len = fin.read(data);
			fin.close();
		} catch (IOException e) {
			len = 0;
		}
		try {
			cout.println("" + len);
			ostream.write(data, 0, len);
			ostream.flush();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	protected void executeControlLoop() {
		String command;
		while (true) {
			try {
				command = cin.readLine();
				if (command == null) {
					break;
				} else {
					if (command.equals("PRINT_LOGIN_LIST")) {
						StringBuffer sbuf = new StringBuffer();
						int num = commonInfo.loginMap.size();
						if (num == 0) {
							cout.println("empty");
						} else {
							Set<String> keySet = commonInfo.loginMap.keySet();
							Iterator<String> it = keySet.iterator();
							while (it.hasNext()) {
								String key = it.next();
								String val = commonInfo.loginMap.get(key);
								sbuf.append(key).append("|").append(val)
										.append("$");
							}
							cout.println(sbuf.toString());
						}
					} else if (command.equals("PRINT_THREAD_COUNT")) {
						int cnt = Thread.activeCount();
						cout.println("activeCount: " + cnt);
					}
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	protected boolean qualifiedToUpdateSyllabus(String teacherCode, Person me) {
		String staffCode = me.STAFF_CODE;
		int qual = 0;
		try {
			qual = Integer.parseInt(me.QUALIFICATION);
		} catch (Exception e) {
		}

		if (staffCode.equals(teacherCode)) {
			return true;
		} else if ((qual >= 3) && (qual != 7)) {
			return true;
		} else {
			return false;
		}
	}

	protected boolean qualifiedToSeisekiReport(String teacherCode, Person me) {
		String staffCode = me.STAFF_CODE;
		int qual = 0;
		try {
			qual = Integer.parseInt(me.QUALIFICATION);
		} catch (Exception e) {
		}

		if (staffCode.equals(teacherCode)) {
			return true;
		} else if ((qual == 8) || (qual == 9)) {
			return true;
		} else {
			return false;
		}
	}

	protected void sendErrorMessage(String message) {
		try {
			cout.println(message);
		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}

	protected int getUserMapCount(String userID) {
		if (commonInfo.userMap.containsKey(userID)) {
			return commonInfo.userMap.get(userID);
		} else {
			return 0;
		}
	}

	protected void setUserMapCount(String userID, int count) {
		if (commonInfo.userMap.containsKey(userID)) {
			commonInfo.userMap.put(userID, count);
		} else {
			commonInfo.userMap.put(userID, 0);
		}
	}

	protected void removeLoginMap() {
		commonInfo.removeLoginMap(me);
	}

	protected void appendLog(String res, String userID, String STAFF_CODE,
			String QUALIFICATION, String inetAddress) {
		try {
			String insert = "insert into TEACHER.TEACHER_LOG (STAFF_ID, STAFF_CODE, QUALIFICATION, LOGIN_DATE, LOGIN_RESULT, INET_ADDRESS) values ('"
					+ userID
					+ "','"
					+ STAFF_CODE
					+ "','"
					+ QUALIFICATION
					+ "', sysdate, '" + res + "','" + inetAddress + "')";
			jdbc.executeKyomuUpdate(insert);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
