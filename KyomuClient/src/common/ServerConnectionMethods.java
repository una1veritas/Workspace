package common;

import clients.*;
import java.util.*;
//import java.awt.*;
//import java.awt.event.*;
//import java.net.*;
import java.io.*;
//import javax.net.ssl.SSLSocketFactory;
import javax.swing.*;

//import javax.swing.border.*;

public class ServerConnectionMethods {
	private CommonInfo commonInfo;
	// private InputStream istream;
	// private OutputStream ostream;
	private BufferedReader cin;
	private PrintWriter cout;
	private boolean qualified = false;

	public ServerConnectionMethods(CommonInfo commonInfo,
	// InputStream istream,
	// OutputStream ostream,
			BufferedReader cin, PrintWriter cout) {
		this.commonInfo = commonInfo;
		// this.istream = istream;
		// this.ostream = ostream;
		this.cin = cin;
		this.cout = cout;
	}

	public void setQualification(boolean b) {
		qualified = b;
	}

	public String getSyllabusXml(String paramValues) {
		if (!qualified) {
			commonInfo
					.showMessage("User is not qualified to get data from DB.");
			System.exit(0);
		}
		commonInfo.timerRestart();

		String param = paramValues.trim();
		if (param.equals("")) {
			param = "empty";
		}
		String msg = "SENDSYLLABUS|SYLLABUS|XML:" + param;
		System.out.println("getSyllabusXml = " + msg);
		String lineSeparator = commonInfo.lineSeparator;

		try {
			cout.println(msg);
			StringBuffer sbuf = new StringBuffer();
			String line;
			while ((line = cin.readLine()) != null) {
				if (line.equals("."))
					break;
				sbuf.append(line).append(lineSeparator);
			}
			String text = sbuf.toString();
			System.out.println("answer = "+text);
			if (text.startsWith("null")) {
				return null;
			} else if (text.startsWith("ERROR")) {
				commonInfo.showMessage(text);
				return null;
			} else {
				return text;
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
			return null;
		}
	}

	public int updateSyllabusXml(String paramValues, String text) {
		if (!qualified) {
			commonInfo
					.showMessage("User is not qualified to get data from DB.");
			System.exit(0);
		}
		commonInfo.timerRestart();

		String msg = "SAVESYLLABUS|SYLLABUS|XML:" + paramValues;
//		String lineSeparator = commonInfo.lineSeparator;

		try {
			cout.println(msg);
			cout.println(text);
			cout.println(".");

			String ans = cin.readLine();
			if (ans.equals("null")) {
				return 0;
			} else if (ans.startsWith("ERROR")) {
				commonInfo.showMessage(ans);
				return 0;
			} else {
				int cnt = Integer.parseInt(ans.trim());
				return cnt;
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
			return 0;
		}
	}

	public int cancelRegistration(String paramValues) {
		if (!qualified) {
			commonInfo
					.showMessage("User is not qualified to cancel registration.");
			System.exit(0);
		}
		commonInfo.timerRestart();

		String key = "CANCELREGISTR|RegistrTool|CANCEL";
		String msg = key + ":" + paramValues;

		try {
			cout.println(msg);
			String answer = cin.readLine();
			if (answer.startsWith("ERROR")) {
				int idx = answer.indexOf("$");
				commonInfo.showMessageLong(answer.substring(idx + 1));
				return 0;
			} else if (answer.equals("0")) {
				return 0;
			} else {
				return 1;
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
			return 0;
		}
	}

	public int registrSubject(String paramValues) {
		if (!qualified) {
			commonInfo
					.showMessage("User is not qualified to registrate subject.");
			System.exit(0);
		}
		commonInfo.timerRestart();

		String key = "ADDREGISTR|RegistrTool|ADD";
		String msg = key + ":" + paramValues;
//		int cnt = 0;

		try {
			cout.println(msg);
			String answer = cin.readLine();
			if (answer.startsWith("ERROR")) {
				int idx = answer.indexOf("$");
				commonInfo.showMessageLong(answer.substring(idx + 1));
				return 0;
			} else if (answer.equals("0")) {
				return 0;
			} else {
				return 1;
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
			return 0;
		}
	}

	public int resetAttendControlParamsOfAllSubjects(String paramValues) {
		commonInfo.timerRestart();

		String key = "RESETATTENDCONTROLPARAMS|ATTENDCONTROL|ALLSUBJECTS";
		String msg = key + ":" + paramValues;
//		int cnt = 0;

		try {
			cout.println(msg);
			String answer = cin.readLine();
			if (answer.startsWith("ERROR")) {
				int idx = answer.indexOf("$");
				commonInfo.showMessageLong(answer.substring(idx + 1));
				return 0;
			} else if (answer.equals("0")) {
				return 0;
			} else {
				return 1;
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
			return 0;
		}
	}

	public int resetAttendControlParams(String paramValues) {
		commonInfo.timerRestart();

		String key = "RESETATTENDCONTROLPARAMS|ATTENDCONTROL|SELECTEDSUBJECT";
		String msg = key + ":" + paramValues;
//		int cnt = 0;

		try {
			cout.println(msg);
			String answer = cin.readLine();
			if (answer.startsWith("ERROR")) {
				int idx = answer.indexOf("$");
				commonInfo.showMessageLong(answer.substring(idx + 1));
				return 0;
			} else if (answer.equals("0")) {
				return 0;
			} else {
				return 1;
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
			return 0;
		}
	}

	public int setClassWeek(String paramValues) {
		commonInfo.timerRestart();

		String key = "UPDATECALENDAR|CLASSWEEK|ADD";
		String msg = key + ":" + paramValues;
//		int cnt = 0;

		try {
			cout.println(msg);
			String answer = cin.readLine();
			if (answer.startsWith("ERROR")) {
				int idx = answer.indexOf("$");
				commonInfo.showMessageLong(answer.substring(idx + 1));
				return 0;
			} else if (answer.equals("0")) {
				return 0;
			} else {
				return 1;
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
			return 0;
		}
	}

	public int deleteClassWeek(String paramValues) {
		commonInfo.timerRestart();

		String key = "UPDATECALENDAR|CLASSWEEK|DELETE";
		String msg = key + ":" + paramValues;
//		int cnt = 0;

		try {
			cout.println(msg);
			String answer = cin.readLine();
			if (answer.startsWith("ERROR")) {
				int idx = answer.indexOf("$");
				commonInfo.showMessageLong(answer.substring(idx + 1));
				return 0;
			} else if (answer.equals("0")) {
				return 0;
			} else {
				return 1;
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
			return 0;
		}
	}

	public int addHolidayInfo(String paramValues) {
		commonInfo.timerRestart();

		String key = "UPDATECALENDAR|HOLIDAYINFO|ADD";
		String msg = key + ":" + paramValues;
//		int cnt = 0;

		try {
			cout.println(msg);
			String answer = cin.readLine();
			if (answer.startsWith("ERROR")) {
				int idx = answer.indexOf("$");
				commonInfo.showMessageLong(answer.substring(idx + 1));
				return 0;
			} else if (answer.equals("0")) {
				return 0;
			} else {
				return 1;
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
			return 0;
		}
	}

	public int deleteHolidayInfo(String paramValues) {
		commonInfo.timerRestart();

		String key = "UPDATECALENDAR|HOLIDAYINFO|DELETE";
		String msg = key + ":" + paramValues;
//		int cnt = 0;

		try {
			cout.println(msg);
			String answer = cin.readLine();
			if (answer.startsWith("ERROR")) {
				int idx = answer.indexOf("$");
				commonInfo.showMessageLong(answer.substring(idx + 1));
				return 0;
			} else if (answer.equals("0")) {
				return 0;
			} else {
				return 1;
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
			return 0;
		}
	}

	public int addSchoolEvent(String paramValues) {
		commonInfo.timerRestart();

		String key = "UPDATECALENDAR|SCHOOLEVENT|ADD";
		String msg = key + ":" + paramValues;
//		int cnt = 0;

		try {
			cout.println(msg);
			String answer = cin.readLine();
			if (answer.startsWith("ERROR")) {
				int idx = answer.indexOf("$");
				commonInfo.showMessageLong(answer.substring(idx + 1));
				return 0;
			} else if (answer.equals("0")) {
				return 0;
			} else {
				return 1;
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
			return 0;
		}
	}

	public int deleteSchoolEvent(String paramValues) {
		commonInfo.timerRestart();

		String key = "UPDATECALENDAR|SCHOOLEVENT|DELETE";
		String msg = key + ":" + paramValues;
//		int cnt = 0;

		try {
			cout.println(msg);
			String answer = cin.readLine();
			if (answer.startsWith("ERROR")) {
				int idx = answer.indexOf("$");
				commonInfo.showMessageLong(answer.substring(idx + 1));
				return 0;
			} else if (answer.equals("0")) {
				return 0;
			} else {
				return 1;
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
			return 0;
		}
	}

	public void copyCurriculumToNextYear(int fromYear, int toYear) {
		commonInfo.timerRestart();

		String msg = "COPYCURRICULUMTONEXTYEAR|CURRICULUM|NEXTYEAR:" + fromYear
				+ "|" + toYear;
//		int cnt = 0;

		try {
			cout.println(msg);
			String answer = cin.readLine();
			if (answer.startsWith("ERROR")) {
				int idx = answer.indexOf("$");
				commonInfo.showMessageLong(answer.substring(idx + 1));
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
		}
	}

	public void copyEduCurriculumToNextYear(int fromYear, int toYear) {
		commonInfo.timerRestart();

		String msg = "COPYEDUCURRICULUMTONEXTYEAR|EDUCURRICULUM|NEXTYEAR:"
				+ fromYear + "|" + toYear;
//		int cnt = 0;

		try {
			cout.println(msg);
			String answer = cin.readLine();
			if (answer.startsWith("ERROR")) {
				int idx = answer.indexOf("$");
				commonInfo.showMessageLong(answer.substring(idx + 1));
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
		}
	}

	public void copyClassInfoToNextYear(int fromYear, int toYear) {
		commonInfo.timerRestart();

		String msg = "COPYCLASSINFOTONEXTYEAR|CLASSINFO|NEXTYEAR:" + fromYear
				+ "|" + toYear;
//		int cnt = 0;

		try {
			cout.println(msg);
			String answer = cin.readLine();
			if (answer.startsWith("ERROR")) {
				int idx = answer.indexOf("$");
				commonInfo.showMessageLong(answer.substring(idx + 1));
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
		}
	}

	public void copyJikanwariToNextYear(int fromYear, int toYear) {
		commonInfo.timerRestart();

		String msg = "COPYJIKANWARITONEXTYEAR|JIKANWARI|NEXTYEAR:" + fromYear
				+ "|" + toYear;
//		int cnt = 0;

		try {
			cout.println(msg);
			String answer = cin.readLine();
			if (answer.startsWith("ERROR")) {
				int idx = answer.indexOf("$");
				commonInfo.showMessageLong(answer.substring(idx + 1));
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
		}
	}

	public void copyJikanwariOverlapToNextYear(int fromYear, int toYear) {
		commonInfo.timerRestart();

		String msg = "COPYJIKANWARIOVERLAPTONEXTYEAR|JIKANWARIOVERLAP|NEXTYEAR:"
				+ fromYear + "|" + toYear;
//		int cnt = 0;

		try {
			cout.println(msg);
			String answer = cin.readLine();
			if (answer.startsWith("ERROR")) {
				int idx = answer.indexOf("$");
				commonInfo.showMessageLong(answer.substring(idx + 1));
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
		}
	}

	public void copyGradYokenNextYear(int fromYear, int toYear) {
		commonInfo.timerRestart();

		String msg = "COPYGRADYOKENTONEXTYEAR|GRADYOKEN|NEXTYEAR:" + fromYear
				+ "|" + toYear;
//		int cnt = 0;

		try {
			cout.println(msg);
			String answer = cin.readLine();
			if (answer.startsWith("ERROR")) {
				int idx = answer.indexOf("$");
				commonInfo.showMessageLong(answer.substring(idx + 1));
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
		}
	}

	public void copyEdYokenToNextYear(int fromYear, int toYear) {
		commonInfo.timerRestart();

		String msg = "COPYEDUYOKENTONEXTYEAR|EDUYOKEN|NEXTYEAR:" + fromYear
				+ "|" + toYear;
//		int cnt = 0;

		try {
			cout.println(msg);
			String answer = cin.readLine();
			if (answer.startsWith("ERROR")) {
				int idx = answer.indexOf("$");
				commonInfo.showMessageLong(answer.substring(idx + 1));
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
		}
	}

	public void copyModuleDefToNextYear(int fromYear, int toYear) {
		commonInfo.timerRestart();

		String msg = "COPYMODULEDEFTONEXTYEAR|MODULEDEF|NEXTYEAR:" + fromYear
				+ "|" + toYear;
//		int cnt = 0;

		try {
			cout.println(msg);
			String answer = cin.readLine();
			if (answer.startsWith("ERROR")) {
				int idx = answer.indexOf("$");
				commonInfo.showMessageLong(answer.substring(idx + 1));
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
		}
	}

	public void copyIIFCurriculumToNextYear(int fromYear, int toYear) {
		commonInfo.timerRestart();

		String msg = "COPYIIFCURRICULUMTONEXTYEAR|IIFCURRICULUM|NEXTYEAR:"
				+ fromYear + "|" + toYear;
//		int cnt = 0;

		try {
			cout.println(msg);
			String answer = cin.readLine();
			if (answer.startsWith("ERROR")) {
				int idx = answer.indexOf("$");
				commonInfo.showMessageLong(answer.substring(idx + 1));
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
		}
	}

	public void copyClassTimeZoneToNextYear(int fromYear, int toYear) {
		commonInfo.timerRestart();

		String msg = "COPYCLASSTIMEZONETONEXTYEAR|CLASSTIMEZONE|NEXTYEAR:"
				+ fromYear + "|" + toYear;
//		int cnt = 0;

		try {
			cout.println(msg);
			String answer = cin.readLine();
			if (answer.startsWith("ERROR")) {
				int idx = answer.indexOf("$");
				commonInfo.showMessageLong(answer.substring(idx + 1));
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
		}
	}

	public void copyYomikaeToNextYear(int fromYear, int toYear) {
		commonInfo.timerRestart();

		String msg = "COPYYOMIKAETONEXTYEAR|YOMIKAE|NEXTYEAR:" + fromYear + "|"
				+ toYear;
//		int cnt = 0;

		try {
			cout.println(msg);
			String answer = cin.readLine();
			if (answer.startsWith("ERROR")) {
				int idx = answer.indexOf("$");
				commonInfo.showMessageLong(answer.substring(idx + 1));
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
		}
	}

	public void copySyllabusToNextYear(int fromYear, int toYear) {
		commonInfo.timerRestart();

		String msg = "COPYSYLLABUSTONEXTYEAR|COPYSYLLABUS|NEXTYEAR:" + fromYear
				+ "|" + toYear;
//		int cnt = 0;

		try {
			cout.println(msg);
			String answer = cin.readLine();
			if (answer.startsWith("ERROR")) {
				int idx = answer.indexOf("$");
				commonInfo.showMessageLong(answer.substring(idx + 1));
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
		}
	}

	// Syllabus Tool

	public int setTeacherEnglishName(String paramValues) {
		// paramValues = SUBJECT_CODE|TEACHER_CODE|TEACHER_NAME_ENGLISH
		if (!qualified) {
			commonInfo
					.showMessage("User is not qualified to set English Name.");
			System.exit(0);
		}
		commonInfo.timerRestart();
		String key = "SYLLABUSTOOL|SyllabusTool|SETTEACHERENGLISHNAME";
		String msg = key + ":" + paramValues;
//		int cnt = 0;

		try {
			cout.println(msg);
			String answer = cin.readLine();
			if (answer.startsWith("ERROR")) {
				int idx = answer.indexOf("$");
				commonInfo.showMessageLong(answer.substring(idx + 1));
				return 0;
			} else if (answer.equals("0")) {
				return 0;
			} else {
				return 1;
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
			return 0;
		}
	}

	public int setSubjectEnglishName(String paramValues) {
		// param = SUBJECT_CODE|TEACHER_CODE|SUBJECT_NAME_ENGLISH
		if (!qualified) {
			commonInfo
					.showMessage("User is not qualified to set English Name.");
			System.exit(0);
		}
		commonInfo.timerRestart();
		String key = "SYLLABUSTOOL|SyllabusTool|SETSUBJECTENGLISHNAME";
		String msg = key + ":" + paramValues;
//		 cnt = 0;

		try {
			cout.println(msg);
			String answer = cin.readLine();
			if (answer.startsWith("ERROR")) {
				int idx = answer.indexOf("$");
				commonInfo.showMessageLong(answer.substring(idx + 1));
				return 0;
			} else if (answer.equals("0")) {
				return 0;
			} else {
				return 1;
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
			return 0;
		}
	}

	// Jikanwari Control

	public int addSubjectClassToJikanwari(String paramValues) {
		// paramValues =
		// SCHOOL_YEAR|FACULTY|SEMESTER|GAKUNEN|WEEK|DEPARTMENT|HOUR|SUBJECT_CODE|CLASS_CODE|
		if (!qualified) {
			commonInfo.showMessage("User is not qualified to edit Jikanwari.");
			System.exit(0);
		}
		commonInfo.timerRestart();
		String key = "JIKANWARICONTROL|JikanwariControl|ADDSUBJECTCLASSTOJIKANWARI";
		String msg = key + ":" + paramValues;
//		int cnt = 0;

		try {
			cout.println(msg);
			String answer = cin.readLine();
			if (answer.startsWith("ERROR")) {
				int idx = answer.indexOf("$");
				commonInfo.showMessageLong(answer.substring(idx + 1));
				return 0;
			} else if (answer.equals("0")) {
				return 0;
			} else {
				return 1;
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
			return 0;
		}
	}

	public int deleteSubjectClassFromJikanwari(String paramValues) {
		// paramValues =
		// SCHOOL_YEAR|FACULTY|SEMESTER|GAKUNEN|WEEK|SUBJECT_CODE|CLASS_CODE|DEPARTMENT|HOUR|
		if (!qualified) {
			commonInfo.showMessage("User is not qualified to edit Jikanwari.");
			System.exit(0);
		}
		commonInfo.timerRestart();
		String key = "JIKANWARICONTROL|JikanwariControl|DELETESUBJECTCLASSFROMJIKANWARI";
		String msg = key + ":" + paramValues;
//		int cnt = 0;

		try {
			cout.println(msg);
			String answer = cin.readLine();
			if (answer.startsWith("ERROR")) {
				int idx = answer.indexOf("$");
				commonInfo.showMessageLong(answer.substring(idx + 1));
				return 0;
			} else if (answer.equals("0")) {
				return 0;
			} else {
				return 1;
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
			return 0;
		}
	}

	public int deleteSubjectClassFromJikanwari2(String paramValues) {
		// paramValues =
		// SCHOOL_YEAR|FACULTY|SEMESTER|GAKUNEN|WEEK|SUBJECT_CODE|CLASS_CODE|DEPARTMENT|HOUR|
		if (!qualified) {
			commonInfo.showMessage("User is not qualified to edit Jikanwari.");
			System.exit(0);
		}
		commonInfo.timerRestart();
		String key = "JIKANWARICONTROL|JikanwariControl|DELETESUBJECTCLASSFROMJIKANWARI2";
		String msg = key + ":" + paramValues;
//		int cnt = 0;

		try {
			cout.println(msg);
			String answer = cin.readLine();
			if (answer.startsWith("ERROR")) {
				int idx = answer.indexOf("$");
				commonInfo.showMessageLong(answer.substring(idx + 1));
				return 0;
			} else if (answer.equals("0")) {
				return 0;
			} else {
				return 1;
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
			return 0;
		}
	}

	public int setRoomToSubjectClassOfJikanwari(String paramValues) {
		// paramValues =
		// SCHOOL_YEAR|FACULTY|SEMESTER|GAKUNEN|WEEK|SUBJECT_CODE|CLASS_CODE|DEPARTMENT|HOUR|ROOM|
		if (!qualified) {
			commonInfo.showMessage("User is not qualified to edit Jikanwari.");
			System.exit(0);
		}
		commonInfo.timerRestart();
		String key = "JIKANWARICONTROL|JikanwariControl|SETROOMTOSUBJECTCLASSOFJIKANWARI";
		String msg = key + ":" + paramValues;
		int cnt = 0;

		try {
			cout.println(msg);
			String answer = cin.readLine();

			if (answer.startsWith("ERROR")) {
				int idx = answer.indexOf("$");
				commonInfo.showMessageLong(answer.substring(idx + 1));
				return 0;
			} else if (answer.equals("0")) {
				return 0;
			} else {
				return 1;
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
			return 0;
		}
	}

	public int setRoomToSubjectClassOfJikanwari2(String paramValues) {
		// paramValues =
		// SCHOOL_YEAR|FACULTY|SEMESTER|GAKUNEN|WEEK|SUBJECT_CODE|CLASS_CODE|DEPARTMENT|HOUR|ROOM|
		if (!qualified) {
			commonInfo.showMessage("User is not qualified to edit Jikanwari.");
			System.exit(0);
		}
		commonInfo.timerRestart();
		String key = "JIKANWARICONTROL|JikanwariControl|SETROOMTOSUBJECTCLASSOFJIKANWARI2";
		String msg = key + ":" + paramValues;
//		int cnt = 0;

		try {
			cout.println(msg);
			String answer = cin.readLine();

			if (answer.startsWith("ERROR")) {
				int idx = answer.indexOf("$");
				commonInfo.showMessageLong(answer.substring(idx + 1));
				return 0;
			} else if (answer.equals("0")) {
				return 0;
			} else {
				return 1;
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
			return 0;
		}
	}

	public int addSubjectClassToShuchuList(String paramValues) {
		// paramValues =
		// SCHOOL_YEAR|FACULTY|SEMESTER|DEPARTMENT|GAKUNEN|SUBJECT_CODE|CLASS_CODE|
		if (!qualified) {
			commonInfo.showMessage("User is not qualified to edit Jikanwari.");
			System.exit(0);
		}
		commonInfo.timerRestart();
		String key = "JIKANWARICONTROL|JikanwariControl|ADDSUBJECTCLASSTOSHUCHULIST";
		String msg = key + ":" + paramValues;
//		int cnt = 0;

		try {
			cout.println(msg);
			String answer = cin.readLine();
			if (answer.startsWith("ERROR")) {
				int idx = answer.indexOf("$");
				commonInfo.showMessageLong(answer.substring(idx + 1));
				return 0;
			} else if (answer.equals("0")) {
				return 0;
			} else {
				return 1;
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
			return 0;
		}
	}

	public int addSubjectClassToGraduateJikanwari(String paramValues) {
		// param =
		// SCHOOL_YEAR|FACULTY|SEMESTER|GAKUNEN|WEEK|DEPARTMENT|HOUR|SUBJECT_CODE|CLASS_CODE|
		if (!qualified) {
			commonInfo.showMessage("User is not qualified to edit Jikanwari.");
			System.exit(0);
		}
		commonInfo.timerRestart();
		String key = "JIKANWARICONTROL|JikanwariControl|ADDSUBJECTCLASSTOGRADUATEJIKANWARI";
		String msg = key + ":" + paramValues;
//		int cnt = 0;

		try {
			cout.println(msg);
			String answer = cin.readLine();
			if (answer.startsWith("ERROR")) {
				int idx = answer.indexOf("$");
				commonInfo.showMessageLong(answer.substring(idx + 1));
				return 0;
			} else if (answer.equals("0")) {
				return 0;
			} else {
				return 1;
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
			return 0;
		}
	}

	public int addSubjectClassToGraduateShuchuList(String paramValues) {
		// param =
		// SCHOOL_YEAR|FACULTY|SEMESTER|DEPARTMENT|GAKUNEN|SUBJECT_CODE|CLASS_CODE|
		if (!qualified) {
			commonInfo.showMessage("User is not qualified to edit Jikanwari.");
			System.exit(0);
		}
		commonInfo.timerRestart();
		String key = "JIKANWARICONTROL|JikanwariControl|ADDSUBJECTCLASSTOGRADUATESHUCHULIST";
		String msg = key + ":" + paramValues;
//		int cnt = 0;

		try {
			cout.println(msg);
			String answer = cin.readLine();
			if (answer.startsWith("ERROR")) {
				int idx = answer.indexOf("$");
				commonInfo.showMessageLong(answer.substring(idx + 1));
				return 0;
			} else if (answer.equals("0")) {
				return 0;
			} else {
				return 1;
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
			return 0;
		}
	}

	public int deleteSubjectClassFromShuchuList(String paramValues) {
		// param =
		// SCHOOL_YEAR|FACULTY|SEMESTER|DEPARTMENT|GAKUNEN|SUBJECT_CODE|CLASS_CODE|
		if (!qualified) {
			commonInfo.showMessage("User is not qualified to edit Jikanwari.");
			System.exit(0);
		}
		commonInfo.timerRestart();
		String key = "JIKANWARICONTROL|JikanwariControl|DELETESUBJECTCLASSFROMSHUCHULIST";
		String msg = key + ":" + paramValues;
//		int cnt = 0;

		try {
			cout.println(msg);
			String answer = cin.readLine();
			if (answer.startsWith("ERROR")) {
				int idx = answer.indexOf("$");
				commonInfo.showMessageLong(answer.substring(idx + 1));
				return 0;
			} else if (answer.equals("0")) {
				return 0;
			} else {
				return 1;
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
			return 0;
		}
	}

	public int deleteSubjectClassFromShuchuList2(String paramValues) {
		// param =
		// SCHOOL_YEAR|FACULTY|SEMESTER|DEPARTMENT|GAKUNEN|SUBJECT_CODE|CLASS_CODE|
		if (!qualified) {
			commonInfo.showMessage("User is not qualified to edit Jikanwari.");
			System.exit(0);
		}
		commonInfo.timerRestart();
		String key = "JIKANWARICONTROL|JikanwariControl|DELETESUBJECTCLASSFROMSHUCHULIST2";
		String msg = key + ":" + paramValues;
//		int cnt = 0;

		try {
			cout.println(msg);
			String answer = cin.readLine();
			if (answer.startsWith("ERROR")) {
				int idx = answer.indexOf("$");
				commonInfo.showMessageLong(answer.substring(idx + 1));
				return 0;
			} else if (answer.equals("0")) {
				return 0;
			} else {
				return 1;
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
			return 0;
		}
	}

	public int updateGakusekiData(String paramValues) {
		if (!qualified) {
			commonInfo.showMessage("User is not qualified to update GAKUSEKI.");
			System.exit(0);
		}
		commonInfo.timerRestart();

		String key = "UPDATEGAKUSEKI|GAKUSEKI|UPDATE";
		String msg = key + ":" + paramValues;
//		int cnt = 0;

		try {
			cout.println(msg);
			String answer = cin.readLine();
			if (answer.startsWith("ERROR")) {
				int idx = answer.indexOf("$");
				commonInfo.showMessageLong(answer.substring(idx + 1));
				return 0;
			} else if (answer.equals("0")) {
				return 0;
			} else {
				return 1;
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
			return 0;
		}
	}

	public int ninteiToSeiseki(String param) {
		commonInfo.timerRestart();
		try {
			String msg = "NINTEITOSEISEKI|NINTEI|INSERT:" + param;
			cout.println(msg);
			String text = cin.readLine();

			if (text.equals("null")) {
				return 0;
			} else if (text.startsWith("ERROR")) {
				commonInfo.showMessage(text);
				return 0;
			} else {
				int cnt = Integer.parseInt(text.trim());
				return cnt;
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
			return 0;
		}
	}

	/*
	 * public int resetTransferFile() { commonInfo.timerRestart();
	 * 
	 * try { String msg = "RESETTRANSFERFILE|RESET|RESET:dummy";
	 * cout.println(msg); String text = cin.readLine(); if (text.equals("null"))
	 * { return 0; } else if (text.startsWith("ERROR")) {
	 * commonInfo.showMessage(text); return 0; } else { int cnt =
	 * Integer.parseInt(text.trim()); return cnt; } } catch (Exception e) {
	 * commonInfo.showMessage( e.toString() ); return 0; } }
	 * 
	 * public int ftpTransferFile() { commonInfo.timerRestart(); try { String
	 * msg = "FTPTRANSFERFILE|FTP|FTP:dummy"; cout.println(msg); String text =
	 * cin.readLine();
	 * 
	 * if (text.equals("null")) { return 0; } else if (text.startsWith("ERROR"))
	 * { commonInfo.showMessage(text); return 0; } else { int cnt =
	 * Integer.parseInt(text.trim()); return cnt; } } catch (Exception e) {
	 * commonInfo.showMessage( e.toString() ); return 0; } }
	 * 
	 * 
	 * public int appendTransferFile(String items) { commonInfo.timerRestart();
	 * try { String msg = "APPENDTOTRANSFERFILE|TRANSFER|TRANSFER:" + items;
	 * cout.println(msg); String text = cin.readLine();
	 * 
	 * if (text.equals("null")) { return 0; } else if (text.startsWith("ERROR"))
	 * { commonInfo.showMessage(text); return 0; } else { int cnt =
	 * Integer.parseInt(text.trim()); return cnt; } } catch (Exception e) {
	 * commonInfo.showMessage( e.toString() ); return 0; } }
	 */

	public int rtorokuToSeiseki(String param) {
		commonInfo.timerRestart();
		try {
			String msg = "KIMATSUREPORTTOSEISEKI|KIMATSU|UPDATE:" + param;
			cout.println(msg);
			String text = cin.readLine();

			if (text.equals("null")) {
				return 0;
			} else if (text.startsWith("ERROR")) {
				commonInfo.showMessage(text);
				return 0;
			} else {
				int cnt = Integer.parseInt(text.trim());
				return cnt;
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
			return 0;
		}
	}

	public int saishiToSeiseki(String param) {
		commonInfo.timerRestart();
		try {
			String msg = "SAISHIREPORTTOSEISEKI|SAISHI|UPDATE:" + param;
			cout.println(msg);
			String text = cin.readLine();

			if (text.equals("null")) {
				return 0;
			} else if (text.startsWith("ERROR")) {
				commonInfo.showMessage(text);
				return 0;
			} else {
				int cnt = Integer.parseInt(text.trim());
				return cnt;
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
			return 0;
		}
	}

	public int saveKimatsuData(String key, ArrayList<String> list) {
		if (!qualified) {
			commonInfo
					.showMessage("User is not qualified to save Seiseki data.");
			System.exit(0);
		}
		commonInfo.timerRestart();

		try {
			String msg = "SAVEKIMATSU|KIMATSU|DATA:" + key;
			cout.println(msg);
			for (int i = 0; i < list.size(); i++) {
				String line = (String) list.get(i);
				cout.println(line);
			}
			cout.println(".");
			String text = cin.readLine();

			if (text.equals("null")) {
				return 0;
			} else if (text.startsWith("ERROR")) {
				commonInfo.showMessage(text);
				return 0;
			} else {
				int cnt = Integer.parseInt(text.trim());
				return cnt;
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
			return 0;
		}
	}

	public int reportKimatsuData(String key, ArrayList<String> list) {
		if (!qualified) {
			commonInfo
					.showMessage("User is not qualified to save Seiseki data.");
			System.exit(0);
		}
		commonInfo.timerRestart();

		try {
			String msg = "REPORTKIMATSU|KIMATSU|DATA:" + key;
			cout.println(msg);
			for (int i = 0; i < list.size(); i++) {
				String line = (String) list.get(i);
				cout.println(line);
			}
			cout.println(".");
			String text = cin.readLine();

			if (text.equals("null")) {
				return 0;
			} else if (text.startsWith("ERROR")) {
				commonInfo.showMessage(text);
				return 0;
			} else {
				int cnt = Integer.parseInt(text.trim());
				return cnt;
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
			return 0;
		}
	}

	public int saveSaishiData(String key, ArrayList<String> list) {
		if (!qualified) {
			commonInfo
					.showMessage("User is not qualified to save Seiseki data.");
			System.exit(0);
		}
		commonInfo.timerRestart();

		try {
			String msg = "SAVESAISHI|SAISHI|DATA:" + key;
			cout.println(msg);
			for (int i = 0; i < list.size(); i++) {
				String line = (String) list.get(i);
				cout.println(line);
			}
			cout.println(".");
			String text = cin.readLine();

			if (text.equals("null")) {
				return 0;
			} else if (text.startsWith("ERROR")) {
				commonInfo.showMessage(text);
				return 0;
			} else {
				int cnt = Integer.parseInt(text.trim());
				return cnt;
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
			return 0;
		}
	}

	public int reportSaishiData(String key, ArrayList<String> list) {
		if (!qualified) {
			commonInfo
					.showMessage("User is not qualified to save Seiseki data.");
			System.exit(0);
		}
		commonInfo.timerRestart();

		try {
			String msg = "REPORTSAISHI|SAISHI|DATA:" + key;
			cout.println(msg);
			for (int i = 0; i < list.size(); i++) {
				String line = (String) list.get(i);
				cout.println(line);
			}
			cout.println(".");
			String text = cin.readLine();

			if (text.equals("null")) {
				return 0;
			} else if (text.startsWith("ERROR")) {
				commonInfo.showMessage(text);
				return 0;
			} else {
				int cnt = Integer.parseInt(text.trim());
				return cnt;
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
			return 0;
		}
	}

	public int reportNinteiData(String key, ArrayList<String> list) {
		if (!qualified) {
			commonInfo
					.showMessage("User is not qualified to save Nintei data.");
			System.exit(0);
		}
		commonInfo.timerRestart();

		try {
			String msg = "REPORTNINTEI|NINTEI|DATA:" + key;
			cout.println(msg);
			for (int i = 0; i < list.size(); i++) {
				String line = (String) list.get(i);
				cout.println(line);
			}
			cout.println(".");
			String text = cin.readLine();

			if (text.equals("null")) {
				return 0;
			} else if (text.startsWith("ERROR")) {
				commonInfo.showMessage(text);
				return 0;
			} else {
				int cnt = Integer.parseInt(text.trim());
				return cnt;
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
			return 0;
		}
	}

	public String getStudentGakuseki(String key, String studentCode) {
		if (!qualified) {
			commonInfo
					.showMessage("User is not qualified to get data from DB.");
			System.exit(0);
		}
		commonInfo.timerRestart();

		String msg = key + ":" + studentCode;

		try {
			cout.println(msg);
			String answer = cin.readLine();
			if (answer.equals("null")) {
				return null;
			} else if (answer.startsWith("ERROR")) {
				commonInfo.showMessage(answer);
				return null;
			} else {
				return answer;
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
			return null;
		}
	}

	public String getStaffName(String key, String staffCode) {
		String param = staffCode.trim();
		if (param.equals("")) {
			return null;
		}
		String msg = key + ":" + param;

		try {
			cout.println(msg);
			return cin.readLine();
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
			return null;
		}
	}

	public ImageIcon getStudentPhoto(String key, String studentCode) {
		if (!qualified) {
			commonInfo
					.showMessage("User is not qualified to get data from DB.");
			System.exit(0);
		}
		commonInfo.timerRestart();

		String param = studentCode.trim();
		if (param.equals("")) {
			param = "empty";
		}
		String msg = key + ":" + param;

		try {
			cout.println(msg);
			String answer = cin.readLine();
			int size = Integer.parseInt(answer);

			byte[] imageData = new byte[size];
			for (int i = 0; i < size; i++) {
				imageData[i] = (byte) cin/* istream */.read();
			}

			ImageIcon icon;
			try {
				icon = new ImageIcon(imageData);
			} catch (Exception e) {
				icon = null;
			}
			return icon;
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
			return null;
		}
	}

}
