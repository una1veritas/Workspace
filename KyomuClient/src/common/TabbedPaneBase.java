package common;

import clients.*;
import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.*;

public class TabbedPaneBase extends JTabbedPane {
	private String serviceName;
	private String nodePath;
	private CommonInfo commonInfo;
	private TabbedPaneBase parentTabbedPane;
	private String methodInvokedWhenSelected;
	private String oldServiceName;
	private String oldNodePath;
	private int oldIndex = -1;
	private int defaultOpenIndex = 0;

	private String[] tabItemKey = { "CHILD_NODE_ID", "CHILD_TAB_TITLE",
			"CHILD_TAB_FG_COLOR", "CHILD_TAB_BG_COLOR", "CHILD_COMPONENT_TYPE",
			"CHILD_PANEL_ID", "CHILD_METHOD_WHEN_SELECTED",
			"CHILD_QUALIFICATION_METHOD", "CHILD_QUALIFICATION_PARAM" };

	private ArrayList<Map<String, String>> childTabInfoList = new ArrayList<Map<String, String>>();

	public TabbedPaneBase(String serviceName, String nodePath,
			CommonInfo commonInfo, TabbedPaneBase parentTabbedPane,
			String methodInvokedWhenSelected) {
		this.serviceName = serviceName;
		this.nodePath = nodePath;
		this.commonInfo = commonInfo;
		this.parentTabbedPane = parentTabbedPane;
		this.methodInvokedWhenSelected = methodInvokedWhenSelected;

		makeChildTabInfoList();

		for (int i = 0; i < childTabInfoList.size(); i++) {
			Map<String, String> tabItemMap = childTabInfoList.get(i);
			String tabTitle = tabItemMap.get("CHILD_TAB_TITLE");
			Color fgColor = commonInfo.getTabFgColor(tabItemMap
					.get("CHILD_TAB_FG_COLOR"));
			Color bgColor = commonInfo.getTabBgColor(tabItemMap
					.get("CHILD_TAB_BG_COLOR"));
			String nodeID = tabItemMap.get("CHILD_NODE_ID");
			nodeIDToTabIndexMap.put(nodeID, i);

			if (tabTitle != null) {
				String qualificationMethod = tabItemMap
						.get("CHILD_QUALIFICATION_METHOD");
				String qualificationParam = tabItemMap
						.get("CHILD_QUALIFICATION_PARAM");
				boolean tabEnabled = commonInfo.commonInfoMethods
						.checkQualification(qualificationMethod,
								qualificationParam, parentTabbedPane);
				if (tabEnabled) {
					addTab(tabTitle, null);
					setForegroundAt(i, fgColor);
					setBackgroundAt(i, bgColor);
					setEnabledAt(i, true);
				} else {
					addTab("", null);
					setForegroundAt(i, fgColor);
					setBackgroundAt(i, bgColor);
					setEnabledAt(i, false);
				}
			} else {
				addTab("", null);
				setBackgroundAt(i, fgColor);
				setForegroundAt(i, bgColor);
				setEnabledAt(i, false);
			}
		}
		setFont(new Font("DialogInput", Font.PLAIN, 12));
		addChangeListener(changeListener);

		if (methodInvokedWhenSelected != null) {
			invokeMethodWhenSelected(methodInvokedWhenSelected);
		}

		oldIndex = -1;
		setSelectedIndex(oldIndex);
	}

	private void makeChildTabInfoList() {
		String childTabsInfo = commonInfo.getTabbedPaneStruct(serviceName,
				nodePath);
		if (childTabsInfo != null) {
			String[] tabInfoArray = childTabsInfo.split("\\$");
			for (String tabInfo : tabInfoArray) {
				String[] tabItemVal = tabInfo.split("\\|");
				if (tabItemVal.length == 9) {
					HashMap<String, String> map = new HashMap<String, String>();
					for (int i = 0; i < 9; i++) {
						String key = tabItemKey[i];
						String val = tabItemVal[i].trim();
						if (val.equals(""))
							val = null;
						map.put(key, val);
					}
					childTabInfoList.add(map);
				} else {
					commonInfo
							.showMessageLong("TabbedPaneStruct: $ format error $ "
									+ tabInfo);
				}
			}
		} else {
			commonInfo
					.showMessageLong("TabbedPaneStruct not found: $ serviceName = "
							+ serviceName + " $ nodePath = " + nodePath);
		}
	}

	public void invokeMethodWhenSelected(String methodName) {
		try {
			getClass().getMethod(methodName, (java.lang.Class[]) null).invoke(
					this, (java.lang.Object[]) null);
		} catch (Exception ex) {
			commonInfo
					.showMessageLong("TabbedPaneBase: $ MethodInvokedWhenSelected is not invoked $ "
							+ methodName);
		}
	}

	public void pageOpened() {
		if (oldIndex == -1) {
			setSelectedIndex(defaultOpenIndex);
		} else {
			pageOpened(oldIndex);
		}
	}

	public void setOldIndex(int index) {
		oldIndex = index;
	}

	public void pageOpened(int index) {
		String childServiceName;
		String childNodePath;

		if (index < 0)
			return;

		Map<String, String> tabItemMap = childTabInfoList.get(index);
		if (tabItemMap != null) {
			String childNodeID = tabItemMap.get("CHILD_NODE_ID");
			String childComponentType = tabItemMap.get("CHILD_COMPONENT_TYPE");
			String childPanelID = tabItemMap.get("CHILD_PANEL_ID");
			String childMethodWhenSelected = tabItemMap
					.get("CHILD_METHOD_WHEN_SELECTED");

			if (!(childNodeID.startsWith("/"))) {
				childServiceName = serviceName;
				childNodePath = nodePath + "." + childNodeID;
			} else {
				int pos = childNodeID.indexOf(".");
				childServiceName = childNodeID.substring(1, pos);
				childNodePath = childNodeID.substring(pos + 1);
			}

			if (index != oldIndex) {
				if (oldIndex >= 0) {
					addColumnCodeMap("OLD_TAB_INDEX", "" + oldIndex, ""
							+ oldIndex);
					nodeIDToTabIndexMap.put("OLD_TAB_INDEX", oldIndex);
					oldServiceName = commonInfo.getPresentServiceName();
					oldNodePath = commonInfo.getPresentNodePath();
				}
				addColumnCodeMap("PRESENT_TAB_INDEX", "" + index, "" + index);
				commonInfo.setPresentServiceName(childServiceName);
				commonInfo.setPresentNodePath(childNodePath);

				if (childComponentType.equals("TabbedPane")) {
					Component oldTab = getComponentAt(index);
					if ((oldTab == null) || (!(oldTab instanceof JTabbedPane))) {
						TabbedPaneBase tab = new TabbedPaneBase(
								childServiceName, childNodePath, commonInfo,
								this, childMethodWhenSelected);
						setComponentAt(index, tab);
						tab.pageOpened();
					} else {
						TabbedPaneBase tab = (TabbedPaneBase) getComponentAt(index);
						if (childMethodWhenSelected != null) {
							tab.invokeMethodWhenSelected(childMethodWhenSelected);
						}
						tab.pageOpened();
					}
				} else if (childComponentType.equals("DataPanel")) {
					Component oldTab = getComponentAt(index);
					if ((oldTab == null) || (!(oldTab instanceof JPanel))) {
						DataPanelBase tab = new DataPanelBase(childServiceName,
								childNodePath, childPanelID, commonInfo, this,
								childMethodWhenSelected);
						setComponentAt(index, tab);
						tab.pageOpened();
					} else {
						DataPanelBase tab = (DataPanelBase) getComponentAt(index);
						if (childMethodWhenSelected != null) {
							tab.invokeMethodWhenSelected(childMethodWhenSelected);
						}
						tab.pageOpened();
					}
				} else if (childComponentType.equals("SplitPane")) {
					SplitPaneBase tab = new SplitPaneBase(childServiceName,
							childNodePath, commonInfo, this,
							childMethodWhenSelected);
					setComponentAt(index, tab);
					tab.pageOpened();
				} else {
					commonInfo
							.showMessage("TabbedPaneBase : unknown CHILD_COMPONENT_TYPE: "
									+ childComponentType);
				}
			}

			if (oldIndex >= 0) {
				Map<String, String> tabItemMap2 = childTabInfoList
						.get(oldIndex);
				if (tabItemMap2 != null) {
					String oldComponentType = tabItemMap2
							.get("CHILD_COMPONENT_TYPE");
					if (oldComponentType.equals("TabbedPane")) {
						Component oldTab = getComponentAt(index);
						removeChangeListener(changeListener);
						setComponentAt(oldIndex, new JPanel());
						addChangeListener(changeListener);
						oldTab = null;
					}
				}
			}
		}

		commonInfo.timerRestart();
		oldIndex = index;
	}

	// --------------------------------------------------------------
	// the following method is invoked by TableView of DataPanel
	// to open new page (of this TabbedPane) identified by the nodeID.
	// --------------------------------------------------------------

	private HashMap<String, Integer> nodeIDToTabIndexMap = new HashMap<String, Integer>();

	public void openTab(String nodeID) {
		if (nodeIDToTabIndexMap.containsKey(nodeID)) {
			int newTabIndex = nodeIDToTabIndexMap.get(nodeID);
			setSelectedIndex(newTabIndex);
		}
	}

	public void openOldTab() {
		if (nodeIDToTabIndexMap.containsKey("OLD_TAB_INDEX")) {
			int oldTabIndex = Integer
					.parseInt(getValueFromColumnCodeMap("OLD_TAB_INDEX"));
			setSelectedIndex(oldTabIndex);
		}
	}

	// --------------------------------------------------------------
	// the following maps store correspondance ( columnCode => columnValue )
	// and ( columnCode => columnDisplay )
	// --------------------------------------------------------------

	private HashMap<String, String> columnCodeToValueMap = new HashMap<String, String>();
	private HashMap<String, String> columnCodeToDisplayMap = new HashMap<String, String>();

	public void addColumnCodeMap(String columnCode, String value, String display) {
		if (columnCode != null) {
			if (value == null)
				value = " ";
			columnCodeToValueMap.put(columnCode, value);
			if (display == null)
				display = " ";
			columnCodeToDisplayMap.put(columnCode, display);
		}
	}

	public boolean columnCodeMapContains(String columnCode) {
		return columnCodeToValueMap.containsKey(columnCode);
	}

	public String getValueFromColumnCodeMap(String columnCode) {
		return columnCodeToValueMap.get(columnCode);
	}

	public String getDisplayFromColumnCodeMap(String columnCode) {
		return columnCodeToDisplayMap.get(columnCode);
	}

	// --------------------------------------------------------------
	// the following inner class (which implements ChangeListener)
	// defines tab_open procedure invoked by 'tab select action'
	// --------------------------------------------------------------

	private TabChangeListener changeListener = new TabChangeListener();

	class TabChangeListener implements ChangeListener {

		public void stateChanged(ChangeEvent e) {
			JTabbedPane tab = (JTabbedPane) e.getSource();
			int index = tab.getSelectedIndex();

			if (commonInfo.unsavedDataExists()) {
				Object[] obj = { " まだ「セーブ」されていない入力データが存在します。", " ",
						"  入力したデータを捨てて次の画面に進みますか？    ==> 了解 ",
						"  　　　　　　　      　元の画面に戻りますか？    ==> 取消 ", " " };
				int ans = JOptionPane.showConfirmDialog(commonInfo.getFrame(),
						obj, "セーブしてないデータ", JOptionPane.OK_CANCEL_OPTION);
				if (ans == JOptionPane.OK_OPTION) {
					commonInfo.setUnsavedDataExistFlag(false);
					pageOpened(index);
				} else {
					commonInfo.setUnsavedDataExistFlag(false);
					setSelectedIndex(oldIndex);
					commonInfo.setUnsavedDataExistFlag(true);
				}
			} else {
				pageOpened(index);
			}
		}
	}

	// --------------------------------------------------------------
	// the following ArrayList is used to pass a list of student info
	// from one DataPanel (ex. which displays studentList)
	// to another DataPanel (ex. which displays studentPhoto or mailAddressList)
	// located under this TabbedPane
	// --------------------------------------------------------------

	public ArrayList<String> studentListToPass = new ArrayList<String>();

	public void setStudentListToPass(ArrayList<String> list) {
		studentListToPass = list;
	}

	public ArrayList<String> getStudentListToPass() {
		return studentListToPass;
	}

	// --------------------------------------------------------------
	// the following methods (methodWhenSelected) are methods
	// invoked by invokeMethodWhenSelected(String methodName)
	// --------------------------------------------------------------

	public void cloneColumnCodeMap() {
		if (parentTabbedPane != null) {
			Set<String> set = parentTabbedPane.columnCodeToValueMap.keySet();
			for (String columnCode : set) {
				String value = parentTabbedPane
						.getValueFromColumnCodeMap(columnCode);
				String display = parentTabbedPane
						.getDisplayFromColumnCodeMap(columnCode);
				addColumnCodeMap(columnCode, value, display);
			}
		}
	}

	public void setCalendarToOpen() {
		int mon = commonInfo.thisMonth;
		if (mon >= 4) {
			defaultOpenIndex = mon - 4;
		} else {
			defaultOpenIndex = mon + 8;
		}
		cloneColumnCodeMap();
		setThisSchoolYear();
	}

	public void setThisSchoolYear() {
		addColumnCodeMap("SCHOOL_YEAR", "" + commonInfo.thisSchoolYear, ""
				+ commonInfo.thisSchoolYear + "年");
	}

	public void setStudentAttrib() {
		addColumnCodeMap("STUDENT_CODE", commonInfo.STUDENT_CODE,
				commonInfo.STUDENT_NAME);
		String faculty = commonInfo.STUDENT_FACULTY;
		addColumnCodeMap("FACULTY", faculty,
				commonInfo.getGakumuCodeShorterName("FACULTY", faculty));
		String dept = commonInfo.STUDENT_DEPARTMENT;
		addColumnCodeMap("DEPARTMENT", dept,
				commonInfo.getGakumuCodeShorterName("DEPARTMENT", dept));
		String course = commonInfo.STUDENT_COURSE;
		addColumnCodeMap("COURSE", course,
				commonInfo.getGakumuCodeShorterName("COURSE", course));
		String gakunen = commonInfo.STUDENT_GAKUNEN;
		addColumnCodeMap("GAKUNEN", gakunen, gakunen + "年");
		addColumnCodeMap("CURRICULUM_YEAR", commonInfo.STUDENT_CURRICULUM_YEAR,
				commonInfo.STUDENT_CURRICULUM_YEAR + "年");
		addColumnCodeMap("SUPERVISOR", commonInfo.STUDENT_SUPERVISOR,
				commonInfo.STUDENT_SUPERVISOR_NAME);
		addColumnCodeMap("SCHOOL_YEAR", "" + commonInfo.thisSchoolYear, ""
				+ commonInfo.thisSchoolYear + "年");
		addColumnCodeMap("SEMESTER", "" + commonInfo.thisSemester,
				commonInfo.thisSemesterName);
	}

	public void setStudentJikanwariInfo() {
		addColumnCodeMap("SCHOOL_YEAR", "" + commonInfo.thisSchoolYear, ""
				+ commonInfo.thisSchoolYear + "年");
		addColumnCodeMap("THIS_SCHOOL_YEAR_FOR_JIKANWARI", ""
				+ commonInfo.thisSchoolYear, "" + commonInfo.thisSchoolYear
				+ "年");
		addColumnCodeMap("SEMESTER", "" + commonInfo.thisSemester,
				commonInfo.thisSemesterName);
		String faculty = commonInfo.STUDENT_FACULTY;
		String department = commonInfo.STUDENT_DEPARTMENT;
		String gakunen = commonInfo.STUDENT_GAKUNEN;
		if (!faculty.equals("11")) {
			faculty = "32";
			department = "70";
			gakunen = "1";
		}
		addColumnCodeMap("FACULTY", faculty,
				commonInfo.getGakumuCodeShorterName("FACULTY", faculty));
		addColumnCodeMap("DEPARTMENT", department,
				commonInfo.getGakumuCodeShorterName("DEPARTMENT", department));
		addColumnCodeMap("GAKUNEN", gakunen, gakunen + "年");
		defaultOpenIndex = 1;
	}

	public void setJikanwariThisYear() {
		addColumnCodeMap("THIS_SCHOOL_YEAR_FOR_JIKANWARI", ""
				+ commonInfo.thisSchoolYear, "" + commonInfo.thisSchoolYear
				+ "年");
	}

	public void setJikanwariNextYear() {
		addColumnCodeMap("THIS_SCHOOL_YEAR_FOR_JIKANWARI", ""
				+ (commonInfo.thisSchoolYear + 1), ""
				+ (commonInfo.thisSchoolYear + 1) + "年");
	}

	public void cloneColumnCodeMapAndSetPageToOpen() {
		if (parentTabbedPane != null) {
			cloneColumnCodeMap();
			if (commonInfo.STUDENT_FACULTY.equals("11")) {
				String dept = getValueFromColumnCodeMap("DEPARTMENT");
				if (commonInfo.STUDENT_DEPARTMENT.equals(dept)) {
					defaultOpenIndex = 1;
				} else {
					defaultOpenIndex = 0;
				}
			} else {
				defaultOpenIndex = 0;
			}
		}
	}

	public void cloneColumnCodeMapAndSetViewTypeToOpen() {
		if (parentTabbedPane != null) {
			cloneColumnCodeMap();
			String viewType = getValueFromColumnCodeMap("DATA_VIEW_TYPE");
			if (viewType.equals("SimpleTableView")) {
				defaultOpenIndex = 0;
			} else if (viewType.equals("VarTableView")) {
				defaultOpenIndex = 1;
			} else if (viewType.equals("JikanwariView")) {
				defaultOpenIndex = 2;
			} else if (viewType.equals("HtmlView")) {
				defaultOpenIndex = 3;
			} else {
				defaultOpenIndex = 0;
			}
		}
	}

	public void setStudentAttribAndSetSeisekiRTorokuInfo() {
		commonInfo.commonInfoMethods.registrInfo.showRegistrWarning();
		setStudentAttrib();
		commonInfo.commonInfoMethods.registrInfo.setSeisekiRTorokuInfo();
		String faculty = commonInfo.STUDENT_FACULTY;
		String department = commonInfo.STUDENT_DEPARTMENT;
		String gakunen = commonInfo.STUDENT_GAKUNEN;
		if (!faculty.equals("11")) {
			faculty = "32";
			department = "70";
			gakunen = "1";
		}
		addColumnCodeMap("FACULTY", faculty,
				commonInfo.getGakumuCodeShorterName("FACULTY", faculty));
		addColumnCodeMap("DEPARTMENT", department,
				commonInfo.getGakumuCodeShorterName("DEPARTMENT", department));
		addColumnCodeMap("GAKUNEN", gakunen, gakunen + "年");
		defaultOpenIndex = 1;
	}

	public void setStudentAttribAndSetJikanwariInfo() {
		setStudentAttrib();
		setStudentJikanwariInfo();
	}
}
