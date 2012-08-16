package common;

import clients.*;
import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.table.*;
//import javax.swing.text.*;
import javax.swing.border.*;

//import java.lang.reflect.*;

public class TableViewBase extends JTable {
	protected CommonInfo commonInfo;
	protected TabbedPaneBase parentTabbedPane;
	protected DataPanelBase parentDataPanel;
	protected String serviceName;
	protected String nodePath;
	protected String panelID;
	protected String tableViewType;
	protected TableModelBase tableModel;

	// **** TableViewInfo ****//
	protected String methodWhenOpened;
	protected int fontSize;
	protected int rowHeight;
	protected Color tableFgColor;
	protected Color tableBgColor;
	protected String sorterType;
	protected String selectionMode;
	protected String switchMethod;
	protected String switchMethodParam;
	protected String rowSelectMethod;
	protected String rowSelectMethodParam;

	protected String switchCode = "0"; // default switch code for table query
	protected String queryParams;
	protected String queryParamValues;
	protected String queryParamDisplays;
	protected String oldQueryParamValues = "abdakabudarah";
	protected boolean refreshTableFlag = false;
	protected int selectedRow = -1;

	// **** TableColumnInfo ****//
	public ArrayList<Integer> columnNumberList = new ArrayList<Integer>();
	public ArrayList<String> columnTitleList = new ArrayList<String>();
	public ArrayList<String> columnCodeList = new ArrayList<String>();
	public ArrayList<String> columnDisplayList = new ArrayList<String>();
	public ArrayList<Integer> columnWidthList = new ArrayList<Integer>();
	public ArrayList<Color> columnFgColorList = new ArrayList<Color>();
	public ArrayList<Color> columnBgColorList = new ArrayList<Color>();
	public ArrayList<String> columnRendererList = new ArrayList<String>();
	public int originalColumnCount;

	// **** VarTableViewInfo ****//
	protected String keyColumnCode;
	protected String addColumnDisplay;
	protected int addColumnWidth;
	protected String addColumnRenderer;
	protected String addDataSwitchMethod;
	protected String addDataSwitchParam;

	protected String addColumnSwitchCode = "3"; // default switch code for
												// add_column query
	protected String addColumnQueryParams; // 追加カラムを取得するためのパラメータ
	protected String addColumnQueryParamValues; // 追加カラムを取得するためのパラメータ値
	protected String addDataSwitchCode = "4"; // default switch code for table
												// add_data query
	protected String addDataQueryParams; // 追加データを取得するためのパラメータ
	protected String addDataQueryParamValues; // 追加データを取得するためのパラメータ

	public TableViewBase(String tableViewType, String serviceName,
			String nodePath, String panelID, CommonInfo commonInfo,
			TabbedPaneBase parentTabbedPane, DataPanelBase parentDataPanel) {
		this.tableViewType = tableViewType;
		this.serviceName = serviceName;
		this.nodePath = nodePath;
		this.panelID = panelID;
		this.commonInfo = commonInfo;
		this.parentTabbedPane = parentTabbedPane;
		this.parentDataPanel = parentDataPanel;

		setTableViewInfo();
		setTableColumnInfo();
		setSwitchCode();
		setQueryParams();
//		setTableModel();
		tableModel = new TableModelBase(commonInfo, serviceName, panelID,
				columnTitleList, columnCodeList, columnDisplayList);
//
		setTableView();
		setAutoResizeMode(AUTO_RESIZE_OFF);
	}

	public TableViewBase(String tableViewType, String serviceName,
			String nodePath, String panelID, CommonInfo commonInfo,
			TabbedPaneBase parentTabbedPane, DataPanelBase parentDataPanel,
			String dummy) {
		this.tableViewType = tableViewType;
		this.serviceName = serviceName;
		this.nodePath = nodePath;
		this.panelID = panelID;
		this.commonInfo = commonInfo;
		this.parentTabbedPane = parentTabbedPane;
		this.parentDataPanel = parentDataPanel;

		setTableViewInfo();
		setTableColumnInfo();
		// setSwitchCode();
		// setQueryParams();
		// setTableModel();
		// setTableView();
		setAutoResizeMode(AUTO_RESIZE_OFF);
	}

	private void setTableViewInfo() {
		String tableViewStruct = commonInfo.getTableViewStruct(serviceName,
				panelID);
		if (tableViewStruct != null) {
			String[] lines = tableViewStruct.split("\\$");
			String[] tokens = lines[0].split("\\|");
			int size = tokens.length;
			if (size == 11) {
				for (int i = 0; i < 11; i++) {
					String token = tokens[i].trim();
					if (token.equals(""))
						token = null;
					switch (i) {
					case 0:
						methodWhenOpened = token;
						break;
					case 1:
						fontSize = commonInfo.getFontSize(token);
						break;
					case 2:
						rowHeight = commonInfo.getRowHeight(token);
						break;
					case 3:
						tableFgColor = commonInfo.getTableFgColor(token);
						break;
					case 4:
						tableBgColor = commonInfo.getTableBgColor(token);
						break;
					case 5:
						sorterType = token;
						break;
					case 6:
						selectionMode = token;
						break;
					case 7:
						switchMethod = token;
						break;
					case 8:
						switchMethodParam = token;
						break;
					case 9:
						rowSelectMethod = token;
						break;
					case 10:
						rowSelectMethodParam = token;
						break;
					}
				}
			} else {
				commonInfo.showMessageLong("format error : TableViewStruct $ "
						+ tableViewStruct);
			}
		} else {
			commonInfo
					.showMessageLong("TableViewStruct is not found $ serviceName = "
							+ serviceName + ", panelID = " + panelID);
		}
	}

	private void setTableColumnInfo() {
		String tableColumnStruct = commonInfo.getTableColumnStruct(serviceName,
				panelID);
		if (tableColumnStruct != null) {
			String[] tableColumnsInfo = tableColumnStruct.split("\\$");
			for (String tableColumnInfo : tableColumnsInfo) {
				String[] tokens = tableColumnInfo.split("\\|");
				int size = tokens.length;
				if (size == 8) {
					for (int i = 0; i < 8; i++) {
						String token = tokens[i].trim();
						if (token.equals(""))
							token = null;
						switch (i) {
						case 0:
							columnNumberList.add(new Integer(token));
							break;
						case 1:
							columnTitleList.add(token);
							break;
						case 2:
							columnCodeList.add(token);
							break;
						case 3:
							columnDisplayList.add(token);
							break;
						case 4:
							columnWidthList.add(new Integer(token));
							break;
						case 5:
							columnFgColorList.add(commonInfo
									.getTableFgColor(token));
							break;
						case 6:
							columnBgColorList.add(commonInfo
									.getTableBgColor(token));
							break;
						case 7:
							columnRendererList.add(token);
							break;
						}
					}
				} else {
					commonInfo
							.showMessageLong("format error in TableColumnStruct $ "
									+ tableColumnInfo);
				}
			}
			originalColumnCount = columnTitleList.size();
		} else {
			commonInfo
					.showMessageLong("TableColumnStruct is not found $ serviceName = "
							+ serviceName + ", panelID = " + panelID);
		}
	}

	protected void setSwitchCode() {
		if (switchMethod != null) {
			boolean ret = invokeSwitchMethod(switchMethod, switchMethodParam);
			if (ret) {
				switchCode = "1"; // switched switch_code for table query
			} else {
				switchCode = "0";
			}
		}
	}

	protected void setQueryParams() {
		String ans = commonInfo.getServerQueryParams(serviceName, panelID,
				switchCode);
		if (ans == null) {
			queryParams = null;
		} else {
			if (ans.trim().equals("")) {
				queryParams = null;
			} else {
				queryParams = ans.trim();
			}
		}
	}

	private void setVarTableViewInfo() {
		String varTableViewStruct = commonInfo.getVarTableViewStruct(
				serviceName, panelID);
		if (varTableViewStruct != null) {
			String[] lines = varTableViewStruct.split("\\$");
			String[] tokens = lines[0].split("\\|");
			int size = tokens.length;
			if (size == 6) {
				for (int i = 0; i < 6; i++) {
					String token = tokens[i].trim();
					if (token.equals(""))
						token = null;
					switch (i) {
					case 0:
						keyColumnCode = token;
						break;
					case 1:
						addColumnDisplay = token;
						break;
					case 2:
						addColumnWidth = Integer.parseInt(token);
						break;
					case 3:
						addColumnRenderer = token;
						break;
					case 4:
						addDataSwitchMethod = token;
						break;
					case 5:
						addDataSwitchParam = token;
						break;
					}
				}
			} else {
				commonInfo
						.showMessageLong("format error in VarTableViewStruct $ "
								+ varTableViewStruct);
			}
		} else {
			commonInfo
					.showMessageLong(" VarTableViewStruct is not found $ serviceName = "
							+ serviceName + ", panelID = " + panelID);
		}
	}

	private void setVarSwitchCode() {
		if (addDataSwitchMethod != null) {
			boolean ret = invokeSwitchMethod(addDataSwitchMethod,
					addDataSwitchParam);
			if (ret) {
				addDataSwitchCode = "5"; // switched switch_code for
											// add_data_query
			} else {
				addDataSwitchCode = "4";
			}
		}
	}

	private void setVarQueryParams() {
		String ans = commonInfo.getServerQueryParams(serviceName, panelID,
				addColumnSwitchCode);
		if (ans == null) {
			addColumnQueryParams = null;
		} else {
			if (ans.trim().equals("")) {
				addColumnQueryParams = null;
			} else {
				addColumnQueryParams = ans.trim();
			}
		}

		ans = commonInfo.getServerQueryParams(serviceName, panelID,
				addDataSwitchCode);
		if (ans == null) {
			addDataQueryParams = null;
		} else {
			if (ans.trim().equals("")) {
				addDataQueryParams = null;
			} else {
				addDataQueryParams = ans.trim();
			}
		}
	}

	private void setVarTableColumnInfo() {
		String ans = commonInfo.getQueryResult(serviceName, panelID,
				addColumnSwitchCode, addColumnQueryParamValues);
		if (ans != null) {
			int cnt = columnTitleList.size();
			if (cnt > originalColumnCount) {
				ArrayList<Integer> columnNumberList2 = new ArrayList<Integer>();
				ArrayList<String> columnTitleList2 = new ArrayList<String>();
				ArrayList<String> columnCodeList2 = new ArrayList<String>();
				ArrayList<String> columnDisplayList2 = new ArrayList<String>();
				ArrayList<Integer> columnWidthList2 = new ArrayList<Integer>();
				ArrayList<Color> columnFgColorList2 = new ArrayList<Color>();
				ArrayList<Color> columnBgColorList2 = new ArrayList<Color>();
				ArrayList<String> columnRendererList2 = new ArrayList<String>();
				for (int i = 0; i < originalColumnCount; i++) {
					columnNumberList2.add(columnNumberList.get(i));
					columnTitleList2.add(columnTitleList.get(i));
					columnCodeList2.add(columnCodeList.get(i));
					columnDisplayList2.add(columnDisplayList.get(i));
					columnWidthList2.add(columnWidthList.get(i));
					columnFgColorList2.add(columnFgColorList.get(i));
					columnBgColorList2.add(columnBgColorList.get(i));
					columnRendererList2.add(columnRendererList.get(i));
				}
				columnNumberList = columnNumberList2;
				columnTitleList = columnTitleList2;
				columnCodeList = columnCodeList2;
				columnDisplayList = columnDisplayList2;
				columnWidthList = columnWidthList2;
				columnFgColorList = columnFgColorList2;
				columnBgColorList = columnBgColorList2;
				// columnRendererList = columnRendererList;
			}

			cnt = columnTitleList.size();
			String[] lines = ans.split("\\$");
			for (String line : lines) {
				String[] tokens = line.split("\\|");
				String columnTitle = tokens[0];
				String columnCode = tokens[1];
				Integer columnIndex = cnt;
				columnNumberList.add(columnIndex);
				columnTitleList.add(columnTitle);
				columnCodeList.add(columnCode);
				columnDisplayList.add(addColumnDisplay);
				columnWidthList.add(new Integer(addColumnWidth));
				columnFgColorList.add(tableFgColor);
				columnBgColorList.add(tableBgColor);
				columnRendererList.add(addColumnRenderer);
				cnt++;
			}
		} else {
			commonInfo
					.showMessageLong(" AddColumnQuery is not found $ serviceName = "
							+ serviceName + ", panelID = " + nodePath);
		}
	}
/*
	protected void setTableModel() {
		tableModel = new TableModelBase(commonInfo, serviceName, panelID,
				columnTitleList, columnCodeList, columnDisplayList);
	}
*/
	protected void setTableView() {
		setFont(new Font("DialogInput", Font.PLAIN, fontSize));
		setRowHeight(rowHeight);
//		setForeground(tableFgColor);
//		setBackground(tableBgColor);
		if (selectionMode == null) {
			setSelectionMode(ListSelectionModel.MULTIPLE_INTERVAL_SELECTION);
		} else if (selectionMode.equals("singleSelection")) {
			setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
		} else {
			setSelectionMode(ListSelectionModel.MULTIPLE_INTERVAL_SELECTION);
		}
		if (sorterType != null) {
			if (sorterType.equals("simpleSorter")) {
				tableModel.addHeaderMouseListenerToSort(this);
			} else if (sorterType.equals("orderSorter")) {
				tableModel.addHeaderMouseListenerToOrderedSort(this);
			}
		}
		setModel(tableModel);
		addMouseListener(new RowSelectionListener());
	}

	public void pageOpened() {
		if (methodWhenOpened != null) {
			invokeMethodWhenOpened(methodWhenOpened);
		}

		setSwitchCode();
		setQueryParams();
		setQueryParamValuesAndDisplays();
		parentDataPanel.setTitleText(queryParamDisplays);

		if (queryParamValues != null) {
			if (tableViewType.equals("VarTableView")) {
				tableModel.setTableData(serviceName, panelID, switchCode,
						queryParamValues);
				setVarTableViewInfo();
				setVarSwitchCode();
				setVarQueryParams();
				setVarQueryParamValues();
				setVarTableColumnInfo();
				tableModel.setColumnLists(columnTitleList, columnCodeList,
						columnDisplayList);
				tableModel.setVarTableData(serviceName, panelID,
						addDataSwitchCode, addDataQueryParamValues,
						addColumnDisplay, keyColumnCode);
			} else if ((!queryParamValues.equals(oldQueryParamValues))
					|| (refreshTableFlag == true)) {
				tableModel.setTableData(serviceName, panelID, switchCode,
						queryParamValues);
			}
			setColumnDisplay();
			if (selectedRow >= 0) {
				setRowSelectionInterval(selectedRow, selectedRow);
			}
			oldQueryParamValues = queryParamValues;
			commonInfo.timerRestart();
		} else if (refreshTableFlag == true) {
			tableModel.setTableData(serviceName, panelID, switchCode,
					queryParamValues);
			setColumnDisplay();
			if (selectedRow >= 0) {
				setRowSelectionInterval(selectedRow, selectedRow);
			}
			commonInfo.timerRestart();
		}
	}

	protected void setQueryParamValuesAndDisplays() {
		if (queryParams == null) {
			queryParamValues = "empty";
			queryParamDisplays = "";
		} else {
			StringBuilder sbuf1 = new StringBuilder();
			StringBuilder sbuf2 = new StringBuilder();
			String[] tokens = queryParams.split("\\:");
			for (String token : tokens) {
				String columnCode = token;
				String value = "";
				String display = "";
				if (parentTabbedPane.columnCodeMapContains(columnCode)) {
					value = parentTabbedPane
							.getValueFromColumnCodeMap(columnCode);
					display = parentTabbedPane
							.getDisplayFromColumnCodeMap(columnCode);
					sbuf1.append(value).append("|");
					sbuf2.append(display).append(" ");
				} else if (commonInfo.commonCodeMapContains(columnCode)) {
					value = commonInfo.getValueFromCommonCodeMap(columnCode);
					display = commonInfo
							.getDisplayFromCommonCodeMap(columnCode);
					sbuf1.append(value).append("|");
					sbuf2.append(display).append(" ");
				} else {
					if (!commonInfo.structControlDebugMode) {
						commonInfo.showMessageLong("TableViewBase: $ "
								+ columnCode + " の値が設定されていません。");
						return;
					} else { // structControl のデバッグ時の設定
						String str = "debugMode: パラメータ " + columnCode
								+ " の値を指定して下さい。";
						value = commonInfo.getDialogInput(str);
						if (value == null) {
							value = " ";
						} else {
							String val = value.trim();
							if (val.equals("")) {
								value = " ";
							} else {
								value = val;
							}
						}
						parentTabbedPane.addColumnCodeMap(columnCode, value,
								value);
						sbuf1.append(value).append("|");
						sbuf2.append(value).append(" ");
					}
				}
			}
			queryParamValues = sbuf1.toString();
			queryParamDisplays = sbuf2.toString();
		}
	}

	private void setVarQueryParamValues() {
		if (addColumnQueryParams == null) {
			addColumnQueryParamValues = "empty";
		} else {
			StringBuilder sbuf = new StringBuilder();
			String[] tokens = addColumnQueryParams.split("\\:");
			for (String token : tokens) {
				String columnCode = token;
				String value = "";
				if (parentTabbedPane.columnCodeMapContains(columnCode)) {
					value = parentTabbedPane
							.getValueFromColumnCodeMap(columnCode);
					sbuf.append(value).append("|");
				} else if (commonInfo.commonCodeMapContains(columnCode)) {
					value = commonInfo.getValueFromCommonCodeMap(columnCode);
					sbuf.append(value).append("|");
				} else {
					if (!commonInfo.structControlDebugMode) {
						commonInfo.showMessageLong("TableViewBase: $ "
								+ columnCode + " の値が設定されていません。");
						return;
					} else { // structControl のデバッグ時の設定
						String str = "debugMode: パラメータ " + columnCode
								+ " の値を指定して下さい。";
						value = commonInfo.getDialogInput(str);
						if (value == null) {
							value = " ";
						} else {
							String val = value.trim();
							if (val.equals("")) {
								value = " ";
							} else {
								value = val;
							}
						}
						parentTabbedPane.addColumnCodeMap(columnCode, value,
								value);
						sbuf.append(value).append("|");
					}
				}
			}
			addColumnQueryParamValues = sbuf.toString();
		}

		if (addDataQueryParams == null) {
			addDataQueryParamValues = "empty";
		} else {
			StringBuilder sbuf = new StringBuilder();
			String[] tokens = addDataQueryParams.split("\\:");
			for (String token : tokens) {
				String columnCode = token;
				String value = "";
				if (parentTabbedPane.columnCodeMapContains(columnCode)) {
					value = parentTabbedPane
							.getValueFromColumnCodeMap(columnCode);
					sbuf.append(value).append("|");
				} else if (commonInfo.commonCodeMapContains(columnCode)) {
					value = commonInfo.getValueFromCommonCodeMap(columnCode);
					sbuf.append(value).append("|");
				} else {
					if (!commonInfo.structControlDebugMode) {
						commonInfo.showMessageLong("TableViewBase: $ "
								+ columnCode + " の値が設定されていません。");
						return;
					} else { // structControl のデバッグ時の設定
						String str = "debugMode: パラメータ " + columnCode
								+ " の値を指定して下さい。";
						value = commonInfo.getDialogInput(str);
						if (value == null) {
							value = " ";
						} else {
							String val = value.trim();
							if (val.equals("")) {
								value = " ";
							} else {
								value = val;
							}
						}
						parentTabbedPane.addColumnCodeMap(columnCode, value,
								value);
						sbuf.append(value).append("|");
					}
				}
			}
			addDataQueryParamValues = sbuf.toString();
		}
	}

	public String getColumnTitle(String columnCode) {
		for (int i = 0; i < getColumnCount(); i++) {
			String columnCode2 = columnCodeList.get(i);
			String columnTitle2 = columnTitleList.get(i);
			if (columnCode.equals(columnCode2)) {
				return columnTitle2;
			}
		}
		return null;
	}

	public String getCodeAt(int row, String columnCode) {
		for (int i = 0; i < getColumnCount(); i++) {
			String columnCode2 = columnCodeList.get(i);
			String columnTitle2 = columnTitleList.get(i);
			if (columnCode.equals(columnCode2)) {
				return tableModel.getCellValueAt(row, columnTitle2);
			}
		}
		return null;
	}

	public String getNameAt(int row, String columnCode) {
		for (int i = 0; i < getColumnCount(); i++) {
			String columnCode2 = columnCodeList.get(i);
			String columnTitle2 = columnTitleList.get(i);
			if (columnCode.equals(columnCode2)) {
				return tableModel.getCellDisplayAt(row, columnTitle2);
			}
		}
		return null;
	}

	protected void setColumnDisplay() {
		for (int i = 0; i < getColumnCount(); i++) {
			String title = columnTitleList.get(i);
			TableColumn tc = getColumn(title);
			int width = columnWidthList.get(i);
			tc.setWidth(width);
			tc.setPreferredWidth(width);
			tc.setHeaderRenderer(new HeaderRenderer(title));
			String rendererName = columnRendererList.get(i);
			Color fgColor = columnFgColorList.get(i);
			Color bgColor = columnBgColorList.get(i);
			if (rendererName == null) {
				tc.setCellRenderer(new SimpleRenderer(fgColor, bgColor));
			} else {
				tc.setCellRenderer(new SpecialRenderer(rendererName, fgColor,
						bgColor));
			}
			if ((title.equals("順位")) && (sorterType.equals("orderSorter"))) {
				tc.setCellRenderer(new OrderRenderer());
			}
		}
	}

	protected void invokeMethodWhenOpened(String methodName) {
		try {
			getClass().getMethod(methodName, (java.lang.Class[]) null).invoke(
					this, (java.lang.Object[]) null);
		} catch (Exception ex) {
			commonInfo
					.showMessageLong(" MethodWhenOpened is not invoked $ methodWhenOpened = "
							+ methodName);
		}
	}

	public void refreshTable() {
		setQueryParamValuesAndDisplays();
		parentDataPanel.setTitleText(queryParamDisplays);

		if (queryParamValues != null) {
			tableModel.setTableData(serviceName, panelID, switchCode,
					queryParamValues);
		}
		setColumnDisplay();
		if (selectedRow >= 0) {
			setRowSelectionInterval(selectedRow, selectedRow);
		}
		oldQueryParamValues = queryParamValues;
		commonInfo.timerRestart();
	}

	public void setRefreshTableFlag() {
		refreshTableFlag = true;
	}

	public void setYearMonth() {
		int year, month;
		String val = parentTabbedPane
				.getValueFromColumnCodeMap("PRESENT_TAB_INDEX");
		int index = Integer.parseInt(val);

		if (index <= 8) {
			month = index + 4;
			year = commonInfo.thisSchoolYear;
		} else {
			month = index - 8;
			year = commonInfo.thisSchoolYear + 1;
		}
		String monthString;
		if (month < 10) {
			monthString = "0" + month;
		} else {
			monthString = "" + month;
		}
		parentTabbedPane.addColumnCodeMap("YEAR", "" + year, "" + year + "年");
		parentTabbedPane.addColumnCodeMap("MONTH", monthString, monthString
				+ "月");
		if ((year == commonInfo.thisSchoolYear)
				&& (month == commonInfo.thisMonth)) {
			selectedRow = commonInfo.thisDay - 1;
		} else {
			selectedRow = -1;
		}
	}

	public void setYearMonth2() {
		int year, month;
		String value = parentTabbedPane
				.getValueFromColumnCodeMap("PRESENT_TAB_INDEX");
		String schoolYear = parentTabbedPane
				.getValueFromColumnCodeMap("SCHOOL_YEAR");
		if (schoolYear == null) {
			schoolYear = commonInfo
					.getValueFromCommonCodeMap("THIS_SCHOOL_YEAR");
		}
		int index = Integer.parseInt(value);
		int syear = Integer.parseInt(schoolYear);

		if (index <= 8) {
			month = index + 4;
			year = syear;
		} else {
			month = index - 8;
			year = syear + 1;
		}
		String monthString;
		if (month < 10) {
			monthString = "0" + month;
		} else {
			monthString = "" + month;
		}
		parentTabbedPane.addColumnCodeMap("YEAR", "" + year, "" + year + "年");
		parentTabbedPane.addColumnCodeMap("MONTH", monthString, monthString
				+ "月");
	}

	class RowSelectionListener extends MouseAdapter {

		public void mousePressed(MouseEvent e) {
			if (SwingUtilities.isLeftMouseButton(e)) {
				if (getSelectedRowCount() > 0) {
					int row = getSelectedRow();
					for (int i = 0; i < getColumnCount(); i++) {
						String columnCode = columnCodeList.get(i);
						String columnTitle = columnTitleList.get(i);
						if (columnCode != null) {
							String cellValue = tableModel.getCellValueAt(row,
									columnTitle);
							String cellDisplay = tableModel.getCellDisplayAt(
									row, columnTitle);
							parentTabbedPane.addColumnCodeMap(columnCode,
									cellValue, cellDisplay);
							if (columnCode.startsWith("CHILD_")) {
								String columnCode2 = columnCode.substring(6);
								parentTabbedPane.addColumnCodeMap(columnCode2,
										cellValue, cellDisplay);
							}
							if (columnCode.equals("COMMAND_CODE")) {
								if (cellValue.startsWith("1")) {
									parentTabbedPane.addColumnCodeMap(
											"DELETE_CODE", cellValue,
											cellDisplay);
								} else if (cellValue.startsWith("2")) {
									parentTabbedPane.addColumnCodeMap(
											"UPDATE_CODE", cellValue,
											cellDisplay);
								} else if (cellValue.startsWith("3")) {
									parentTabbedPane.addColumnCodeMap(
											"INSERT_CODE", cellValue,
											cellDisplay);
								} else if (cellValue.startsWith("4")) {
									parentTabbedPane.addColumnCodeMap(
											"SPECIAL_CODE", cellValue,
											cellDisplay);
								}
							}
						}
					}
				}
			}
		}

		public void mouseReleased(MouseEvent e) {
			if (SwingUtilities.isLeftMouseButton(e)) {
				if (rowSelectMethod != null) {
					invokeMethodWhenRowSelected(rowSelectMethod,
							rowSelectMethodParam);
				}
			}
		}
	}

	public void rowSelectAction(int row) {
		for (int i = 0; i < getColumnCount(); i++) {
			String columnCode = columnCodeList.get(i);
			String columnTitle = columnTitleList.get(i);
			if (columnCode != null) {
				String cellValue = tableModel.getCellValueAt(row, columnTitle);
				String cellDisplay = tableModel.getCellDisplayAt(row,
						columnTitle);
				parentTabbedPane.addColumnCodeMap(columnCode, cellValue,
						cellDisplay);
				if (columnCode.startsWith("CHILD_")) {
					String columnCode2 = columnCode.substring(6);
					parentTabbedPane.addColumnCodeMap(columnCode2, cellValue,
							cellDisplay);
				}
			}
		}
	}

	// **** methods invoked by invokeSwitchMethod ****//

	private boolean invokeSwitchMethod(String switchMethod,
			String switchMethodParam) {
		// switchMethodParam is not used now
		try {
			RetBoolean ret = new RetBoolean();
			getClass()
					.getMethod(switchMethod, new Class[] { RetBoolean.class })
					.invoke(this, new Object[] { ret });
			return ret.getValue();
		} catch (Exception ex) {
			commonInfo.showMessageLong(" switchMethod is not invoked $ "
					+ switchMethod);
			return false;
		}
	}

	public void switchBy30(RetBoolean ret) {
		String DEPARTMENT = parentTabbedPane
				.getValueFromColumnCodeMap("DEPARTMENT");
		if (DEPARTMENT.equals("30")) {
			ret.accept();
		} else {
			ret.reject();
		}
	}

	public void switchByCourse(RetBoolean ret) {
		String COURSE = parentTabbedPane.getValueFromColumnCodeMap("COURSE");
		if (COURSE.equals("30")) {
			ret.accept();
		} else {
			ret.reject();
		}
	}

	public void switchByStaffQual(RetBoolean ret) {
		boolean b = checkStaffQual();
		if (b == false) {
			ret.accept();
		} else {
			ret.reject();
		}
	}

	public void switchByQualification(RetBoolean ret) {
		boolean b = checkStaffQual();
		if (b == false) {
			ret.accept();
		} else {
			ret.reject();
		}
	}

	public void switchByMD(RetBoolean ret) {
		String FACULTY = parentTabbedPane.getValueFromColumnCodeMap("FACULTY");
		if (FACULTY.equals("51")) {
			ret.accept();
		} else {
			ret.reject();
		}
	}

	public void switchByStudent(RetBoolean ret) {
		if (!commonInfo.structControlDebugMode) {
			if (!commonInfo.STUDENT_CODE.equals("")) {
				ret.accept();
			} else {
				ret.reject();
			}
		} else {
			if (commonInfo.studentDebugUserSelected) {
				ret.accept();
			} else {
				ret.reject();
			}
		}
	}

	private boolean checkStaffQual() {
		String staffQual = commonInfo.STAFF_QUALIFICATION;
		String staffDept = commonInfo.STAFF_DEPARTMENT;

		if (commonInfo.structControlDebugMode) {
			if (commonInfo.staffDebugUserSelected) {
				staffQual = commonInfo.STAFF_QUALIFICATION_D;
				staffDept = commonInfo.STAFF_DEPARTMENT_D;
			}
		}

		if ((staffQual.equals("4")) || (staffQual.equals("5"))
				|| (staffQual.equals("6")) || (staffQual.equals("8"))
				|| (staffQual.equals("9"))) {
			return true;
		} else if (staffQual.equals("3")) {
			String FACULTY = parentTabbedPane
					.getValueFromColumnCodeMap("FACULTY");
			String DEPARTMENT = parentTabbedPane
					.getValueFromColumnCodeMap("DEPARTMENT");
			String COURSE = parentTabbedPane
					.getValueFromColumnCodeMap("COURSE");

			String STAFF_ATTRIB = commonInfo.STAFF_ATTRIB;

			if (commonInfo.structControlDebugMode) {
				if (commonInfo.staffDebugUserSelected) {
					STAFF_ATTRIB = commonInfo.STAFF_ATTRIB_D;
				}
			}
			int staffAttrib = Integer.parseInt(STAFF_ATTRIB);

			if (COURSE == null) {
				if ((staffAttrib == 81) || (staffAttrib == 235)) {
					if (((FACULTY.equals("32")) && (DEPARTMENT.equals("75")))
							|| ((FACULTY.equals("51")) && (DEPARTMENT
									.equals("93")))) {
						return true;
					} else {
						return false;
					}
				}
				return false;
			} else {

				switch (staffAttrib) {
				case 55:
				case 205:
					if (((FACULTY.equals("11")) && (DEPARTMENT.equals("31")))
							|| ((FACULTY.equals("32"))
									&& (DEPARTMENT.equals("73")) && (COURSE
										.equals("85")))
							|| ((FACULTY.equals("51"))
									&& (DEPARTMENT.equals("91")) && (COURSE
										.equals("85")))) {
						return true;
					} else {
						return false;
					}
				case 57:
				case 210:
					if (((FACULTY.equals("11")) && (DEPARTMENT.equals("32")))
							|| ((FACULTY.equals("32"))
									&& (DEPARTMENT.equals("74")) && (COURSE
										.equals("88")))
							|| ((FACULTY.equals("51"))
									&& (DEPARTMENT.equals("92")) && (COURSE
										.equals("88")))) {
						return true;
					} else {
						return false;
					}
				case 59:
				case 215:
					if (((FACULTY.equals("11")) && (DEPARTMENT.equals("33")))
							|| ((FACULTY.equals("32"))
									&& (DEPARTMENT.equals("73")) && (COURSE
										.equals("86")))
							|| ((FACULTY.equals("51"))
									&& (DEPARTMENT.equals("91")) && (COURSE
										.equals("86")))) {
						return true;
					} else {
						return false;
					}
				case 61:
				case 220:
					if (((FACULTY.equals("11")) && (DEPARTMENT.equals("34")))
							|| ((FACULTY.equals("32"))
									&& (DEPARTMENT.equals("74")) && (COURSE
										.equals("89")))
							|| ((FACULTY.equals("51"))
									&& (DEPARTMENT.equals("92")) && (COURSE
										.equals("89")))) {
						return true;
					} else {
						return false;
					}
				case 63:
				case 225:
					if (((FACULTY.equals("11")) && (DEPARTMENT.equals("35")))
							|| ((FACULTY.equals("32"))
									&& (DEPARTMENT.equals("73")) && (COURSE
										.equals("87")))
							|| ((FACULTY.equals("51"))
									&& (DEPARTMENT.equals("91")) && (COURSE
										.equals("87")))) {
						return true;
					} else {
						return false;
					}
				case 65:
				case 230:
					return false;
				case 81:
				case 235:
					if (((FACULTY.equals("32")) && (DEPARTMENT.equals("75")))
							|| ((FACULTY.equals("51")) && (DEPARTMENT
									.equals("93")))
							|| ((FACULTY.equals("32"))
									&& (DEPARTMENT.equals("75")) && (COURSE
										.equals("90")))
							|| ((FACULTY.equals("51"))
									&& (DEPARTMENT.equals("93")) && (COURSE
										.equals("90")))) {
						return true;
					} else {
						return false;
					}
				default:
					return false;
				}
			}
		} else if (staffQual.equals("2")) {

			// この設定は、学科別に設定されている「自学科の学生の成績情報の閲覧を
			// 許可する」というアクセス権限に関する設定を実現するためのパッチである。

			String FACULTY = parentTabbedPane
					.getValueFromColumnCodeMap("FACULTY");
			String DEPARTMENT = parentTabbedPane
					.getValueFromColumnCodeMap("DEPARTMENT");
			String COURSE = parentTabbedPane
					.getValueFromColumnCodeMap("COURSE");

			String STAFF_ATTRIB = commonInfo.STAFF_ATTRIB;

			if (commonInfo.structControlDebugMode) {
				if (commonInfo.staffDebugUserSelected) {
					STAFF_ATTRIB = commonInfo.STAFF_ATTRIB_D;
				}
			}
			int staffAttrib = Integer.parseInt(STAFF_ATTRIB);

			switch (staffAttrib) {
			case 55:
			case 205:
				if ((FACULTY.equals("11")) && (DEPARTMENT.equals("31"))) {
					return true;
				} else {
					return false;
				}
			default:
				return false;
			}

		} else {
			return false;
		}

	}

	// **** methods invoked by Row Selection ****//

	protected void invokeMethodWhenRowSelected(String methodName,
			String methodParam) {
		try {
			getClass().getMethod(methodName, new Class[] { String.class })
					.invoke(this, new Object[] { methodParam });
		} catch (Exception ex) {
			commonInfo
					.showMessageLong("MethodWhenRowSelected is not invoked $ rowSelectMethod = "
							+ methodName
							+ " $ rowSelectMethodParam = "
							+ methodParam);
		}
	}

	public void openTab(String nodeID) {
		parentTabbedPane.openTab(nodeID);
	}

	public class OrderRenderer extends DefaultTableCellRenderer {
		public Component getTableCellRendererComponent(JTable table,
				Object value, boolean isSelected, boolean hasFocus, int row,
				int column) {
			if (value == null) {
				return null;
			}
			int val = 1 + row;
			CellObject cobj = new CellObject(val, 3);
			setValueAt(cobj, row, column);
			setText(" " + val + " ");
			setHorizontalAlignment(RIGHT);
			setBackground(table.getSelectionBackground());
			setFont(new Font("DialogInput", Font.PLAIN, 12));
			return this;
		}
	}

	public class SpecialRenderer extends DefaultTableCellRenderer {
		String name;
		Color fgColor;
		Color bgColor;

		public SpecialRenderer(String name, Color fgColor, Color bgColor) {
			this.name = name;
			this.fgColor = fgColor;
			this.bgColor = bgColor;
		}

		public Component getTableCellRendererComponent(JTable tb, Object val,
				boolean isSelected, boolean hasFocus, int row, int column) {
			if (val == null) {
				return null;
			}

			setForeground(fgColor);
			setBackground(bgColor);
			if (name.equals("yokenUnit")) {
				CellObject co = (CellObject) val;
				String disp = co.getDisplay().trim();
				if (disp.startsWith("*")) {
					setForeground(Color.red);
				} else if (disp.startsWith("+")) {
					setForeground(Color.green);
				}
			} else if (name.equals("weekOfDay")) {
				CellObject co = (CellObject) val;
				String code = co.getCodeValue();
				if (code.equals("0")) {
					setForeground(Color.red);
				} else if (code.equals("6")) {
					setForeground(Color.blue);
				} else {
					setForeground(Color.black);
				}
			} else if (name.equals("eventDisplay")) {
				CellObject co = (CellObject) val;
				String code = co.getCodeValue();
				if (code.indexOf("期末試験") >= 0) {
					setForeground(commonInfo.getColor("DodgerBlue"));
				} else if (code.indexOf("申告期間") >= 0) {
					setForeground(commonInfo.getColor("DarkGreen"));
				} else if (code.indexOf("休業") >= 0) {
					setForeground(commonInfo.getColor("DarkRed"));
				} else {
					setForeground(Color.black);
				}
			} else if (name.equals("classWeek")) {
				CellObject co = (CellObject) val;
				String code = co.getCodeValue();
				if (code != null) {
					if (code.equals("0")) {
						setForeground(Color.red);
					} else if (code.equals("6")) {
						setForeground(Color.blue);
					} else if (code.equals("1")) {
						setForeground(commonInfo.getColor("DodgerBlue"));
					} else if (code.equals("2")) {
						setForeground(commonInfo.getColor("MediumBlue"));
					} else if (code.equals("3")) {
						setForeground(commonInfo.getColor("Navy"));
					} else if (code.equals("4")) {
						setForeground(commonInfo.getColor("DarkGreen"));
					} else if (code.equals("5")) {
						setForeground(commonInfo.getColor("DarkRed"));
					} else {
						setForeground(Color.black);
					}
				}
			} else if (name.equals("Tokuten")) {
				setForeground(Color.blue);
			} else if (name.equals("registrStatus")) {
				CellObject co = (CellObject) val;
				String code = co.getCodeValue();
				if (code.equals("1")) {
					setForeground(commonInfo.getColor("DodgerBlue"));
				} else if (code.equals("2")) {
					setForeground(commonInfo.getColor("DarkSalmon"));
				} else if (code.equals("3")) {
					setForeground(commonInfo.getColor("SeaGreen"));
				} else if (code.equals("4")) {
					setForeground(commonInfo.getColor("Firebrick"));
				}
			} else if (name.equals("Attend")) {
				CellObject co = (CellObject) val;
				String code = co.getCodeValue();
				if (code != null) {
					StringTokenizer stk = new StringTokenizer(code, "|");
					String attendFlag = stk.nextToken();
					String readFlag = stk.nextToken();

					if (readFlag.equals("0")) {
						setForeground(Color.blue);
					} else if (readFlag.equals("1")) {
						setForeground(Color.red);
					} else if (readFlag.equals("2")) {
						setForeground(Color.green);
					} else if (readFlag.equals("3")) {
						setForeground(Color.cyan);
					}
				}
			} else if (name.equals("ReadFlag")) {
				CellObject co = (CellObject) val;
				String code = co.getCodeValue();
				if (code != null) {
					String readFlag = code;
					if (readFlag.equals("0")) {
						setForeground(Color.blue);
					} else if (readFlag.equals("1")) {
						setForeground(Color.red);
					} else if (readFlag.equals("2")) {
						setForeground(Color.green);
					} else if (readFlag.equals("3")) {
						setForeground(Color.cyan);
					}
				}
			} else if (name.equals("Blue")) {
				setForeground(Color.blue);
			} else if (name.equals("bgColor")) {
				CellObject co = (CellObject) val;
				Color color = (Color) co.getCode();
				setBackground(color);
			} else if (name.equals("fgColor")) {
				CellObject co = (CellObject) val;
				Color color = (Color) co.getCode();
				setForeground(color);
			}

			if (isSelected) {
				setBackground(tb.getSelectionBackground());
			}

			CellObject co = (CellObject) val;
			if (co.isHighlighted() || co.isChanged()) {
				setBackground(Color.yellow);
			}

			String str = co.getDisplay();
			setFont(new Font("DialogInput", Font.PLAIN, 12));
			setBorder(new EmptyBorder(2, 5, 2, 2));
			setText(str);
			return this;
		}
	}

	public class SimpleRenderer extends DefaultTableCellRenderer {
		Color fgColor;
		Color bgColor;
		String name;

		public SimpleRenderer(Color fgColor, Color bgColor) {
			this.fgColor = fgColor;
			this.bgColor = bgColor;
		}

		public Component getTableCellRendererComponent(JTable tb, Object val,
				boolean isSelected, boolean hasFocus, int row, int column) {
			setForeground(fgColor);
			setBackground(bgColor);
			if (isSelected) {
				setBackground(tb.getSelectionBackground());
			}

			CellObject co = (CellObject) val;
			if (co.isHighlighted() || co.isChanged()) {
				setBackground(Color.yellow);
			}
			String str = co.getDisplay();
			setText(str);
			setFont(new Font("DialogInput", Font.PLAIN, 12));
			setBorder(new EmptyBorder(2, 5, 2, 2));
			return this;
		}
	}

	public class HeaderRenderer extends DefaultTableCellRenderer {
		ArrayList<String> list = new ArrayList<String>();

		public HeaderRenderer(String title) {
			super();
			String[] tokens = title.split("\\#");
			for (String token : tokens) {
				list.add(token);
			}
		}

		public Component getTableCellRendererComponent(JTable table,
				Object value, boolean isSelected, boolean hasFocus, int row,
				int column) {
			if (value == null) {
				return null;
			}
			JPanel header = new JPanel();
			header.setLayout(new BoxLayout(header, BoxLayout.Y_AXIS));
			for (int i = 0; i < list.size(); i++) {
				JLabel label = new JLabel(list.get(i));
				label.setBorder(new EmptyBorder(1, 5, 1, 1));
				label.setFont(new Font("DialogInput", Font.PLAIN, 12));
				header.add(label);
			}
			header.setBorder(BorderFactory.createRaisedBevelBorder());
			return header;
		}
	}

}
