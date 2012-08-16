package common;

import clients.*;
import java.util.*;
import javax.swing.*;
import javax.swing.table.*;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.io.File;

public class ReportTableView extends TableViewBase {
	static JFileChooser chooser = new JFileChooser();
	public PopupInput popupInput;
	public PopupListener popupListener;

	private int selectedRow;
	private String toDay;

	public ReportTableView(String tableViewType, String serviceName,
			String nodePath, String panelID,
			// CommonInfoBase commonInfo,
			CommonInfo commonInfo, TabbedPaneBase tabbedPane,
			DataPanelBase dataPanel) {
		super(tableViewType, serviceName, nodePath, panelID, commonInfo,
				tabbedPane, dataPanel);
		popupInput = new PopupInput(commonInfo, this);
		popupListener = new PopupListener();
		addMouseListener(popupListener);

		String mon, day;
		if (commonInfo.thisMonth < 10) {
			mon = "0" + commonInfo.thisMonth;
		} else {
			mon = "" + commonInfo.thisMonth;
		}
		if (commonInfo.thisDay < 10) {
			day = "0" + commonInfo.thisDay;
		} else {
			day = "" + commonInfo.thisDay;
		}
		toDay = "" + commonInfo.thisYear + "-" + mon + "-" + day;
	}

	class PopupListener extends MouseAdapter {
		Dimension itsSize;
		Point origin;
		int x, y;

		public PopupListener() {
			// popupInput.setSize(260, 350);
			itsSize = popupInput.getSize();
			popupInput.setVisible(false);
		}

		public void mouseReleased(MouseEvent e) {
			if (origin == null) {
				origin = getLocationOnScreen();
				x = e.getX() - itsSize.width *2/3 ;
				y = e.getY() - itsSize.height *2/3;
				origin.x = Math.max(0, origin.x + x);
				origin.y = Math.max(0, origin.y + y);
				popupInput.setLocation(origin);
			}
			selectedRow = getSelectedRow();
			showPopupInput();
		}

		public void selectNext() {
			if (selectedRow < getRowCount() - 1) {
				selectedRow++;
				setRowSelectionInterval(selectedRow, selectedRow);
				doSelectAction(selectedRow);
				y = y + rowHeight + 1;
				showPopupInput();
			} /*
			 * else { clearSelection(); removePopupInput(); }
			 */
		}

		public void selectPrevious() {
			if (selectedRow > 0) {
				selectedRow--;
				setRowSelectionInterval(selectedRow, selectedRow);
				doSelectAction(selectedRow);
				y = y - rowHeight + 1;
				showPopupInput();
			} /*
			 * else { clearSelection(); removePopupInput(); }
			 */
		}

		public void showPopupInput() {
			String studentCode = parentTabbedPane
					.getValueFromColumnCodeMap("STUDENT_CODE");
			String studentName = parentTabbedPane
					.getDisplayFromColumnCodeMap("STUDENT_CODE");
			String marks = parentTabbedPane.getValueFromColumnCodeMap("MARKS");
			String shoriStatus = parentTabbedPane
					.getValueFromColumnCodeMap("SHORI_STATUS");

			popupInput.setInfo(studentCode, studentName, marks, shoriStatus);
			// popupInput.setLocation(origin.x + x, origin.y + y);
			popupInput.toFront();
			popupInput.setVisible(true);
		}

		public void removePopupInput() {
			popupInput.setVisible(false);
		}
	}

	public void doSelectAction(int row) {
		for (int i = 0; i < getColumnCount(); i++) {
			String columnCode = columnCodeList.get(i);
			String columnTitle = columnTitleList.get(i);
			CellObject cobj = (CellObject) tableModel.getValueAt(row,
					columnTitle);
			parentTabbedPane.addColumnCodeMap(columnCode, cobj.getCodeValue(),
					cobj.getDisplay());
		}
	}

	public void setMarks(int gain) {
		setMarks(selectedRow, gain);
	}

	public void setMarks(int row, int gain) {
		if ((gain >= 0) && (gain <= 100)) {
			// String faculty =
			// parentTabbedPane.getValueFromColumnCodeMap("FACULTY");
			// if (faculty.equals("32")) {
			// if (gain < 60) {
			// gain = 0;
			// }
			// }
			CellObject cobj = new CellObject(new Integer(gain), 4);
			cobj.setHighlighted();
			tableModel.setValueAt(cobj, row, "得点");
			String hyoka = "";
			if (gain == 0) {
				hyoka = "不合格";
			} else if (gain < 60) {
				hyoka = "再試験";
			} else if (gain < 70) {
				hyoka = "可";
			} else if (gain < 80) {
				hyoka = " 良";
			} else if (gain < 90) {
				hyoka = "  優";
			} else if (gain <= 100) {
				hyoka = "   秀";
			}
			cobj = new CellObject(new Integer(gain), hyoka);
			cobj.setHighlighted();
			tableModel.setValueAt(cobj, row, "評価");
			cobj = new CellObject(toDay);
			cobj.setHighlighted();
			tableModel.setValueAt(cobj, row, "報告年月日");
			commonInfo.setUnsavedDataExistFlag(true);
		} else if (gain == -1) {
			CellObject cobj = new CellObject(new Integer(-1), "");
			cobj.setHighlighted();
			tableModel.setValueAt(cobj, row, "得点");
			cobj = new CellObject(new Integer(-1), "");
			cobj.setHighlighted();
			tableModel.setValueAt(cobj, row, "評価");
			cobj = new CellObject(toDay);
			cobj.setHighlighted();
			tableModel.setValueAt(cobj, row, "報告年月日");
			commonInfo.setUnsavedDataExistFlag(true);
		} else {
			commonInfo.showMessage("条件  0 ≦ 得点 ≦ 100  が満たされていません");
		}
	}

	public void cancelInputData() {
		pageOpened();
	}

	public void saveKimatsuData() {
		if (nodePath.indexOf("kimatsu") < 0)
			return;

		Object[] msg = { " 記入された期末試験の得点データをデータベースにセーブ ", " します。",
				" セーブされた得点データは「仮報告」得点として扱われ、 ",
				" 該当学生の履修登録科目表に「仮成績」として表示されます。", " " };
		int ans = JOptionPane.showConfirmDialog(commonInfo.getFrame(), msg,
				"セーブの確認", JOptionPane.OK_CANCEL_OPTION);
		if (ans == JOptionPane.OK_OPTION) {
			String schoolYear = parentTabbedPane
					.getValueFromColumnCodeMap("SCHOOL_YEAR");
			String subjectCode = parentTabbedPane
					.getValueFromColumnCodeMap("SUBJECT_CODE");
			String subjectName = parentTabbedPane
					.getDisplayFromColumnCodeMap("SUBJECT_CODE");
			String classCode = parentTabbedPane
					.getValueFromColumnCodeMap("CLASS_CODE");
			String teacherCode = parentTabbedPane
					.getValueFromColumnCodeMap("TEACHER_CODE");
			String teacherName = parentTabbedPane
					.getDisplayFromColumnCodeMap("TEACHER_CODE");

			String key = schoolYear + "|" + subjectCode + "|" + classCode + "|"
					+ teacherCode;
			ArrayList<String> list = makeDataToSave();
			int num = commonInfo.commonInfoMethods.saveKimatsuData(key,
					teacherCode, list);
			commonInfo.setUnsavedDataExistFlag(true);
			if (num > 0) {
				refreshTable();
			}
			commonInfo.setUnsavedDataExistFlag(false);
		}
	}

	public void doKimatsuReport() {
		if (nodePath.indexOf("kimatsu") < 0)
			return;

		Object[] msg = { /*
				" 「成績報告」は、今回入力した得点データと以前に入力した ",
				" 「仮報告」状態の得点データを合わせて、「確定した得点」 ", " としてデータベースに格納する作業です。 ", " ",
				" 「不合格」の場合には「得点」を 0点、「再試験」の場合 ",
				" には「得点」を 1点 から 59点 の範囲に設定して下さい。", " ",
				" 「得点」欄が空欄のままの学生 (成績が未確定の学生) について ",
				" は「成績報告」は実行されないので注意して下さい。   ", " ",
				" 「成績報告」が実行されると、学務係に対して「報告通知」が ",
				" 送られ、学務係は得点データをチェックした上で、得点データ ", " を「成績表」または「再試験表」に記入します。",
				" ", " 「報告済の成績」を修正するには、学務係に「試験成績修正願い」 ",
				" を提出し、「手作業」による得点修正を学務係に依頼することが ", " 必要となります。", " " 
				*/
				"すでに仮報告済みの成績とあわせて教務係に成績報告します。よろしいですね？"
				};
		int ans = JOptionPane.showConfirmDialog(commonInfo.getFrame(), msg,
				"報告の確認", JOptionPane.OK_CANCEL_OPTION);
		if (ans == JOptionPane.OK_OPTION) {
			String schoolYear = parentTabbedPane
					.getValueFromColumnCodeMap("SCHOOL_YEAR");
			String subjectCode = parentTabbedPane
					.getValueFromColumnCodeMap("SUBJECT_CODE");
			String subjectName = parentTabbedPane
					.getDisplayFromColumnCodeMap("SUBJECT_CODE");
			String classCode = parentTabbedPane
					.getValueFromColumnCodeMap("CLASS_CODE");
			String teacherCode = parentTabbedPane
					.getValueFromColumnCodeMap("TEACHER_CODE");
			String teacherName = parentTabbedPane
					.getDisplayFromColumnCodeMap("TEACHER_CODE");

			String key = schoolYear + "|" + subjectCode + "|" + classCode + "|"
					+ teacherCode;
			ArrayList<String> list = makeDataToReport();
			int num = commonInfo.commonInfoMethods.reportKimatsuData(key,
					teacherCode, list);
			if (num > 0) {
				commonInfo.commonInfoMethods.emitKimatsuReportMail(schoolYear,
						subjectName, classCode, teacherName);
				refreshTable();
			}
			commonInfo.setUnsavedDataExistFlag(true);
		}
	}

	public void makeKimatsuReportFile() {
		if (nodePath.indexOf("kimatsu") < 0)
			return;

		int ret = chooser.showSaveDialog(commonInfo.getFrame());
		if (ret != JFileChooser.APPROVE_OPTION)
			return;
		File f = chooser.getSelectedFile();

		String schoolYear = parentTabbedPane
				.getValueFromColumnCodeMap("SCHOOL_YEAR");
		String subjectCode = parentTabbedPane
				.getValueFromColumnCodeMap("SUBJECT_CODE");
		String subjectName = parentTabbedPane
				.getDisplayFromColumnCodeMap("SUBJECT_CODE");
		String classCode = parentTabbedPane
				.getValueFromColumnCodeMap("CLASS_CODE");
		String teacherCode = parentTabbedPane
				.getValueFromColumnCodeMap("TEACHER_CODE");
		String teacherName = parentTabbedPane
				.getDisplayFromColumnCodeMap("TEACHER_CODE");

		try {
			PrintWriter fout = new PrintWriter(new BufferedWriter(
					new FileWriter(f)));
			String line = "1|" + subjectCode + "|" + classCode + "|"
					+ schoolYear;
			fout.println(line);
			fout.println("#  先頭行は「科目識別コード」なので変更しないで下さい。");
			fout.println("#----------------------------------------------------------------  ");
			fout.println("#  期末試験 成績報告記入フォーム ");
			fout.println("#  ");
			fout.println("#        科目名： " + subjectName);
			fout.println("#      担当教員： " + teacherName);
			fout.println("#  クラスコード： " + classCode);
			fout.println("#      開講年度： " + schoolYear + "年");
			fout.println("#  ");
			fout.println("# ( " + commonInfo.thisYear + "年 "
					+ commonInfo.thisMonth + "月 " + commonInfo.thisDay + "日 )");
			/*
			fout.println("#----------------------------------------------------------------  ");
			fout.println("#   以下のリストの「得点欄」に得点を記入して下さい。");
			fout.println("#  ");
			fout.println("#   不正な得点 (例えば100点以上) が記入された場合は、読込み時にデータ ");
			fout.println("#   がチェックされ、そのデータは読み飛ばされます。 ");
			fout.println("#  ");
			fout.println("#  ・ 得点は 0 ≦ 得点 ≦ 100 の範囲の整数でなければなりません。");
			fout.println("#  ・   60 ≦ 得点 ≦ 100 の場合は「合格」   ");
			fout.println("#        1 ≦ 得点 ≦ 59  の場合は「再試験」 ");
			fout.println("#        得点 == 0        の場合は「不合格」と判定されます。 ");
			fout.println("#   ( 60 ≦ 可 ≦ 69, 70 ≦ 良 ≦ 79,  80 ≦ 優 ≦ 89,  90 ≦ 秀 ≦ 100 )  ");
			fout.println("#  ");
			fout.println("#----------------------------------------------------------------  ");
			*/
			fout.println("#  先頭文字が # である行はコメント行と見なされます。");
			fout.println("#----------------------------------------------------------------  ");
			fout.println("#  ");
			fout.println("#  学生番号:  得点記入欄 :  氏名     :  所属          :学年  :在籍状況");
			fout.println("#  ");
			ArrayList<String> list = makeDataForReportForm();
			for (int i = 0; i < list.size(); i++) {
				line = list.get(i);
				fout.println(line);
			}
			fout.close();
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
		}
	}

	public void readKimatsuReportFile() {
		if (nodePath.indexOf("kimatsu") < 0)
			return;

		if (chooser.showOpenDialog(commonInfo.getFrame()) != JFileChooser.APPROVE_OPTION)
			return;
		File f = chooser.getSelectedFile();

		String schoolYear = parentTabbedPane
				.getValueFromColumnCodeMap("SCHOOL_YEAR");
		String subjectCode = parentTabbedPane
				.getValueFromColumnCodeMap("SUBJECT_CODE");
		String subjectName = parentTabbedPane
				.getDisplayFromColumnCodeMap("SUBJECT_CODE");
		String classCode = parentTabbedPane
				.getValueFromColumnCodeMap("CLASS_CODE");
		String teacherCode = parentTabbedPane
				.getValueFromColumnCodeMap("TEACHER_CODE");
		String teacherName = parentTabbedPane
				.getDisplayFromColumnCodeMap("TEACHER_CODE");

		try {
			BufferedReader fin = new BufferedReader(new FileReader(f));
			String line = fin.readLine();
			String[] tokens = line.split("\\|");
			String code2 = tokens[0];
			String subjectCode2 = tokens[1];
			String classCode2 = tokens[2];
			String schoolYear2 = tokens[3];
			if ((code2.equals("1")) && (subjectCode2.equals(subjectCode))
					&& (classCode2.equals(classCode))
					&& (schoolYear2.equals(schoolYear))) {
				readFromFile(fin);
			} else {
				commonInfo
						.showMessageLong("読み込もうとした期末試験報告ファイルの科目・クラス・年度 $ が報告ツールの科目・クラス・年度と一致しません。");
				fin.close();
				return;
			}
			fin.close();
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
		}
	}

	public void saveSaishiData() {
		if (nodePath.indexOf("saishi") < 0)
			return;

		Object[] msg = { " 記入された再試験の得点をデータベースにセーブします。 ", " ",
				" セーブされた得点データは「仮報告」得点として扱われ、 ", " 該当学生の再試験科目表に「仮成績」として表示されます。",
				" " };

		int ans = JOptionPane.showConfirmDialog(commonInfo.getFrame(), msg,
				"セーブの確認", JOptionPane.OK_CANCEL_OPTION);
		if (ans == JOptionPane.OK_OPTION) {
			if (commonInfo.unsavedDataExists()) {
				String schoolYear = parentTabbedPane
						.getValueFromColumnCodeMap("SCHOOL_YEAR");
				String subjectCode = parentTabbedPane
						.getValueFromColumnCodeMap("SUBJECT_CODE");
				String subjectName = parentTabbedPane
						.getDisplayFromColumnCodeMap("SUBJECT_CODE");
				String classCode = parentTabbedPane
						.getValueFromColumnCodeMap("CLASS_CODE");
				String teacherCode = parentTabbedPane
						.getValueFromColumnCodeMap("TEACHER_CODE");
				String teacherName = parentTabbedPane
						.getDisplayFromColumnCodeMap("TEACHER_CODE");

				String key = schoolYear + "|" + subjectCode + "|" + classCode
						+ "|" + teacherCode;
				ArrayList<String> list = makeDataToSave();
				int num = commonInfo.commonInfoMethods.saveSaishiData(key,
						teacherCode, list);
				commonInfo.setUnsavedDataExistFlag(true);
				if (num > 0) {
					refreshTable();
				}
			}
		}
	}

	public void doSaishiReport() {
		if (nodePath.indexOf("saishi") < 0)
			return;

		Object[] msg = { " 「成績報告」は、今回入力した得点データと以前に入力した ",
				" 「仮報告」状態の得点データを合わせて、「確定した得点」 ", " としてデータベースに格納する作業です。 ", " ",
				" 「不合格」の場合には「得点」を 0点、 「再試験」の場合 ",
				" には 1点 から 59点 の「得点」を記入して下さい。", " ",
				" 「得点」欄が空欄のままの学生 (成績が未確定の学生) について ",
				" は「成績報告」は実行されないので注意して下さい。   ", " ",
				" 「成績報告」が実行されると、学務係に対して「報告通知」が ",
				" 送られ、学務係は得点データをチェックした上で、得点データ ", " を「成績表」または「再試験表」に記入します。",
				" ", " 「報告済の成績」を修正するには、学務係に「試験成績修正願い」 ",
				" を提出し、「手作業」による得点修正を学務係に依頼することが ", " 必要となります。", " " };

		int ans = JOptionPane.showConfirmDialog(commonInfo.getFrame(), msg,
				"報告の確認", JOptionPane.OK_CANCEL_OPTION);
		if (ans == JOptionPane.OK_OPTION) {
			String schoolYear = parentTabbedPane
					.getValueFromColumnCodeMap("SCHOOL_YEAR");
			String subjectCode = parentTabbedPane
					.getValueFromColumnCodeMap("SUBJECT_CODE");
			String subjectName = parentTabbedPane
					.getDisplayFromColumnCodeMap("SUBJECT_CODE");
			String classCode = parentTabbedPane
					.getValueFromColumnCodeMap("CLASS_CODE");
			String teacherCode = parentTabbedPane
					.getValueFromColumnCodeMap("TEACHER_CODE");
			String teacherName = parentTabbedPane
					.getDisplayFromColumnCodeMap("TEACHER_CODE");

			String key = schoolYear + "|" + subjectCode + "|" + classCode + "|"
					+ teacherCode;
			ArrayList<String> list = makeDataToReport();
			int num = commonInfo.commonInfoMethods.reportSaishiData(key,
					teacherCode, list);
			if (num > 0) {
				commonInfo.commonInfoMethods.emitSaishiReportMail(schoolYear,
						subjectName, classCode, teacherName);
				refreshTable();
			}
			commonInfo.setUnsavedDataExistFlag(true);
		}
	}

	public void makeSaishiReportFile() {
		if (nodePath.indexOf("saishi") < 0)
			return;

		int ret = chooser.showSaveDialog(commonInfo.getFrame());
		if (ret != JFileChooser.APPROVE_OPTION)
			return;
		File f = chooser.getSelectedFile();

		String schoolYear = parentTabbedPane
				.getValueFromColumnCodeMap("SCHOOL_YEAR");
		String subjectCode = parentTabbedPane
				.getValueFromColumnCodeMap("SUBJECT_CODE");
		String subjectName = parentTabbedPane
				.getDisplayFromColumnCodeMap("SUBJECT_CODE");
		String classCode = parentTabbedPane
				.getValueFromColumnCodeMap("CLASS_CODE");
		String teacherCode = parentTabbedPane
				.getValueFromColumnCodeMap("TEACHER_CODE");
		String teacherName = parentTabbedPane
				.getDisplayFromColumnCodeMap("TEACHER_CODE");

		try {
			PrintWriter fout = new PrintWriter(new BufferedWriter(
					new FileWriter(f)));
			String line = "3|" + subjectCode + "|" + classCode + "|"
					+ teacherCode + "|" + schoolYear;
			fout.println(line);
			fout.println("#  先頭行は「科目識別コード」なので変更しないで下さい。");
			fout.println("#----------------------------------------------------------------  ");
			fout.println("#  再試験 成績報告記入フォーム ");
			fout.println("#  ");
			fout.println("#        科目名： " + subjectName);
			fout.println("#      担当教員： " + teacherName);
			fout.println("#  クラスコード： " + classCode);
			fout.println("#      開講年度： " + schoolYear + "年");
			fout.println("#  ");
			fout.println("# ( " + commonInfo.thisYear + "年 "
					+ commonInfo.thisMonth + "月 " + commonInfo.thisDay + "日 )");
			fout.println("#----------------------------------------------------------------  ");
			fout.println("#   以下のリストの「得点欄」に得点を記入して下さい。");
			fout.println("#  ");
			fout.println("#   不正な得点 (例えば100点以上) が記入された場合は、読込み時にデータ ");
			fout.println("#   がチェックされ、そのデータは読み飛ばされます。 ");
			fout.println("#  ");
			fout.println("#  ・ 得点は 0 ≦ 得点 ≦ 100 の範囲の整数でなければなりません。");
			fout.println("#  ・   60 ≦ 得点 ≦ 100 の場合は「合格」   ");
			fout.println("#        1 ≦ 得点 ≦ 59  の場合は「再試験」 ");
			fout.println("#        得点 == 0        の場合は「不合格」と判定されます。 ");
			fout.println("#   ( 60 ≦ 可 ≦ 69, 70 ≦ 良 ≦ 79,  80 ≦ 優 ≦ 89,  90 ≦ 秀 ≦ 100 )  ");
			fout.println("#  ");
			fout.println("#----------------------------------------------------------------  ");
			fout.println("#  先頭文字が # である行はコメント行と見なされます。");
			fout.println("#----------------------------------------------------------------  ");
			fout.println("#  ");
			fout.println("#  学生番号:  得点記入欄 :  氏名     :  所属          :学年  :在籍状況");
			fout.println("#  ");
			ArrayList<String> list = makeDataForReportForm();
			for (int i = 0; i < list.size(); i++) {
				line = (String) list.get(i);
				fout.println(line);
			}
			fout.close();
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
		}
	}

	public void readSaishiReportFile() {
		if (nodePath.indexOf("saishi") < 0)
			return;

		if (chooser.showOpenDialog(commonInfo.getFrame()) != JFileChooser.APPROVE_OPTION)
			return;
		File f = chooser.getSelectedFile();

		String schoolYear = parentTabbedPane
				.getValueFromColumnCodeMap("SCHOOL_YEAR");
		String subjectCode = parentTabbedPane
				.getValueFromColumnCodeMap("SUBJECT_CODE");
		String subjectName = parentTabbedPane
				.getDisplayFromColumnCodeMap("SUBJECT_CODE");
		String classCode = parentTabbedPane
				.getValueFromColumnCodeMap("CLASS_CODE");
		String teacherCode = parentTabbedPane
				.getValueFromColumnCodeMap("TEACHER_CODE");
		String teacherName = parentTabbedPane
				.getDisplayFromColumnCodeMap("TEACHER_CODE");

		try {
			BufferedReader fin = new BufferedReader(new FileReader(f));
			String line = fin.readLine();
			String[] tokens = line.split("\\|");
			String code2 = tokens[0];
			String subjectCode2 = tokens[1];
			String classCode2 = tokens[2];
			String teacherCode2 = tokens[3];
			String schoolYear2 = tokens[4];
			if ((code2.equals("3")) && (subjectCode2.equals(subjectCode))
					&& (classCode2.equals(classCode))
					&& (teacherCode2.equals(teacherCode))
					&& (schoolYear2.equals(schoolYear))) {
				readFromFile(fin);
			} else {
				commonInfo
						.showMessageLong("読み込もうとした再試験報告ファイルの科目・クラス・担当教員・年度 $ が報告ツールの科目・クラス・担当教員・年度と一致しません。");
				fin.close();
				return;
			}
			fin.close();
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
		}
	}

	public ArrayList<String> makeDataToSave() {
		ArrayList<String> list = new ArrayList<String>();
		for (int i = 0; i < tableModel.getRowCount(); i++) {
			String studentCode = tableModel.getCellValueAt(i, "学生番号");
			String marks = tableModel.getCellValueAt(i, "得点");
			String shoriStatus = tableModel.getCellValueAt(i, "処理状況");

			if ((!shoriStatus.equals("Y")) && (!shoriStatus.equals("F"))) {
				CellObject cobj = (CellObject) tableModel.getValueAt(i, "得点");
				if (cobj.isHighlighted()) {
					int gain = 0;
					try {
						gain = Integer.parseInt(marks);
					} catch (Exception e) {
						commonInfo.showMessage("不正な得点： 学生番号：" + studentCode
								+ "  得点： " + marks);
						continue;
					}
					if ((gain <= 100) && (gain >= 0)) {
						String data = studentCode + "|" + gain + "|K";
						list.add(data);
					} else if (gain > 100) {
						commonInfo.showMessage("不正な得点： 学生番号：" + studentCode
								+ "  得点： " + marks);
						continue;
					}
				}
			}
		}
		return list;
	}

	public ArrayList<String> makeDataToReport() {
		ArrayList<String> list = new ArrayList<String>();
		int count = 0;
		int rcount = 0;
		for (int i = 0; i < tableModel.getRowCount(); i++) {
			String studentCode = tableModel.getCellValueAt(i, "学生番号");
			String marks = tableModel.getCellValueAt(i, "得点");
			String shoriStatus = tableModel.getCellValueAt(i, "処理状況");

			if ((!shoriStatus.equals("Y")) && (!shoriStatus.equals("F"))) {
				rcount++;
				int gain = 0;
				try {
					gain = Integer.parseInt(marks);
				} catch (Exception e) {
					commonInfo.showMessage("不正な得点： 学生番号：" + studentCode
							+ "  得点： " + marks);
					continue;
				}
				if ((gain <= 100) && (gain >= 0)) {
					String data = studentCode + "|" + gain + "|Y";
					list.add(data);
					count++;
				} else if (gain > 100) {
					commonInfo.showMessage("不正な得点： 学生番号：" + studentCode
							+ "  得点： " + marks);
					continue;
				}
			}
		}
		if (rcount > count) {
			commonInfo.showMessageLong(" " + (rcount - count)
					+ " 人の学生の成績データが報告されていません。$ 未報告のままで放置しないようご注意下さい。");
		}
		return list;
	}

	public ArrayList<String> makeDataForReportForm() {
		ArrayList<String> list = new ArrayList<String>();
		for (int i = 0; i < tableModel.getRowCount(); i++) {
			String studentCode = tableModel.getCellValueAt(i, "学生氏名");
			String studentName = tableModel.getCellDisplayAt(i, "学生氏名");
			String marks = tableModel.getCellValueAt(i, "得点");
			String department = tableModel.getCellDisplayAt(i, "学科");
			String gakunen = tableModel.getCellDisplayAt(i, "学年");
			String shoriStatus = tableModel.getCellValueAt(i, "処理状況");
			String shoriName = tableModel.getCellDisplayAt(i, "処理状況");
			String studentStatus = tableModel.getCellDisplayAt(i, "在籍状況");

			String line;
			if ((!shoriStatus.equals("Y")) && (!shoriStatus.equals("F"))) {
				int gain = 0;
				try {
					gain = Integer.parseInt(marks);
				} catch (Exception e) {
					commonInfo.showMessage("不正な得点： 学生番号：" + studentCode
							+ "  得点： " + marks);
					continue;
				}
				if (gain < 0) {
					line = " " + studentCode + ":      \t: " + studentName
							+ " : " + department + " : " + gakunen + " : "
							+ studentStatus;
					list.add(line);
				} else if (gain < 10) {
					line = " " + studentCode + ":    " + gain + "\t: "
							+ studentName + " : " + department + " : "
							+ gakunen + " : " + studentStatus;
					list.add(line);
				} else if (gain < 100) {
					line = " " + studentCode + ":   " + gain + "\t: "
							+ studentName + " : " + department + " : "
							+ gakunen + " : " + studentStatus;
					list.add(line);
				} else if (gain == 100) {
					line = " " + studentCode + ":  " + gain + "\t: "
							+ studentName + " : " + department + " : "
							+ gakunen + " : " + studentStatus;
					list.add(line);
				} else {
					commonInfo.showMessage("不正な得点： 学生番号：" + studentCode
							+ "  得点： " + marks);
					continue;
				}
			}
		}
		return list;
	}

	public void readFromFile(BufferedReader fin) {
		int errorCount = 0;
		StringBuffer sbuf = new StringBuffer();

		HashMap<String, String> studentNameMap = new HashMap<String, String>();
		HashMap<String, Integer> studentRowMap = new HashMap<String, Integer>();
		HashMap<String, String> shoriStatusMap = new HashMap<String, String>();
		for (int i = 0; i < getRowCount(); i++) {
			String studentCode = tableModel.getCellValueAt(i, "学生氏名");
			String studentName = tableModel.getCellDisplayAt(i, "学生氏名");
			String shoriStatus = tableModel.getCellValueAt(i, "処理状況");
			studentRowMap.put(studentCode, new Integer(i));
			studentNameMap.put(studentCode, studentName);
			if (shoriStatus.equals("Y")) {
				shoriStatusMap.put(studentCode, "(報告済)");
			} else if (shoriStatus.equals("F")) {
				shoriStatusMap.put(studentCode, "(処理済)");
			}
		}

		try {
			String line;
			while ((line = fin.readLine()) != null) {
				if (!(line.startsWith("#"))) {
					String[] tokens = line.split("\\:");
					String studentCode = tokens[0].trim();
					String marks = tokens[1].trim();
					String studentName = studentNameMap.get(studentCode);
					if (!shoriStatusMap.containsKey(studentCode)) {
						if (marks.length() > 0) {
							int gain = 0;
							boolean errorFlag = false;
							try {
								gain = Integer.parseInt(marks);
								if ((gain < 0) || (gain > 100)) {
									errorFlag = true;
								}
							} catch (Exception ex) {
								errorFlag = true;
							}
							if (errorFlag) {
								commonInfo.showMessage("学生: " + studentName
										+ "(" + studentCode + ") の得点: " + marks
										+ " は不正です。");
								errorCount++;
								sbuf.append("$" + line);
							} else {
								try {
									Integer intObj = (Integer) studentRowMap
											.get(studentCode);
									int row = intObj.intValue();
									setMarks(row, gain);
								} catch (Exception ex) {
									errorCount++;
									sbuf.append("$" + line);
								}
							}
						}
					} else {
						commonInfo.showMessage("学生: " + studentName + "("
								+ studentCode + ") の成績は報告済または処理済です。");
						errorCount++;
						sbuf.append("$" + line);
					}
				}
			}
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
		}
		if (errorCount > 0) {
			Object[] obj = { "成績報告入力ファイルには不正な得点データが存在しています。", " ",
					"  読み飛ばした報告データをファイルに出力しますか？    ==> 了解 ",
					"  　　　   　 ファイル出力をせずに先に進みますか？    ==> 取消 ", " " };
			int ans = JOptionPane.showConfirmDialog(commonInfo.getFrame(), obj,
					"読込めなかった得点", JOptionPane.OK_CANCEL_OPTION);
			if (ans == JOptionPane.OK_OPTION) {
				chooser.setSelectedFile(null);
				if (chooser.showSaveDialog(commonInfo.getFrame()) == JFileChooser.APPROVE_OPTION) {
					File g = chooser.getSelectedFile();
					try {
						PrintWriter gout = new PrintWriter(new BufferedWriter(
								new FileWriter(g)));
						String data = sbuf.toString();
						String[] lines = data.split("\\$");
						for (String line : lines) {
							gout.println(line);
						}
						gout.close();
					} catch (Exception e) {
					}
				}
			}
		}
	}
}
