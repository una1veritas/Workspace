package common;

import clients.*;
//import java.io.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.border.*;
import javax.swing.*;

public class PopupInput extends JDialog {
	Container pane;
	// CommonInfoBase commonInfo;
	CommonInfo commonInfo;
	ReportTableView hokokuTable;

	/*JTextField*/ JLabel scodeTF;
	/*JTextField*/ JLabel snameTF;
	JTextField gainsTF;
	JButton okButton;
	JButton cancelButton;

	String studentCode;
	String studentName;
	String gainsText;
	int gain;

	/****
	 * gain = -1 ==> 空欄 gain = -9 ==> 入力不可
	 ***/

	public PopupInput(CommonInfo /* Base */commonInfo, ReportTableView tableView) {
		this.commonInfo = commonInfo;
		this.hokokuTable = tableView;

		addMouseListener(new MMouseListener());
		addFocusListener(new FFocusListener());
		FKeyListener keyListener = new FKeyListener();
		addKeyListener(keyListener);

		pane = getContentPane();		
		pane.setLayout(new BoxLayout(pane, BoxLayout.X_AXIS));

		JPanel bordered = new JPanel();
		bordered.setLayout(new BoxLayout(bordered, BoxLayout.X_AXIS));
  		scodeTF = new JLabel(); //new JTextField();
		scodeTF.setBorder(new TitledBorder(" 学生番号 "));
		bordered.add(scodeTF);

		snameTF = new JLabel(); //JTextField();
		snameTF.setBorder(new TitledBorder(" 学生氏名 "));
		bordered.add(snameTF);
		bordered.add(Box.createRigidArea(new Dimension(5, 5)));
		gainsTF = new JTextField();
		gainsTF.setBorder(new TitledBorder(" 得点 "));
		gainsTF.setEditable(false);
		gainsTF.setColumns(4);
		bordered.add(gainsTF);
		bordered.add(Box.createRigidArea(new Dimension(10, 10)));

		ActionListener listener = new ActionModel();
		JPanel buttons = new JPanel();
		buttons.setLayout(new BoxLayout(buttons, BoxLayout.X_AXIS));
		buttons.add(Box.createRigidArea(new Dimension(5, 5)));
		okButton = new JButton("記入");
		okButton.addActionListener(listener);
		okButton.addKeyListener(keyListener);
		buttons.add(okButton);
		buttons.add(Box.createRigidArea(new Dimension(5, 5)));
		cancelButton = new JButton("Cancel");
		cancelButton.addActionListener(listener);
		cancelButton.addKeyListener(keyListener);
		buttons.add(cancelButton);
		buttons.add(Box.createRigidArea(new Dimension(10, 10)));
		bordered.add(buttons);

		pane.add(bordered);
		
		setSize(450,72);
		setTitle("得点入力");
		setFocusable(true);
	}

	class FKeyListener extends KeyAdapter {
		public void keyPressed(KeyEvent e) {
			char ch = e.getKeyChar();
			if (gain != -9) {
				if (Character.isDigit(ch)) {
					String str = gainsTF.getText().trim() + String.valueOf(ch);
					int val = -1;
					try {
						val = Integer.parseInt(str);
					} catch (java.lang.NumberFormatException ex) {}
					if ( val >= 0 && val <= 100 ) {
						gainsTF.setText(str);
					}
				} else if ( Character.toUpperCase(ch) == 'E' ) {
					gainsTF.setText("");
				} else if (e.getKeyCode() == KeyEvent.VK_DELETE
						|| e.getKeyCode() == KeyEvent.VK_BACK_SPACE) {
					String s = gainsTF.getText();
					if (s.length() > 0) {
						gainsTF.setText(s.substring(0, s.length() - 1));
					}
				}
			}
			if (e.getKeyCode() == KeyEvent.VK_ENTER || e.getKeyCode() == KeyEvent.VK_TAB ) {
				setMarks();
			} else if (e.getKeyCode() == KeyEvent.VK_DOWN) {
				hokokuTable.popupListener.selectNext();
			} else if (e.getKeyCode() == KeyEvent.VK_UP) {
				hokokuTable.popupListener.selectPrevious();
			}
			else if ( e.getKeyCode() == KeyEvent.VK_ESCAPE ) {
				hokokuTable.popupListener.removePopupInput();
			}
		}
	}

	class MMouseListener extends MouseAdapter {
		public void mouseEntered(MouseEvent e) {
			requestFocus();
		}

		public void mousePressed(MouseEvent e) {
			requestFocus();
		}

		public void mouseClicked(MouseEvent e) {
			requestFocus();
		}
	}

	class FFocusListener extends FocusAdapter {
		public void focusGained(FocusEvent e) {
			gainsTF.setBackground(new Color(245, 255, 250));
		}

		public void focusLost(FocusEvent e) {
			gainsTF.setBackground(Color.lightGray);
		}
	}

	public void setInfo(String scode, String sname, String marks,
			String reported) {
		studentCode = scode;
		studentName = sname;
		gainsText = marks;
		scodeTF.setText(" " + studentCode);
		snameTF.setText(" " + studentName);

		if (reported.equals("Y")) {
			gainsTF.setText(" (報告済) ");
			gain = -9;
		} else if (reported.equals("F")) {
			gainsTF.setText(" (処理済) ");
			gain = -9;
		} else {
			try {
				gain = Integer.parseInt(gainsText.trim());
				if (gain == -1) {
					gainsTF.setText(" ");
				} else {
					gainsTF.setText(" " + gain);
				}
			} catch (NumberFormatException ex) {
				gain = -1;
				gainsTF.setText(" ");
			}
		}
		commonInfo.timerRestart();
	}

	public void setMarks() {
		int newGain;
		gainsText = gainsTF.getText().trim();
		if (gainsText.length() == 0) {
			if (gain >= 0) {
				gain = -1;
			}
		} else {
			try {
				newGain = Integer.parseInt(gainsText);
				if (gain != newGain) {
					gain = newGain;
				}
			} catch (NumberFormatException ex) {
				gain = -1;
			}
		}
		hokokuTable.setMarks(gain);
		hokokuTable.popupListener.selectNext();
	}

	class ActionModel implements ActionListener {
		public void actionPerformed(ActionEvent e) {
			JButton b = (JButton) e.getSource();
			if (b.getText().equals("記入")) {
				setMarks();
			} else if (b.getText().equals("Cancel")) {
				setVisible(false);
			}
		}
	}
}
