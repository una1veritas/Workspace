package clients;

import common.*;

import java.awt.Font;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.*;

import javax.swing.*;
import javax.swing.border.*;
//import java.io.*;
//import java.awt.*;
//import java.awt.event.*;

public class ServerConnection extends ServerConnectionBase { 
	private String login_id = "";
	private String password = "";
	private String card_id = "";

	public ServerConnection(String serverHost, 
			int serverPort,
			CommonInfo commonInfo) { 
		super(serverHost, serverPort, commonInfo);
	}

	public void checkUserQualification() {
		boolean res = checkPassword();
		setQualification(res);
		if (res == false) {
			System.exit(0);
		}
		if ( commonInfo.rootServiceName.equals("TeacherTool") ) {
			getTeacherAttrib();
		} else if ( commonInfo.rootServiceName.equals("StaffTool") ) {
			getStaffAttrib();
		} else {
			getStudentAttrib();
		}
	}

	public boolean checkPassword() {
		JButton quitButton = new JButton("Quit/終了");
		//	    quitButton.setBackground(Color.pink);
		quitButton.addActionListener(new ActionListener() {     
			public void actionPerformed(ActionEvent e) {
				System.exit(0);
			}
		} );      
		JTextField idField = new JTextField();
		idField.setFont(new Font("DialogInput", Font.PLAIN, 14));
		idField.setBorder(new EmptyBorder(2, 5, 2, 2));
		JPasswordField passwdField = new JPasswordField();
		passwdField.setBorder(new EmptyBorder(2, 5, 2, 2));
		passwdField.setBorder( new TitledBorder( " Password " ) );
		passwdField.setEchoChar('*');
		passwdField.setFont(new Font("DialogInput", Font.PLAIN, 14));

		idField.setBorder( new TitledBorder( " ユーザーＩＤ " ) );
		passwdField.setEchoChar('*');
		while (true) {
			idField.setText("");
			passwdField.setText("");
			Object[] contents = {idField, passwdField};      
			Object[] options = { "ログイン/Login", "やめる/Give up"};
			int ans = JOptionPane.showOptionDialog(commonInfo.getFrame(), 
					contents, 
					"教務情報システム みんな用 P-110820", 
					JOptionPane.YES_NO_OPTION,
					JOptionPane.QUESTION_MESSAGE,
					null,
					options,
					options[0]);

			System.out.println(ans);

			if ((ans == JOptionPane.NO_OPTION) || ans == JOptionPane.CLOSED_OPTION ) {
				// do nothing and quit
				/*
				JOptionPane.showMessageDialog(commonInfo.getFrame(),
						"教務情報システム みんな用　を終了します．", 
						"確認メッセージ", JOptionPane.WARNING_MESSAGE);
						*/
				//commonInfo.showMessage("教務情報システム みんな用　を終了します．");
				System.exit(0);
			} else if (ans == JOptionPane.YES_OPTION ) {
				login_id = idField.getText().trim();
				password = (new String(passwdField.getPassword())).trim();

				if ((login_id.length() > 0) && (password.length() > 5)) {
					//String message = "CHECKPASSWORD|" + commonInfo.rootServiceName + "|PASSWORD:" + login_id  + "|" + password;
					String key = "CHECKPASSWORD|" + commonInfo.rootServiceName + "|PASSWORD";
					String value = login_id  + "|" + password;
					try {
						String answer = commonInfo.serverConn.queryCommon(key, value);
						if (answer.startsWith("success")) {
							commonInfo.login_id = login_id;
							int index = answer.indexOf("|");
							card_id = answer.substring(index+1);
							return true;
						} else {
							commonInfo.showMessage(answer);
						}
					} catch (Exception e) {
						commonInfo.showMessage("サーバとの通信に失敗: " + e.toString());
						System.exit(0);
					} 
				}
			} 
		}
	} 

	private void getTeacherAttrib() {
		try {
			String message = "QUERYATTRIB|" + commonInfo.rootServiceName + "|STAFF:"+ card_id;
			cout.println(message);
			String answer = cin.readLine();

			StringTokenizer stk = new StringTokenizer(answer, "|");
			commonInfo.STAFF_CODE    = stk.nextToken();
			commonInfo.STAFF_NAME    = stk.nextToken();
			commonInfo.STAFF_ATTRIB  = stk.nextToken();
			commonInfo.STAFF_QUALIFICATION = stk.nextToken();
			commonInfo.MAIL_ADDRESS = stk.nextToken();
			commonInfo.LOCAL_ATTRIB = stk.nextToken();
			commonInfo.STAFF_DEPARTMENT = stk.nextToken();      
			commonInfo.addStaffAttribToCommonCodeMap();

			int qual = Integer.parseInt(commonInfo.STAFF_QUALIFICATION);
			if (qual < 2) {
				commonInfo.showMessageLong("このツールの利用者は、飯塚キャンパスの講師以上の $ アクセス権限を持つ職員に限られています。");
				System.exit(0);
			}	
		} catch (Exception e) {
			commonInfo.showMessage("職員属性の取得に失敗: " + e.toString());
			System.exit(0);
		} 
	}

	private void getStaffAttrib() {
		try {
			String message = "QUERYATTRIB|" + commonInfo.rootServiceName + "|STAFF:"+card_id;
			cout.println(message);
			String answer = cin.readLine();

			StringTokenizer stk = new StringTokenizer(answer, "|");
			commonInfo.STAFF_CODE    = stk.nextToken();
			commonInfo.STAFF_NAME    = stk.nextToken();
			commonInfo.STAFF_ATTRIB  = stk.nextToken();
			commonInfo.STAFF_QUALIFICATION = stk.nextToken();
			commonInfo.MAIL_ADDRESS = stk.nextToken();
			commonInfo.LOCAL_ATTRIB = stk.nextToken();
			commonInfo.STAFF_DEPARTMENT = stk.nextToken();

			commonInfo.addStaffAttribToCommonCodeMap();
		} catch (Exception e) {
			commonInfo.showMessage("職員属性の取得に失敗: " + e.toString());
			System.exit(0);
		} 
	}

	private void getStudentAttrib() {
		try {
			String message = "QUERYATTRIB|" + commonInfo.rootServiceName + "|STUDENT:"+login_id;
			cout.println(message);
			String answer = cin.readLine();

			StringTokenizer stk = new StringTokenizer(answer, "|");
			commonInfo.STUDENT_CODE       = stk.nextToken();
			commonInfo.STUDENT_NAME       = stk.nextToken();
			commonInfo.STUDENT_STATUS     = stk.nextToken();
			commonInfo.STUDENT_FACULTY    = stk.nextToken();
			commonInfo.STUDENT_DEPARTMENT = stk.nextToken();
			commonInfo.STUDENT_COURSE     = stk.nextToken();
			commonInfo.STUDENT_COURSE_2   = stk.nextToken();
			commonInfo.STUDENT_GAKUNEN    = stk.nextToken();
			commonInfo.STUDENT_CURRICULUM_YEAR    = stk.nextToken();
			commonInfo.STUDENT_SUPERVISOR         = stk.nextToken();
			commonInfo.STUDENT_SUPERVISOR_NAME    = stk.nextToken();

			commonInfo.addStudentAttribToCommonCodeMap();
		} catch (Exception e) {
			commonInfo.showMessage("学生属性の取得に失敗: " + e.toString());
			System.exit(0);
		} 
	}


}
