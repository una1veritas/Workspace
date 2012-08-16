package clients;

import java.awt.Font;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.net.URI;

import javax.swing.*;
import javax.swing.border.*;

import common.*;
//import javax.swing.*;

public class KyomuTool extends RootPanelBase {
	//  private String rootServiceName = "MinnanoTool";
	private CommonInfo infoBase;


	public KyomuTool(URI tool) {
		super("教務情報システム", 1000, 700);

		System.out.println("Involking " + tool.getScheme());
		infoBase = new CommonInfo(frame, tool); //rootServiceName);

		increaseProgressValue("サーバをよんでます...");
		infoBase.connectToServer();

		increaseProgressValue("はじめます...");
		infoBase.init();

		increaseProgressValue("パスワードをたしかめます... ");
		infoBase.serverConn.checkUserQualification();

		increaseProgressValue("サービス画面を構築中 ... ");
		TabbedPaneBase tabbedPane = new TabbedPaneBase(infoBase.rootServiceName,
				"root",
				infoBase, null, null);
		increaseProgressValue("もう少し待って ... ");
		tabbedPane.pageOpened();
		showTabbedPane(tabbedPane);

		expandMenu();
	}

	void expandMenu() {
		JMenuBar myMenuBar = frame.getJMenuBar();
		JMenu myMenu = myMenuBar.getMenu(0);
		JMenuItem myItem;
		myItem = new JMenuItem("パスワードの変更 / Change password...");
		myItem.addActionListener( new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				changePassword();
			}
		} );
		myMenu.insert(myItem, 0);
		myItem = new JMenuItem("利用資格・権限の変更 / Switch role");
		myItem.setEnabled(false);
		myMenu.insert(myItem,1);
//		myMenu.addSeparator();
	}
	
	public boolean changePassword() {		
		//		protected JButton quitButton;
//		JOptionPane myOptionPane;
		JTextField idField = new JTextField(); 
		idField.setFont(new Font("DialogInput", Font.PLAIN, 14));
		idField.setBorder(new EmptyBorder(2, 5, 2, 2));
		JPasswordField currentPwd = new JPasswordField(16);
		JPasswordField newPwd1 = new JPasswordField(16);
		JPasswordField newPwd2 = new JPasswordField(16);
		
		int act1 = JOptionPane.showConfirmDialog(null, 
	    		currentPwd,
	    		"現在のパスワードを入力してください",
	    		JOptionPane.OK_CANCEL_OPTION);
	    if(act1 == JOptionPane.CANCEL_OPTION) {
	    	JOptionPane.showMessageDialog(null,"Change password canceled.");
	    	return false;
	    } else 
	    	JOptionPane.showMessageDialog(null,"あらためてパスワードを確認できました．");  

        //Create an array of the text and components to be displayed.
        Object[] array = {"あたらしいパスワードを入力してください。", 
        		"(確認のため同じパスワードを二度入力してください)", 
        		newPwd1, newPwd2};

        //Create an array specifying the number of dialog buttons
        //and their text.
        Object[] options = {"Change", "Cancel"};

		int ans = JOptionPane.showOptionDialog(null, 
				array,
				"パスワード変更画面", 
				JOptionPane.OK_CANCEL_OPTION,
				JOptionPane.QUESTION_MESSAGE,
				null,
				options,
				options[0]);
		if (ans == JOptionPane.OK_OPTION) {
			String oldPass = (new String(currentPwd.getPassword())).trim();
			String newPass1 = (new String(newPwd1.getPassword())).trim();
			String newPass2 = new String(newPwd2.getPassword());
			if ((infoBase.login_id.length() > 0) && (oldPass.length() > 5)) {
				String msg = "CHECKPASSWORD|" + infoBase.rootServiceName + "|PASSWORD";
				String param = infoBase.login_id  + "|" + oldPass;
				try {
					String answer = infoBase.serverConn.query(msg, param);
					if (answer.startsWith("success")) {
						int index = answer.indexOf("|");
						System.err.println(answer);
						infoBase.STAFF_CODE = answer.substring(index+1);
						answer = "";
						if (newPass1.equals(newPass2)) {   
							msg = "CHANGEPASSWORD|" + infoBase.rootServiceName + "|PASSWORD";
							param = infoBase.login_id + "|" + oldPass + "|" + newPass1;
							answer = infoBase.serverConn.query(msg, param);
						}
						if (answer.equals("success")) { 
							return true;
						} else if ( answer.equals("") ){
							infoBase.showMessage("２つの新パスワードが一致しません。");
						} else {
							infoBase.showMessage("パスワードの変更に失敗しました。");
						}
					} else {
						infoBase.showMessage("ユーザ認証に失敗しました。");
					}
				} catch (Exception e) {
					infoBase.showMessage("サーバとの通信に失敗: " + e.toString());
					System.exit(0);
				} 
			} else {
				infoBase.showMessage("ID または Password の形式が不正です。");
			}
		}
		return false;
	}

}
