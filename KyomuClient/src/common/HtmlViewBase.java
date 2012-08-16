package common;

import java.util.*;
import java.net.*;
import javax.swing.*;
import javax.swing.event.*;
import java.awt.*;
import java.awt.event.*;
import clients.*;

public class HtmlViewBase extends JPanel {
	private JEditorPane editPane;
	private JButton backButton = new JButton(" ← 戻る ");
	private JTextField addressTF = new JTextField(36);
	private Stack<URL> history = new Stack<URL>();
	private JToolBar toolBar = new JToolBar();
	private CommonInfo commonInfo;
	private String urlPath;

	public HtmlViewBase(String serviceName, String nodePath, String panelID,
			CommonInfo commonInfo, TabbedPaneBase tabbedPane,
			DataPanelBase dataPanel) {
		this.commonInfo = commonInfo;
		String text = commonInfo.getHtmlViewStruct(serviceName, panelID);
		StringTokenizer stk = new StringTokenizer(text, "|");
		urlPath = stk.nextToken();
		setMainComponent();
	}

	public HtmlViewBase(CommonInfo commonInfo, String urlPath) {
		this.commonInfo = commonInfo;
		this.urlPath = urlPath;
		setMainComponent();
	}

	public void setMainComponent() {
		setLayout(new BorderLayout());

		editPane = new JEditorPane();
		editPane.setEditable(false);
		editPane.addHyperlinkListener(new HtmlPageChanged());
		editPane.setContentType("text/html; charset=EUC-JP");

		backButton.setEnabled(false);
		backButton.addActionListener(new EnterPage());

		addressTF.setFont(new Font("DialogInput", Font.PLAIN, 12));
		addressTF.setForeground(Color.blue);
		addressTF.addActionListener(new EnterPage());

		toolBar.add(backButton);
		toolBar.add(addressTF);
		toolBar.setPreferredSize(new Dimension(1000, 32));
		toolBar.setMaximumSize(new Dimension(1000, 32));
		add("North", toolBar);
		add("Center", new JScrollPane(editPane));
	}

	public void pageOpened() {
		try {
			URL url = new URL(urlPath);
			addressTF.setText(" " + url.toString());
			editPane.setPage(url);
			editPane.setCaretPosition(0);
		} catch (Exception e) {
			commonInfo.showMessage(e.toString());
		}
	}

	class EnterPage implements ActionListener {

		public void actionPerformed(ActionEvent e) {
			if (e.getSource() == addressTF) {
				if (editPane.getPage() != null) {
					history.push(editPane.getPage());
					backButton.setEnabled(true);
				}
				try {
					editPane.setPage(addressTF.getText().trim());
				} catch (Exception err) {
					commonInfo.showMessage(err.toString());
				}
			} else if (e.getSource() == backButton) {
				try {
					URL url = history.pop();
					addressTF.setText(" " + url.toString());
					editPane.setPage(url);
				} catch (Exception err) {
					commonInfo.showMessage(err.toString());
				}
				if (history.isEmpty()) {
					backButton.setEnabled(false);
				}
			}
		}
	}

	class HtmlPageChanged implements HyperlinkListener {
		public void hyperlinkUpdate(HyperlinkEvent e) {
			if (e.getEventType() != HyperlinkEvent.EventType.ACTIVATED)
				return;

			String url = e.getURL().toString();
			String file = e.getURL().getFile();
			if (file.endsWith(".pdf")) {
			//	new ReadPdf(e.getURL(), commonInfo).start();
			} else {
				history.push(editPane.getPage());
				backButton.setEnabled(true);
				try {
					addressTF.setText(" " + url);
					editPane.setPage(url);
				} catch (Exception ex) {
					commonInfo.showMessage(ex.toString());
				}
			}
		}
	}

}
