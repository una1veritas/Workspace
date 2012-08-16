package common;

//import javax.swing.border.*;
import javax.swing.*;
//import javax.swing.border.EmptyBorder;
//import javax.swing.border.TitledBorder;

import java.awt.event.*;
import java.awt.*;
//import java.util.*;
//import java.io.*;

public class RootPanelBase extends JPanel {
	public  JFrame frame;
	private int width;
	private int height;
//	private String frameTitle;
	private Dimension screenSize;
	private JLabel progressLabel;
	private JProgressBar progressBar;
	int totalPreparations = 0;   
	int currentProgressValue = 0;

	public RootPanelBase(String frameTitle, int width, int height) { 
//		this.frameTitle = frameTitle;
		this.width = width;
		this.height = height;

//		String os = System.getProperty("os.name");
//		String vers = System.getProperty("java.version");

		frame = new JFrame( frameTitle );
		frame.addWindowListener(new WindowAdapter() {
			public void windowClosing(WindowEvent e) {
				System.exit(0);
			}} );    
		JOptionPane.setRootFrame(frame);    
		JPanel progressPanel = new JPanel() {
			public Insets getInsets() {
				return new Insets(40,30,20,30);
			}
		};
		progressPanel.setBackground(new Color(230, 230, 250));
		progressPanel.setLayout(new BoxLayout(progressPanel, BoxLayout.Y_AXIS));

		frame.getContentPane().add(progressPanel, BorderLayout.CENTER);
		Dimension d = new Dimension(400, 20);
		progressLabel = new JLabel("Start loading, please wait...");
		progressLabel.setAlignmentX(CENTER_ALIGNMENT);
		progressLabel.setMaximumSize(d);
		progressLabel.setPreferredSize(d);
		progressPanel.add(progressLabel);
		progressPanel.add(Box.createRigidArea(new Dimension(1,20)));    
		progressBar = new JProgressBar();
		progressLabel.setLabelFor(progressBar);
		progressBar.setAlignmentX(CENTER_ALIGNMENT);
		progressBar.setMinimum(0);
		progressBar.setMaximum(totalPreparations);
		progressBar.setValue(0);
		progressPanel.add(progressBar);

		screenSize = Toolkit.getDefaultToolkit().getScreenSize();
		if (width > screenSize.width - 20) {
			width = screenSize.width - 20;
		}
		if (height > screenSize.height - 20) {
			height = screenSize.height - 20;
		}

		frame.setSize(width, height);
		frame.setLocation(screenSize.width/2 - width/2, screenSize.height/2 - height/2);
		frame.setCursor(Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR));
		frame.setVisible(true);

		JMenuBar menuBar = new JMenuBar();
		frame.setJMenuBar( menuBar );
		JMenu menu = new JMenu("Menu");
		menuBar.add(menu);
		JMenuItem item;
		menu.addSeparator();
		item = new JMenuItem("Quit");
		item.addActionListener( new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				System.exit(0);
			}
		} );
		menu.add(item);
		setLayout(new BorderLayout()); 

		totalPreparations = 5;    
		progressBar.setMaximum(totalPreparations);
		currentProgressValue = 0;
	}

	public void increaseProgressValue(String text) {  
		progressLabel.setText(text);
		progressBar.setValue(++currentProgressValue); 
	}

	public void showTabbedPane(TabbedPaneBase tabbedPane) { 
		frame.getContentPane().removeAll();
		frame.getContentPane().add(tabbedPane, BorderLayout.CENTER);
		frame.setSize(width, height);
		frame.setLocation(screenSize.width/2 - width/2, screenSize.height/2 - height/2);
		frame.setCursor(Cursor.getPredefinedCursor(Cursor.DEFAULT_CURSOR));
		frame.validate();
		frame.repaint();
	} 
}
