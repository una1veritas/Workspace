package common;
import java.util.*;
import javax.swing.*;
import javax.swing.border.*;
import java.awt.*;      
import java.awt.event.*; 
//import java.io.*;
//import java.util.*;
//import xml.*;
//import javax.swing.text.*;
//import javax.swing.tree.*;
//import javax.swing.event.*;
//import javax.xml.parsers.*;
//import org.xml.sax.*;
//import org.w3c.dom.*;
import clients.*;

public class EnqueteSheet extends JPanel {
  protected String tableViewType;
  protected String serviceName;
  protected String nodePath;
  protected String panelID;

  public CommonInfo commonInfo;
  public TabbedPaneBase parentTabbedPane;
  public DataPanelBase  parentDataPanel;
  public Font font;
  public Font font2;
  public Color fgColor;
  public Color fgColor2;
  public Color bgColor;

  private ActionListener listener = new ButtonListener();
  
  private String[][][] enqueteItem
    = { { { "この授業の「授業項目」に関して、下記の質問に回答して下さい。", "" },
	  { "「出欠データ」を参考にして「授業を欠席した授業項目」を選択して下さい。", 
	    "「授業項目」を選択肢とする回答画面を表示" },
	  { "「特に興味を持った授業項目」を選択して下さい。", 
	    "「授業項目」を選択肢とする回答画面を表示" }, 
	  { "「全く興味がなかった授業項目」を選択して下さい。", 
	    "「授業項目」を選択肢とする回答画面を表示" }, 
	  { "「授業内容を十分に理解できなかった授業項目」を選択して下さい。", 
	    "「授業項目」を選択肢とする回答画面を表示" }, 
	  { "「予備的知識が不足していると感じた授業項目」を選択して下さい。", 
	    "「授業項目」を選択肢とする回答画面を表示" } },
	{ { "この授業の「達成目標」に関して、下記の質問に回答して下さい。", "" },
	  { "「十分に達成することが出来たと思う達成目標」を選択して下さい。", 
	    "「達成目標」を選択肢とする回答画面を表示" },
	  { "「達成することは出来なかったと思う達成目標」を選択して下さい。", 
	    "「達成目標」を選択肢とする回答画面を表示" } },
	{ { "この授業の進行に関して、下記の質問に回答して下さい。", "" },
	  { "担当教員の説明は適切で分りやすかったですか？", 
	    "「選択肢」の選択画面を表示する" },
	  { "担当教員の授業のスピードは適切でしたか？", 
	    "「選択肢」の選択画面を表示する" },
	  { "この授業に対し思ったことを書いて下さい。", 
	    "「テキスト回答入力画面」を表示する" } } };
  
  private String[] choice = { "適切だった", "普通だった", "適切でなかった" };
    
  private int num1 = enqueteItem[0].length;
  private int num2 = enqueteItem[1].length;
  private int num3 = enqueteItem[2].length;  
  private JTextArea textArea;

  private JButton[] subjectItemsButton = new JButton[num1];
  private JButton[] objectivesButton = new JButton[num2];
  private JButton[] choiceButton = new JButton[num3];

  private ArrayList<String> listOfSubjectItems = new ArrayList<String>();
  private ArrayList<String> listOfObjectives = new ArrayList<String>();
  private ArrayList<String> listOfChoices = new ArrayList<String>();

  public EnqueteSheet(String tableViewType,
		      String serviceName,
		      String nodePath,
		      String panelID,
		      CommonInfo commonInfo, 
		      TabbedPaneBase parentTabbedPane,
		      DataPanelBase parentDataPanel) {
    this.tableViewType = tableViewType;
    this.serviceName = serviceName;
    this.nodePath = nodePath;
    this.panelID = panelID;
    this.commonInfo = commonInfo;
    this.parentTabbedPane = parentTabbedPane;
    this.parentDataPanel = parentDataPanel;

    setLayout(new BorderLayout());

    font = new Font("Serif", Font.PLAIN, 11);
    font2 = new Font("Serif", Font.BOLD, 11);
    fgColor = Color.blue;
    fgColor2 = new Color(255, 99, 71);
    bgColor = new Color(127, 255, 212);
    JLabel label;
    JButton button;
    int index = 0;

    textArea = new JTextArea(4, 60);
    textArea.setFont(new Font("Serif", Font.PLAIN, 11));
    textArea.setForeground(Color.blue);
    textArea.setLineWrap(true);   
    textArea.setEditable(true);  
    TitledBorder border1  = new TitledBorder(null, 
					     " 回答テキストを入力して下さい。 ", 
					     TitledBorder.RIGHT, TitledBorder.TOP,
					     new Font("Serif", Font.PLAIN, 11) );
    EmptyBorder border2 = new EmptyBorder(2, 2, 2, 2);
    textArea.setBorder(new CompoundBorder(border1, border2)); 
    
    GridBagLayout gbl = new GridBagLayout();
    GridBagConstraints gbc = new GridBagConstraints();
    gbc.anchor = GridBagConstraints.WEST;
    
    JPanel panel = new JPanel();    
    panel.setLayout(gbl);    
   
    parentDataPanel.setTitleText(commonInfo.syllabusView2.titleText);
    
    label = new JLabel(" " + enqueteItem[0][0][0]);
    label.setFont(font2);
    label.setForeground(fgColor2);
    label.setBorder(new EmptyBorder(10, 2, 6, 2));
    label.setHorizontalAlignment(SwingConstants.LEFT);

    gbc.gridx = 0;
    gbc.gridy = index;
    index++;
    gbc.gridwidth = 1;
    gbc.gridheight = 1;
    gbl.setConstraints(label, gbc);
    panel.add(label);

    for (int i = 1; i < num1; i++) {
      String enqueteText = enqueteItem[0][i][0];
      String buttonTitle1 = enqueteItem[0][i][1];

      label = new JLabel(" ・ " + enqueteText);
      label.setFont(font);
      label.setForeground(fgColor);
      label.setBorder(new EmptyBorder(4, 2, 4, 2));   
      label.setHorizontalAlignment(SwingConstants.LEFT);
      
      gbc.gridy = index;
      index++;
      gbl.setConstraints(label, gbc);
      panel.add(label);
      
      JPanel panel2 = new JPanel();
      panel2.setLayout(new BoxLayout(panel2, BoxLayout.X_AXIS));

      panel2.add(Box.createRigidArea(new Dimension(16, 16)));

      button = new JButton(buttonTitle1);
      button.setFont(font);
      button.setBackground(bgColor);
      button.addActionListener(listener);
      button.setBorder(new CompoundBorder(BorderFactory.createRaisedBevelBorder(), 
					  new EmptyBorder(2,4,2,4)));
      subjectItemsButton[i] = button;
      panel2.add(button);

      gbc.gridy = index;
      index++;
      gbl.setConstraints(panel2, gbc);
      panel.add(panel2);

      Component box2 = Box.createRigidArea(new Dimension(3, 3));
      gbc.gridy = index;
      index++;
      gbl.setConstraints(box2, gbc);
      panel.add(box2);        
    }     
   
    Component box = Box.createRigidArea(new Dimension(4, 4));
    gbc.gridy = index;
    index++;
    gbl.setConstraints(box, gbc);
    panel.add(box);
        
    label = new JLabel(" " + enqueteItem[1][0][0]);
    label.setFont(font2);
    label.setForeground(fgColor2);
    label.setBorder(new EmptyBorder(10, 2, 6, 2));
    label.setHorizontalAlignment(SwingConstants.LEFT);

    gbc.gridy = index;
    index++;
    gbl.setConstraints(label, gbc);
    panel.add(label);

    for (int i = 1; i < num2; i++) {
      String enqueteText = enqueteItem[1][i][0];
      String buttonTitle1 = enqueteItem[1][i][1];

      label = new JLabel(" ・ " + enqueteText);
      label.setFont(font);
      label.setForeground(fgColor);
      label.setBorder(new EmptyBorder(4, 2, 4, 2));  
      label.setHorizontalAlignment(SwingConstants.LEFT);  
      
      gbc.gridy = index;
      index++;
      gbl.setConstraints(label, gbc);
      panel.add(label);
      
      JPanel panel2 = new JPanel();
      panel2.setLayout(new BoxLayout(panel2, BoxLayout.X_AXIS));

      panel2.add(Box.createRigidArea(new Dimension(10, 10)));
  
      button = new JButton(buttonTitle1);
      button.setFont(font);
      button.setBackground(bgColor);
      button.addActionListener(listener);
      button.setBorder(new CompoundBorder(BorderFactory.createRaisedBevelBorder(), 
					  new EmptyBorder(2,4,2,4)));
      objectivesButton[i] = button;
      panel2.add(button);

      gbc.gridy = index;
      index++;
      gbl.setConstraints(panel2, gbc);
      panel.add(panel2);

      Component box3 = Box.createRigidArea(new Dimension(3, 3));
      gbc.gridy = index;
      index++;
      gbl.setConstraints(box3, gbc);
      panel.add(box3);      
    }
  
    box = Box.createRigidArea(new Dimension(4, 4));
    gbc.gridy = index;
    index++;
    gbl.setConstraints(box, gbc);
    panel.add(box);
        
    label = new JLabel(" " + enqueteItem[2][0][0]);
    label.setFont(font2);
    label.setForeground(fgColor2);
    label.setBorder(new EmptyBorder(10, 2, 6, 2));
    label.setHorizontalAlignment(SwingConstants.LEFT);

    gbc.gridy = index;
    index++;
    gbl.setConstraints(label, gbc);
    panel.add(label);

    for (int i = 1; i < num3; i++) {
      String enqueteText = enqueteItem[2][i][0];
      String buttonTitle1 = enqueteItem[2][i][1];

      label = new JLabel(" ・ " + enqueteText);
      label.setFont(font);
      label.setForeground(fgColor);
      label.setBorder(new EmptyBorder(4, 2, 4, 2));  
      label.setHorizontalAlignment(SwingConstants.LEFT);  
      
      gbc.gridy = index;
      index++;
      gbl.setConstraints(label, gbc);
      panel.add(label);
      
      JPanel panel2 = new JPanel();
      panel2.setLayout(new BoxLayout(panel2, BoxLayout.X_AXIS));

      panel2.add(Box.createRigidArea(new Dimension(10, 10)));
  
      button = new JButton(buttonTitle1);
      button.setFont(font);
      button.setBackground(bgColor);
      button.addActionListener(listener);
      button.setBorder(new CompoundBorder(BorderFactory.createRaisedBevelBorder(), 
					  new EmptyBorder(2,4,2,4)));
      choiceButton[i] = button;
      panel2.add(button);
      
      gbc.gridy = index;
      index++;
      gbl.setConstraints(panel2, gbc);
      panel.add(panel2);

      Component box3 = Box.createRigidArea(new Dimension(3, 3));
      gbc.gridy = index;
      index++;
      gbl.setConstraints(box3, gbc);
      panel.add(box3);      
    }

    box = Box.createRigidArea(new Dimension(6, 6));
    gbc.gridy = index;
    index++;
    gbl.setConstraints(box, gbc);
    panel.add(box);
        
    JScrollPane jsp = new JScrollPane(panel);
    add("Center", jsp);
  }
 
  public void pageOpened() { 
    makeListOfSubjectItems();
    makeListOfObjectives();
    makeListOfChoices();
  }

  public void makeListOfSubjectItems() {
    listOfSubjectItems.clear();
    String[] lines = commonInfo.syllabusView2.getListOfSubjectItems();
    for (String line : lines) {	
      if (line.length() > 60) {
	listOfSubjectItems.add(line.substring(0, 60));
      } else {
	listOfSubjectItems.add(line);
      }
    }
  }

  public void makeListOfObjectives() {
    listOfObjectives.clear();
    String[] lines = commonInfo.syllabusView2.getListOfObjectives();
    for (String line : lines) {
      if (line.length() > 60) {
	listOfObjectives.add(line.substring(0, 60));
      } else {
	listOfObjectives.add(line);
      }
    }
  }

  public void makeListOfChoices() {
    listOfChoices.clear();
    for (String line : choice) {
      if (line.length() > 60) {
	listOfChoices.add(line.substring(0, 60));
      } else {
	listOfChoices.add(line);
      }
    }
  }

  public class ButtonListener implements ActionListener {

    public void actionPerformed(ActionEvent e) {
      Object o = e.getSource();
      for (int i = 1; i < num1; i++) {
	if (o == subjectItemsButton[i]) {
	  showSubjectItemsDialog(i);
	}
      }
      for (int i = 1; i < num2; i++) {
	if (o == objectivesButton[i]) {
	  showObjectivesDialog(i);
	}
      }  
      for (int i = 1; i <= 2; i++) {
	if (o == choiceButton[i]) {
	  showChoiceDialog(i);
	}
      }
      if (o == choiceButton[3]) {
	showTextInputDialog(3);
      }
    }
  }
    
  public boolean showTextInputDialog(int i) {
    String enqueteText = enqueteItem[2][i][0];

    ArrayList<Object> list = new ArrayList<Object>();
    list.add(enqueteText);
    list.add(textArea);
    
    int ans = JOptionPane.showOptionDialog(commonInfo.getFrame(), 
					   list.toArray(), 
					   "テキスト入力画面", 
					   JOptionPane.OK_CANCEL_OPTION,
					   JOptionPane.QUESTION_MESSAGE,
					   null,
					   null,
					   null);
    return (ans == JOptionPane.OK_OPTION);
  }
    
  public boolean showSubjectItemsDialog(int i) {
    String enqueteText = enqueteItem[0][i][0];

    ArrayList<Object> list = new ArrayList<Object>();
    list.add(enqueteText);
    list.add(" ");
    for (String text : listOfSubjectItems) {
      JCheckBox checkBox = new JCheckBox(" " + text);
      checkBox.setFont(font);
      checkBox.setForeground(fgColor);
      list.add(checkBox);
    }
    
    int ans = JOptionPane.showOptionDialog(commonInfo.getFrame(), 
					   list.toArray(), 
					   "複数選択の選択肢", 
					   JOptionPane.OK_CANCEL_OPTION,
					   JOptionPane.QUESTION_MESSAGE,
					   null,
					   null,
					   null);
    return (ans == JOptionPane.OK_OPTION);
  }
    
  public boolean showObjectivesDialog(int i) {
    String enqueteText = enqueteItem[1][i][0];

    ArrayList<Object> list = new ArrayList<Object>();
    list.add(enqueteText);
    list.add(" ");
    for (String text : listOfObjectives) {
      JCheckBox checkBox = new JCheckBox(" " + text);
      checkBox.setFont(font);
      checkBox.setForeground(fgColor);
      list.add(checkBox);
    }
    
    int ans = JOptionPane.showOptionDialog(commonInfo.getFrame(), 
					   list.toArray(), 
					   "複数選択の選択肢", 
					   JOptionPane.OK_CANCEL_OPTION,
					   JOptionPane.QUESTION_MESSAGE,
					   null,
					   null,
					   null);
    return (ans == JOptionPane.OK_OPTION);
  }

  public boolean showChoiceDialog(int i) {
    String enqueteText = enqueteItem[2][i][0];

    ArrayList<Object> list = new ArrayList<Object>();
    list.add(enqueteText);
    list.add(" ");
    
    ButtonGroup buttonGroup = new ButtonGroup();
    for (String text : listOfChoices) {
      JRadioButton radioButton = new JRadioButton(" " + text);
      radioButton.setFont(font);
      radioButton.setForeground(fgColor);
      buttonGroup.add(radioButton);
      list.add(radioButton);
    }
    
    int ans = JOptionPane.showOptionDialog(commonInfo.getFrame(), 
					   list.toArray(), 
					   "単一選択の選択肢", 
					   JOptionPane.OK_CANCEL_OPTION,
					   JOptionPane.QUESTION_MESSAGE,
					   null,
					   null,
					   null);
    return (ans == JOptionPane.OK_OPTION);
  }

}
