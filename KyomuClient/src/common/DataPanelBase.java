package common;

import clients.*;
import java.util.*;
import java.awt.*;
//import java.io.*;
//import java.awt.event.*;
import javax.swing.*;
//import javax.swing.text.*;
import javax.swing.border.*;
//import java.lang.reflect.*; 

public class DataPanelBase extends JPanel {
  private String serviceName;
  private String nodePath;
  private String panelID;
  private CommonInfo commonInfo;
  private TabbedPaneBase parentTabbedPane;

  private String headTitle;
  private Color  headFgColor;
  private Color  headBgColor;
  private String dataViewType;    
  private String bottomTitle;
  private Color  bottomFgColor;
  private Color  bottomBgColor;

  private JTextField titleField;
  private ButtonFunctions buttonFunctions;
  private DataPanelMethods dataPanelMethods;
  
  //** When dataViewType is SimpleTableView or VarTableView //
  //** or other classes extending TableViewBase  **//
  public TableViewBase tableView = null;
  public TablePrinter tablePrinter = null;

  //** When dataViewType is ReportTableView **//
  public ReportTableView reportTableView = null;
  
  //** When dataViewType is NinteiTableView **//
  public NinteiTableView ninteiTableView = null;
  
  //** When dataViewType is RegistrTableView **//
  public RegistrTableView registrTableView = null;
  
  //** When dataViewType is StructTableView **//
  public StructTableView structTableView = null;

  //** When dataViewType is GakusekiView **//
  public GakusekiView gakusekiView = null;

  //** When dataViewType is HtmlView **//
  public HtmlViewBase htmlView = null;

  //** When dataViewType is JikanwariView **//
  public JikanwariViewBase jikanwariView = null;

  //** When dataViewType is PhotoView **//
  public PhotoViewBase photoView = null;

  //** When dataViewType is MailView **//
//  public MailViewBase mailView = null;

  //** When dataViewType is SyllabusView **//
  public SyllabusView syllabusView = null;

  //** When dataViewType is SyllabusView2 **//
  public SyllabusView2 syllabusView2 = null;

  //** When dataViewType is SyllabusEdit **//
  public SyllabusEdit syllabusEdit = null;

  //** When dataViewType is EnqueteSheet **//
  public EnqueteSheet enqueteSheet = null;

  public TableViewBase getTableView() {
    return tableView;
  }


  public DataPanelBase(String serviceName,
		       String nodePath,
		       String panelID,
		       CommonInfo commonInfo, 
		       TabbedPaneBase parentTabbedPane, 
		       String methodWhenSelected) {
    this.serviceName = serviceName;
    this.nodePath = nodePath;
    this.panelID = panelID;
    this.commonInfo = commonInfo;
    this.parentTabbedPane = parentTabbedPane;
    dataPanelMethods = new DataPanelMethods(serviceName, nodePath, 
					    commonInfo, parentTabbedPane,
					    this);
    makeDataPanelInfo();
    if (methodWhenSelected != null) {
      invokeMethodWhenSelected(methodWhenSelected);
    }
    buttonFunctions = new ButtonFunctions(serviceName, nodePath, panelID, 
					  commonInfo, parentTabbedPane, dataPanelMethods);
    layoutComponentsOnThisPanel();
    if (tableView != null) {
      buttonFunctions.setTargetTableView(tableView);
    }
  }

  private void makeDataPanelInfo() {
    String dataPanelInfo = commonInfo.getDataPanelStruct(serviceName, panelID);
    if (dataPanelInfo != null) {
      String[] lines = dataPanelInfo.split("\\$");  
      String[] tokens = lines[0].split("\\|");
      if (tokens.length == 7) {
	for (int i = 0; i < 7; i++) {
	  String token = tokens[i].trim();
	  if (token.equals("")) token = null;
	  switch (i) {
	  case 0:
	    headTitle = token; break;
	  case 1:
	    headFgColor = commonInfo.getHeadTitleFgColor(token); break;
	  case 2:
	    headBgColor = commonInfo.getHeadTitleBgColor(token); break;
	  case 3:
	    dataViewType = token; break;
	  case 4:
	    bottomTitle = token; break;
	  case 5:
	    bottomFgColor = commonInfo.getBottomTitleFgColor(token); break;
	  case 6:
	    bottomBgColor = commonInfo.getBottomTitleBgColor(token); break;
	  }
	}
      } else {
	commonInfo.showMessageLong("DataPaneBase: $ format error $ " + dataPanelInfo);  
      } 
    } else {
      commonInfo.showMessageLong("DataPanelBase: $ DataPanelInfo is not found $ serviceName = " + serviceName + " $ panelID = " + panelID);  
    }
  }

  private void layoutComponentsOnThisPanel() {
    setLayout(new BorderLayout());
    setHeadPanel();
    setMainComponent();
    setBottomPanel();
  }

  private void setHeadPanel() {
    JPanel headPanel = new JPanel();
    headPanel.setLayout(new BorderLayout());
    
    JPanel panel0 = new JPanel();
    panel0.setLayout(new BoxLayout(panel0, BoxLayout.X_AXIS));
    titleField = new JTextField();
    titleField.setBorder(new EmptyBorder(2, 2, 2, 2));
    titleField.setForeground(headFgColor);
    titleField.setBackground(headBgColor);
    panel0.add("North", titleField);
    ArrayList<JButton> buttonList0 = buttonFunctions.getButtonList(0);
    for (JButton button : buttonList0) {
      panel0.add(Box.createRigidArea(new Dimension(5,5)));
      panel0.add(button);
    }
    panel0.add(Box.createRigidArea(new Dimension(5,5)));
    headPanel.add("North", panel0);
	
    ArrayList<JButton> buttonList1 = buttonFunctions.getButtonList(1);
    if (buttonList1.size() != 0) {
      JPanel panel1 = new JPanel();
      panel1.setLayout(new BoxLayout(panel1, BoxLayout.X_AXIS));
      for (JButton button : buttonList1) {
	panel1.add(Box.createRigidArea(new Dimension(5,5)));
	panel1.add(button);
      }
      panel1.add(Box.createRigidArea(new Dimension(5,5)));
      headPanel.add("Center", panel1);
    }
    	
    ArrayList<JButton> buttonList2 = buttonFunctions.getButtonList(2);
    if (buttonList2.size() != 0) {
      JPanel panel2 = new JPanel();
      panel2.setLayout(new BoxLayout(panel2, BoxLayout.X_AXIS));
      for (JButton button : buttonList2) {
	panel2.add(Box.createRigidArea(new Dimension(5,5)));
	panel2.add(button);
      }
      panel2.add(Box.createRigidArea(new Dimension(5,5)));
      headPanel.add("South", panel2);
    }
    add("North", headPanel);
  }

  private void setMainComponent() {
    if ((dataViewType.equals("SimpleTableView")) || 
	(dataViewType.equals("VarTableView"))) {
      tableView = new TableViewBase(dataViewType,
				    serviceName, nodePath, panelID, 
				    commonInfo, 
				    parentTabbedPane, this);
      dataPanelMethods.tableView = tableView;
      add("Center", new JScrollPane(tableView));
    } else if (dataViewType.equals("CalendarTableView")) {
      tableView = new CalendarTableView(dataViewType,
					serviceName, nodePath, panelID, 
					commonInfo, 
					parentTabbedPane, this);
      dataPanelMethods.tableView = tableView;
      add("Center", new JScrollPane(tableView));
    } else if (dataViewType.equals("HtmlView")) {
      htmlView = new HtmlViewBase(serviceName, nodePath, panelID, 
				  commonInfo, 
				  parentTabbedPane, this);
      dataPanelMethods.htmlView = htmlView;
      add("Center", htmlView);
    } else if (dataViewType.equals("GakusekiView")) {
      gakusekiView = new GakusekiView(serviceName, nodePath, panelID, 
				      commonInfo, 
				      parentTabbedPane, this);
      dataPanelMethods.gakusekiView = gakusekiView;
      add("Center", gakusekiView);
    } else if (dataViewType.equals("JikanwariView")) {
      jikanwariView = new JikanwariViewBase(serviceName, nodePath, panelID, 
					    commonInfo, 
					    parentTabbedPane, this);
      dataPanelMethods.jikanwariView = jikanwariView;
      add("Center", new JScrollPane(jikanwariView));
    } else if (dataViewType.equals("PhotoView")) {
      photoView = new PhotoViewBase(serviceName, nodePath, panelID, 
				    commonInfo, 
				    parentTabbedPane, this);
      dataPanelMethods.photoView = photoView;
      add("Center", photoView);
    } else if (dataViewType.equals("MailView")) {
//      mailView = new MailViewBase(dataViewType, serviceName, nodePath, panelID, 
//				  commonInfo, 
//				  parentTabbedPane, this);
 //     tableView = mailView;
  //    dataPanelMethods.mailView = mailView;
  //    dataPanelMethods.tableView = tableView;
  //    add("Center", new JScrollPane(mailView));
    } else if (dataViewType.equals("SyllabusView")) {
      syllabusView = new SyllabusView(dataViewType, serviceName, nodePath, panelID, 
				      commonInfo, 
				      parentTabbedPane, this);
      dataPanelMethods.syllabusView = syllabusView;
      add("Center", syllabusView);
    } else if (dataViewType.equals("SyllabusView2")) {
      syllabusView2 = new SyllabusView2(dataViewType, serviceName, nodePath, panelID, 
					commonInfo, 
					parentTabbedPane, this);
      dataPanelMethods.syllabusView2 = syllabusView2;
      commonInfo.syllabusView2 = syllabusView2;
      add("Center", syllabusView2);
    } else if (dataViewType.equals("SyllabusEdit")) {
      syllabusEdit = new SyllabusEdit(dataViewType, serviceName, nodePath, panelID, 
				      commonInfo, 
				      parentTabbedPane, this);
      dataPanelMethods.syllabusEdit = syllabusEdit;
      add("Center", new JScrollPane(syllabusEdit));
    } 
	/*
	 else if (dataViewType.equals("EnqueteSheet")) {
      enqueteSheet = new EnqueteSheet(dataViewType, serviceName, nodePath, panelID,
				      commonInfo, 
				      parentTabbedPane, this);
      dataPanelMethods.enqueteSheet = enqueteSheet;
      add("Center", enqueteSheet);
    } */
	else if (dataViewType.equals("ReportTableView")) {
      reportTableView = new ReportTableView(dataViewType, serviceName, nodePath, panelID, 
					    commonInfo, 
					    parentTabbedPane, this);
      tableView = reportTableView;
      dataPanelMethods.reportTableView = reportTableView;
      dataPanelMethods.tableView = reportTableView;
      add("Center", new JScrollPane(reportTableView));
    } else if (dataViewType.equals("NinteiTableView")) {
      ninteiTableView = new NinteiTableView(dataViewType, serviceName, nodePath, panelID, 
					    commonInfo, 
					    parentTabbedPane, this);
      tableView = ninteiTableView;
      dataPanelMethods.ninteiTableView = ninteiTableView;
      dataPanelMethods.tableView = ninteiTableView;
      add("Center", new JScrollPane(ninteiTableView));
    } else if (dataViewType.equals("RegistrTableView")) {
      registrTableView = new RegistrTableView(dataViewType, serviceName, nodePath, panelID, 
					      commonInfo, 
					      parentTabbedPane, this);
      tableView = registrTableView;
      dataPanelMethods.registrTableView = registrTableView;
      dataPanelMethods.tableView = registrTableView;
      add("Center", new JScrollPane(registrTableView));
    } else if (dataViewType.equals("StructTableView")) {
      structTableView = new StructTableView(dataViewType, serviceName, nodePath, panelID, 
					    commonInfo, 
					    parentTabbedPane, this);
      tableView = structTableView;
      dataPanelMethods.structTableView = structTableView;
      dataPanelMethods.tableView = structTableView;
      add("Center", new JScrollPane(structTableView));
    } else {
      commonInfo.showMessageLong("DataPanelBase: Unknown DataViewType: $ " + dataViewType);
    }
  }

  private void setBottomPanel() {
    if (bottomTitle != null) {
      JPanel bottomPanel = new JPanel();
      bottomPanel.setLayout(new BoxLayout(bottomPanel, BoxLayout.Y_AXIS));
      bottomPanel.add(Box.createRigidArea(new Dimension(2,2)));
      String[] textLines = bottomTitle.split("\\#");
      for (String text : textLines) {
	JTextField textField = new JTextField(text);
	textField.setBorder(new EmptyBorder(2, 4, 2, 4));
	textField.setForeground(bottomFgColor);
	textField.setBackground(bottomBgColor);
	bottomPanel.add(textField);
	bottomPanel.add(Box.createRigidArea(new Dimension(1,1)));
      }
      add("South", bottomPanel);
    }
  }

  public void pageOpened() {

    if (dataViewType.equals("ReportTableView")) {
      if (reportTableView != null) {
	reportTableView.pageOpened();
      }
    } else if (dataViewType.equals("NinteiTableView")) {
      if (ninteiTableView != null) {
	ninteiTableView.pageOpened();
      }
    } else if (dataViewType.equals("RegistrTableView")) {
      if (registrTableView != null) {
	registrTableView.pageOpened();
      }
    } else if (dataViewType.equals("StructTableView")) {
      if (structTableView != null) {
	structTableView.pageOpened();
      }
    } else if (dataViewType.equals("JikanwariView")) {
      if (jikanwariView != null) {
	jikanwariView.pageOpened();
      }
    } else if (dataViewType.indexOf("TableView") >= 0) {
      if (tableView != null) {
	tableView.pageOpened();
      }
    } else if (dataViewType.equals("HtmlView")) {
      if (htmlView != null) {
	htmlView.pageOpened();
      }
    } else if (dataViewType.equals("GakusekiView")) {
      if (gakusekiView != null) {
	gakusekiView.pageOpened();
      }
    } else if (dataViewType.equals("PhotoView")) {
      if (photoView != null) {
	photoView.pageOpened();
      }
    } else if (dataViewType.equals("JikanwariView")) {
      if (jikanwariView != null) {
	jikanwariView.pageOpened();
      }
    } else if (dataViewType.equals("MailView")) {
//      if (mailView != null) {
//	mailView.pageOpened();
//      }
    } else if (dataViewType.equals("SyllabusView")) {
      if (syllabusView != null) {
	syllabusView.pageOpened();
      }
    } else if (dataViewType.equals("SyllabusView2")) {
      if (syllabusView2 != null) {
	syllabusView2.pageOpened();
      }
    } else if (dataViewType.equals("SyllabusEdit")) {
      if (syllabusEdit != null) {
	syllabusEdit.pageOpened();
      }
    } else if (dataViewType.equals("EnqueteSheet")) {
      if (enqueteSheet != null) {
	enqueteSheet.pageOpened();
      }
    } else {
      commonInfo.showMessageLong("DataPanelBase: Unknown component: $ " + dataViewType);
    }    
  }

  public void invokeMethodWhenSelected(String methodName) {
    try {
      getClass().getMethod(methodName, (java.lang.Class[]) null).invoke(this, (java.lang.Object[]) null);
    } catch (Exception ex) {
      commonInfo.showMessageLong("DataPanelBase: $ invokeMethodWhenSelected $ " + methodName);  
    }
  }

  public void setYoken03() {
    parentTabbedPane.addColumnCodeMap("YOKEN_TYPE", "03", "３年進級要件"); 
  }

  public void setYoken04() {
    parentTabbedPane.addColumnCodeMap("YOKEN_TYPE", "04", "４年進級要件"); 
  }

  public void setYoken05() {
    parentTabbedPane.addColumnCodeMap("YOKEN_TYPE", "05", "卒業要件"); 
  }

  public void setYokenMG() {
    parentTabbedPane.addColumnCodeMap("YOKEN_TYPE", "MG", "博士前期修了要件"); 
  }

  public void setYokenDG() {
    parentTabbedPane.addColumnCodeMap("YOKEN_TYPE", "DG", "博士後期修了要件"); 
  }

  public void setYokenEI() {
    parentTabbedPane.addColumnCodeMap("YOKEN_TYPE", "EI", "教職(情報)要件"); 
  }

  public void setYokenEM() {
    parentTabbedPane.addColumnCodeMap("YOKEN_TYPE", "EM", "教職(数学)要件");
  }

  public void setYokenGI() {
    parentTabbedPane.addColumnCodeMap("YOKEN_TYPE", "GI", "大学院教職(情報)要件"); 
  }

  public void setYokenGM() {
    parentTabbedPane.addColumnCodeMap("YOKEN_TYPE", "GM", "大学院教職(数学)要件");
  }

  public void setTitleText(String queryParamDisplays) {
    String text;
    if (headTitle != null) {
      text = " " + headTitle + ": " + queryParamDisplays;
    } else {
      text = "  " + queryParamDisplays;
    }
    titleField.setText(text);
  }

  public String getTitleText() {
    return titleField.getText();
  }
}
