package common;
import clients.*;
import java.util.*;
import java.awt.*;
//import java.awt.event.*;
import javax.swing.*;
//import javax.swing.event.*;

public class SplitPaneBase extends JSplitPane {
  private String serviceName;
  private String nodePath;  
  private CommonInfo commonInfo;
  private TabbedPaneBase parentTabbedPane;    
  private String methodInvokedWhenSelected;

  private String[] splitItemKey = { "CHILD_NODE_ID", 
				    "CHILD_COMPONENT_TYPE", 
				    "CHILD_PANEL_ID",
				    "CHILD_METHOD_WHEN_SELECTED", 
				    "CHILD_QUALIFICATION_METHOD", 
				    "CHILD_QUALIFICATION_PARAM" };

  private ArrayList<Map<String, String>> splitInfoList = new ArrayList<Map<String, String>>();

  public SplitPaneBase(String serviceName, String nodePath,
		       CommonInfo commonInfo, 
		       TabbedPaneBase parentTabbedPane, 
		       String methodInvokedWhenSelected) {
    this.serviceName = serviceName;
    this.nodePath = nodePath;
    this.commonInfo = commonInfo;
    this.parentTabbedPane = parentTabbedPane;
    this.methodInvokedWhenSelected = methodInvokedWhenSelected;
    makeSplitInfoList();
    setFont(new Font("DialogInput", Font.PLAIN, 12));
    if (methodInvokedWhenSelected != null) {
      invokeMethodWhenSelected(methodInvokedWhenSelected);
    }
  }

  private void makeSplitInfoList() {
    String splitPanesInfo = commonInfo.getSplitPaneStruct(serviceName, nodePath);
    if (splitPanesInfo != null) {      
      String[] splitPanesArray = splitPanesInfo.split("\\$");
      for (String splitPaneInfo : splitPanesArray) {
	String[] splitItemVal =  splitPaneInfo.split("\\|");
	if (splitItemVal.length == 6) {  
	  HashMap<String, String> map = new HashMap<String, String>();
	  for (int i = 0; i < 6; i++) {
	    String key = splitItemKey[i];
	    String val = splitItemVal[i].trim();
	    if (val.equals("")) val = null;
	    map.put(key, val);
	  }
	  splitInfoList.add(map);
	} else {
	  commonInfo.showMessageLong("SplitPaneStruct: $ format error $ " + splitPaneInfo);
	} 
      }
    } else {
      commonInfo.showMessageLong("SplitPaneStruct not found: $ serviceName = " + serviceName + " $ nodePath = " + nodePath); 
    } 
  }
  
  public void pageOpened() {
    String childServiceName;
    String childNodePath;

    setDividerLocation(500);

    try {
    for (int i = 0; i < 2; i++) {
      Map<String, String> splitInfoMap = splitInfoList.get(i);
      String childNodeID = splitInfoMap.get("CHILD_NODE_ID");
      String childComponentType = splitInfoMap.get("CHILD_COMPONENT_TYPE");     
      String childPanelID = splitInfoMap.get("CHILD_PANEL_ID"); 
      String childMethodWhenSelected = splitInfoMap.get("CHILD_METHOD_WHEN_SELECTED"); 
      String childQualificationMethod = splitInfoMap.get("CHILD_QUALIFICATION_METHOD");
      String childQualificationParam = splitInfoMap.get("CHILD_QUALIFICATION_PARAM");
      boolean qualified = commonInfo.commonInfoMethods.checkQualification(childQualificationMethod, 
									  childQualificationParam,
									  parentTabbedPane);

    
      if (!(childNodeID.startsWith("/"))) {
	childServiceName = serviceName;
	childNodePath = nodePath + "." + childNodeID;
      } else {
	int pos = childNodeID.indexOf(".");
	childServiceName = childNodeID.substring(1, pos);
	childNodePath = childNodeID.substring(pos+1); 
      } 

      if (childComponentType.equals("TabbedPane")) {
	Component oldComponent;
	if (i == 0) {
	  oldComponent = getLeftComponent(); 
	} else {
	  oldComponent = getRightComponent(); 
	}
	if ((oldComponent == null) || (!(oldComponent instanceof JTabbedPane))) { 
	  TabbedPaneBase pane = new TabbedPaneBase(childServiceName, childNodePath,
						   commonInfo, parentTabbedPane,
						   childMethodWhenSelected);
	  if (i == 0) {
	    setLeftComponent(pane); 
	  } else {
	    setRightComponent(pane); 
	  }
	  pane.pageOpened();	
	} else {
	  TabbedPaneBase pane = (TabbedPaneBase) oldComponent;
	  if (childMethodWhenSelected != null) {
	    pane.invokeMethodWhenSelected(childMethodWhenSelected);
	  }
	  pane.pageOpened();
	}
      } else if (childComponentType.equals("DataPanel")) {
	Component oldComponent;
	if (i == 0) {
	  oldComponent = getLeftComponent(); 
	} else {
	  oldComponent = getRightComponent(); 
	}
	if ((oldComponent == null) || (!(oldComponent instanceof JPanel))) {
	  DataPanelBase panel = new DataPanelBase(childServiceName, childNodePath,
						  childPanelID, commonInfo, 
						  parentTabbedPane,
						  childMethodWhenSelected);
	  if (i == 0) {
	    setLeftComponent(panel); 
	  } else {
	    setRightComponent(panel); 
	  }
	  panel.pageOpened();	    
	} else {
	  DataPanelBase panel = (DataPanelBase) oldComponent;
	  if (childMethodWhenSelected != null) {
	    panel.invokeMethodWhenSelected(childMethodWhenSelected);
	  }
	  panel.pageOpened();
	}
      } 
    }
    } catch (Exception ex) {
      ex.printStackTrace();
    }

    commonInfo.timerRestart();
  } 

  public void invokeMethodWhenSelected(String methodName) {
    try {
      getClass().getMethod(methodName, (java.lang.Class[]) null).invoke(this, (java.lang.Object[]) null);
    } catch (Exception ex) {
      commonInfo.showMessageLong("SplitPaneBase: $ MethodInvokedWhenSelected is not invoked $ " + methodName);  
    }
  }
}
