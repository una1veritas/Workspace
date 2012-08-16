package common;

import clients.*;
//import java.util.*;
//import java.awt.*;
//import javax.swing.*;
//import javax.swing.table.*;

public class StructTableView extends TableViewBase {

  public StructTableView(String tableViewType,
			 String serviceName,
			 String nodePath,
			 String panelID,
			 CommonInfo commonInfo, 
			 TabbedPaneBase tabbedPane,
			 DataPanelBase dataPanel) {
    super(tableViewType, serviceName, nodePath, panelID, 
	  commonInfo, tabbedPane, dataPanel);
  }

  public void setTableModel() {
    tableModel = new StructTableModel(commonInfo, serviceName, panelID, 
				      columnTitleList, columnCodeList, 
				      columnDisplayList);
  }
}
