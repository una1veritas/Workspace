package common;

import clients.*;
import java.util.*;

public class StructTableModel extends TableModelBase { 
  public StructTableModel(CommonInfo commonInfo,
			  String serviceName, String panelID,
			  ArrayList<String> columnTitleList,
			  ArrayList<String> columnCodeList,
			  ArrayList<String> columnDisplayList) {
    super(commonInfo, serviceName, panelID,
	  columnTitleList, columnCodeList,
	  columnDisplayList);
  }

  //**** setTableData  ***//
  
  public void setTableData(String serviceName, String panelID, 
			   String switchCode, String queryParamValues) { 
    listOfRows.clear(); 

    if (queryParamValues != null) {
      String answer = commonInfo.getSingleStructQueryResult(serviceName, 
							    panelID, 
							    switchCode, 
							    queryParamValues); 
      if (answer != null) {
	String[] lines = answer.split("\\$");
	for (String line : lines) {
	  String[] tokens = line.split("\\|");
	  LinkedList<String> linkedList = new LinkedList<String>();	    
	  for (String token : tokens) {
	    linkedList.add(token);
	  }	
	  ArrayList<CellObject> rowList = new ArrayList<CellObject>();
	  for (int i = 0; i < getColumnCount(); i++) {
	    String columnDisplay = columnDisplayList.get(i);
	    CellObject cobj = makeCellObject(columnDisplay, linkedList);
	    rowList.add(cobj);
	  }
	  listOfRows.add(rowList);
	}
      }
    }
    fireTableChanged(null); 
  }
}
