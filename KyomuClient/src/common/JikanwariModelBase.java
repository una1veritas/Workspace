package common;
import java.util.*;
//import java.awt.*;
//import java.awt.event.*;
//import javax.swing.*;
//import javax.swing.table.*;
import clients.*;

public class JikanwariModelBase {
  public CommonInfo commonInfo;
  public String serviceName;
  public String panelID;

  public ArrayList<String> rowTitleList;
  public ArrayList<String> columnTitleList;

  public HashMap<String, Integer> rowKeyMap;
  public HashMap<String, Integer> columnKeyMap;

  public ArrayList<ArrayList<ArrayList<String>>> komaInfo;
 
  public JikanwariModelBase(String serviceName,
			    String panelID,
			    CommonInfo commonInfo,
			    JikanwariViewBase jbase) {
    this.commonInfo = commonInfo;
    this.rowTitleList = jbase.rowTitleList;
    this.columnTitleList = jbase.columnTitleList;
    this.rowKeyMap = jbase.rowKeyMap;
    this.columnKeyMap = jbase.columnKeyMap;
    this.serviceName = serviceName;
    this.panelID = panelID;

    makeEmptyJikanwari();
  }
    
  public void makeJikanwariData(String paramValues) {   
    if (paramValues != null) {       
      String answer = commonInfo.getQueryResult(serviceName, panelID, "8", paramValues);
      if (answer == null) {
	for (int i = 1; i < rowTitleList.size(); i++) {
	  for (int j = 1; j < columnTitleList.size(); j++) {
	    komaInfo.get(i).get(j).clear();
	    komaInfo.get(i).get(j).add("empty");
	  }
	}
	return;
      } else {
	for (int i = 1; i < rowTitleList.size(); i++) {
	  for (int j = 1; j < columnTitleList.size(); j++) {
	    komaInfo.get(i).get(j).clear();
	  }
	}
	String[] lines = answer.split("\\$");
	for (String line : lines) {
	  addToKomaInfo(line);	  
	}
	for (int i = 1; i < rowTitleList.size(); i++) {
	  for (int j = 1; j < columnTitleList.size(); j++) {
	    int cnt = komaInfo.get(i).get(j).size();
	    if (cnt == 0) {
	      komaInfo.get(i).get(j).add("empty"); 
	    }
	  }
	}	
      }
    } 
  }

  private void makeEmptyJikanwari() {
    komaInfo = new ArrayList<ArrayList<ArrayList<String>>>();
    for (int i = 0; i < rowTitleList.size(); i++) {
      ArrayList<ArrayList<String>> rowList = new ArrayList<ArrayList<String>>();
      for (int j = 0; j < columnTitleList.size(); j++) {
	ArrayList<String> komaList = new ArrayList<String>();
	rowList.add(komaList);
      }
      komaInfo.add(rowList);
    }
    for (int j = 0; j < columnTitleList.size(); j++) {
      komaInfo.get(0).get(j).add(columnTitleList.get(j));
    }
    for (int i = 0; i < rowTitleList.size(); i++) {
      komaInfo.get(i).get(0).add(rowTitleList.get(i));
    }
  }

  private void addToKomaInfo(String line) {
    String[] tokens = line.split("\\|");
    int count = tokens.length;
    if (count != 19) {
      commonInfo.showMessageLong("JikanwariModelBase: $ format error $ " + line);  
      return;
    } else {
      String rowKeyVal = tokens[0];
      String columnKeyVal = tokens[1];
      String substr = line.substring(line.indexOf("|") + 1);
      String subjectInfo = substr.substring(substr.indexOf("|") + 1);
      int row = rowKeyMap.get(rowKeyVal);
      int col = columnKeyMap.get(columnKeyVal);
      komaInfo.get(row).get(col).add(subjectInfo);
    }
  } 

  public ArrayList<String> getKomaInfo(int row, int col) {
    return komaInfo.get(row).get(col);
  }
}
