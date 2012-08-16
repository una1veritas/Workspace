package common;
import clients.*;
import java.util.*;

public class RegistrTableModel extends TableModelBase { 

  public RegistrTableModel(CommonInfo commonInfo,
			   String serviceName, String panelID,
			   ArrayList<String> columnTitleList,
			   ArrayList<String> columnCodeList,
			   ArrayList<String> columnDisplayList) {
    super(commonInfo, serviceName, panelID,
	  columnTitleList, columnCodeList,
	  columnDisplayList);
  }

  public void setTableData(String serviceName, String panelID, 
			   String switchCode, String queryParamValues) {
    super.setTableData(serviceName, panelID, switchCode, queryParamValues); 

    CellObject cobj;
    for (int i = 0; i < getRowCount(); i++) {
      String SUBJECT_CODE = getCellValueAt(i, "授業科目名");
      String CLASS_CODE = getCellValueAt(i, "クラス");
      int sig = commonInfo.commonInfoMethods.registrInfo.getSubjectStatus(SUBJECT_CODE, CLASS_CODE);
      switch (sig) {
      case 1:
	cobj = new CellObject("1", "修得済み科目"); break;
      case 2:
	cobj = new CellObject("2", "認定済み科目"); break;
      case 3:
	cobj = new CellObject("3", "履修登録科目"); break;
      case 4:
	cobj = new CellObject("4", "仮履修登録"); break;
      case 5:
	cobj = new CellObject("5", "仮履修登録"); break;
      default:
	cobj = new CellObject("0", " "); break;	
      }
      setValueAt(cobj, i, "備考");
    }
    fireTableChanged(null);     
  }
}
