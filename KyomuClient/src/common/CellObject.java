package common;
//import java.util.*;

public class CellObject {
  // JTable ��ɽ����ˤ� CellObject ���Υ��֥������Ȥ��֤�
  // code ��ɽ�����
  // 
  private Object   code;
  private String   display;
  private boolean  changed;
  private boolean  highlighted;

  public CellObject(Object code, String display) {
    this.code = code;
    this.display = display;
    changed = false;
    highlighted = false;
  }

  public CellObject(Object code) {
    this.code = code;
    if (code == null) {
      display = " ";
    } else {
      display = code.toString();
    }
    changed = false;
    highlighted = false;
  }

  public CellObject(Integer o, int m) {
    code = o;
    String str = o.toString();
    int k = m - str.length();
    StringBuffer sb = new StringBuffer();
    while (k > 0) {
      sb.append(" ");
      k--;
    }
    sb.append(str);    
    display = sb.toString();
    changed = false;
    highlighted = false;
  }

  public CellObject(Double o, int m) {
    code = o;
    String str = o.toString();
    int p = str.indexOf(".");
    int k = m - p;
    StringBuffer sb = new StringBuffer();
    while (k > 0) {
      sb.append(" ");
      k--;
    }
    sb.append(str);    
    display = sb.toString();
    changed = false;
    highlighted = false;
  }

  public CellObject(Double o, int m1, int m2) {
    code = o;
    double d = o.doubleValue() + 0.0000000000001;
    int x1 = (int) d;
    double d2 = d - x1;
    for (int i = 0; i < m2; i++) {
      d2 = d2 * 10;
    }
    int x2 = (int) d2;   
    //    System.out.println("" + d + " " + m1 + " " + m2 + " " + d2 + " " + x1 + " " + x2); // 
    String str1 = "" + x1;
    int k = m1 - str1.length();
    StringBuffer sb = new StringBuffer();
    while (k > 0) {
      sb.append(" ");
      k--;
    }
    sb.append(str1);    
    sb.append(".");    
    String str2 = "" + x2;
    k = m2 - str2.length();
    while (k > 0) {
      sb.append("0");
      k--;
    }
    sb.append(str2);    
    display = sb.toString();
    changed = false;
    highlighted = false;
  }

  public CellObject(CellObject co) {
    code = co.getCode();
    display = co.getDisplay();
    changed = false;
    highlighted = false;
  }

  public String toString() {
    return display;
  }

  public String getDisplay() {
    return display;
  }

  public Object toCode() {
    return code;
  }

  public Object getCode() {
    return code;
  }

  public String getCodeValue() {
    if (code == null) {
      return null;
    } else {
      return code.toString();
    }
  }

  public String getValue() {
    if (code == null) {
      return " |" + display.trim();
    } else {
      return code.toString() + "|" + display.trim();
    }    
  }

  public void setData(Object code, String display) {
    this.code = code;
    this.display = display;
    changed = true;
  }

  public void setData(Object code) {
    this.code = code;
    if (code == null) {
      display = " ";
    } else {
      display = code.toString();
    }
    changed = true;
  }

  public void setData(Integer o, int m) {
    code = o;
    String str = o.toString();
    int k = m - str.length();
    StringBuffer sb = new StringBuffer();
    while (k > 0) {
      sb.append(" ");
      k--;
    }
    sb.append(str);    
    display = sb.toString();
    changed = true;
    highlighted = false;
  }

  public void setData(Double o, int m) {
    code = o;
    String str = o.toString();
    int p = str.indexOf(".");
    int k = m - p;
    StringBuffer sb = new StringBuffer();
    while (k > 0) {
      sb.append(" ");
      k--;
    }
    sb.append(str);    
    display = sb.toString();
    changed = true;
    highlighted = false;
  }

  public void setData(Double o, int m1, int m2) {
    code = o;
    double d = o.doubleValue();
    int x1 = (int) d;
    double d2 = d - x1;
    for (int i = 0; i < m2; i++) {
      d2 = d2 * 10;
    }
    int x2 = (int) d2;    
    String str1 = "" + x1;
    int k = m1 - str1.length();
    StringBuffer sb = new StringBuffer();
    while (k > 0) {
      sb.append(" ");
      k--;
    }
    sb.append(str1);    
    sb.append(".");    
    String str2 = "" + x2;
    k = m2 - str2.length();
    while (k > 0) {
      sb.append("0");
      k--;
    }
    sb.append(str2);    
    display = sb.toString();
    changed = true;
    highlighted = false;
  }

  public void setHighlighted() {
    highlighted = true;
  }

  public void resetHighlighted() {
    highlighted = false;
  }

  public void resetChanged() {
    changed = false;
  }

  public boolean isChanged() {
    return changed;
  }

  public boolean isHighlighted() {
    return highlighted;
  }
}

