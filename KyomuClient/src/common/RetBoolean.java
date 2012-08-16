package common;

public class RetBoolean {
  private boolean ret = false;
  
  public void setValue( boolean b ) {
    ret = b;
  }
  
  public void accept() {
    ret = true;
  }
  
  public void reject() {
    ret = false;
  }
  
  public boolean getValue() {
    return ret;
  }
}
