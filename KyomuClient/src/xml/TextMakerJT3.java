package xml;

public class TextMakerJT3 implements IJTVisitor {

  private StringBuffer buffer_ = new StringBuffer();
  private boolean OLMode = false;
  private int OLIndex = 0;
  private boolean getMode = false;
 
  public String getText() {
    return (new String(buffer_));
  }

  public void enter(JTAttr node) {
    // do nothing
  }
  
  public void leave(JTAttr node) {
    // do nothing
  }
  
  public void enter(JTElement node) {
    String tag = node.getName();

    if (tag.equals("授業の達成目標")) {
      getMode = true;

    } else if (tag.equals("UL")) {

    } else if (tag.equals("OL")) {
      if (getMode) {
	OLMode = true;  
	OLIndex = 1;   
      } 
    } else if (tag.equals("LI")) {
      if (getMode) {
	if (OLMode) { 
	  buffer_.append("("+OLIndex+") " + node.getValue() + "|");  
	  OLIndex++;
	} 
      }
    } else if (tag.equals("P")) {

    } else {
      getMode = false;
    }
  }
  
  public void leave(JTElement node) {

  }
  
  public void enter(JTNode node) {
    throw (new InternalError());
  }
  
  public void leave(JTNode node) {
    throw (new InternalError());
  }
}
