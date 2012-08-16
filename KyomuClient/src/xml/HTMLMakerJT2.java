package xml;

public class HTMLMakerJT2 implements IJTVisitor {

  private StringBuffer buffer_ = new StringBuffer();
  private boolean OLMode = false;
  private boolean OLModeSleep = false;
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

    if (tag.equals("講義")) {
      buffer_.append("<html><body>\n");
    }

    if (tag.equals("授業の概要") ||
	tag.equals("授業項目") ||
	tag.equals("授業の達成目標") ||
	tag.equals("キーワード")) {

      if (tag.equals("授業項目")) {
	tag = "授業項目 (授業計画)";
      }

      if (tag.equals("授業の達成目標")) {
	tag = "授業の達成目標 (学習・教育目標との関連)";
      }
            
      buffer_.append("<h5 align=\"left\"><font color=\"blue\">");
      buffer_.append(tag);
      buffer_.append("</font></h5>\n");
      buffer_.append("<FONT SIZE=-2>"); 

      getMode = true;

    } else if (tag.equals("UL")) {
      if (getMode) {
	buffer_.append("<ul>");
	OLModeSleep = true;      
      }
    } else if (tag.equals("OL")) {
      if (getMode) {
	buffer_.append("<font color=\"red\">");  
	buffer_.append("<dl>");  
	OLMode = true;  
	OLIndex = 1;   
      } 
    } else if (tag.equals("LI")) {
      if (getMode) {
	if (OLMode) { 
	  if (!OLModeSleep) {
	    buffer_.append("<dt> <TT><B>("+OLIndex+")</B>　" + node.getValue() + "</TT>"); 
	    OLIndex++;
	  } else {
	    buffer_.append("<li>" + node.getValue());
	  }
	} else {
	  buffer_.append("<li>" + node.getValue());
	}
      }
    } else if (tag.equals("P")) {
      if (getMode) {
	buffer_.append(node.getValue()); 
      }
    } else {
      getMode = false;
    }
  }
  
  public void leave(JTElement node) {
    String tag = node.getName();

    if (tag.equals("講義")) {
      buffer_.append("</body></html>");
    }
    
    if (tag.equals("OL")) {
      if (getMode) {
	buffer_.append("</dl>\n"); 
	buffer_.append("</font>\n");
	OLMode = false;  
	OLIndex = 0;          
      }
    } else if (tag.equals("UL")) {
      if (getMode) {
	buffer_.append("</ul>\n");
      }
    } else if (tag.equals("P")) {
      if (getMode) {
	buffer_.append("<BR>\n");
	OLModeSleep = false;
      }
    }
  }
  
  public void enter(JTNode node) {
    throw (new InternalError());
  }
  
  public void leave(JTNode node) {
    throw (new InternalError());
  }
}
