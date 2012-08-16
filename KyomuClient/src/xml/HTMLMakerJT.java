package xml;

public class HTMLMakerJT implements IJTVisitor {

  private StringBuffer buffer_ = new StringBuffer();
  private boolean OLMode = false;
  private boolean OLModeSleep = false;
  private int OLIndex = 0;
 
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
    
    if (tag.equals("講義内容") || tag.equals("位置付け") ||
        tag.equals("講義項目") || tag.equals("進め方") ||
        tag.equals("評価方法") || tag.equals("備考") ||
        tag.equals("教科書") || tag.equals("参考書") || 
	tag.equals("キーワード") ||
	tag.equals("授業の概要") ||
	tag.equals("カリキュラムにおけるこの授業の位置付け") ||
	tag.equals("授業項目") ||
	tag.equals("授業の進め方") ||
	tag.equals("授業の達成目標") ||
	tag.equals("成績評価の基準および評価方法")) {    

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

    } else if (tag.equals("UL")) {
      buffer_.append("<ul>");
      OLModeSleep = true;      
    } else if (tag.equals("OL")) {
      buffer_.append("<dl>"); 
      OLMode = true;
      OLIndex = 1;  
    } else if (tag.equals("LI")) {
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
    } else if (tag.equals("P")) {
      buffer_.append(node.getValue()); 
    }
  }
  
  public void leave(JTElement node) {
    String tag = node.getName();

    if (tag.equals("講義")) {
      buffer_.append("</body></html>");
    }
    
    if (tag.equals("OL")) {
      buffer_.append("</dl>\n");
      OLMode = false; 
      OLIndex = 0;      
    } else if (tag.equals("UL")) {
      buffer_.append("</ul>\n");
      OLModeSleep = false;
    } else if (tag.equals("P")) {
      buffer_.append("<BR>\n");
    }
  }
  
  public void enter(JTNode node) {
    throw (new InternalError());
  }
  
  public void leave(JTNode node) {
    throw (new InternalError());
  }
}
