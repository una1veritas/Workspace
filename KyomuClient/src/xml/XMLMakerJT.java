package xml;

public class XMLMakerJT implements IJTVisitor {
  private StringBuffer buffer_ = new StringBuffer();
  private int indent_ = 0;
  private boolean flg_ = true;
  private boolean firstflg_ = true;
 
  public String getText() {
    return (new String(buffer_));
  }

  public void up() {
    indent_++;
  }

  public void down() {
    indent_--;
  }

  private void _makeIndent() {
    for (int length = indent_; length > 0; length--) {
      buffer_.append("\t");
    }
  }
  
  public void enter(JTAttr node) {
    String str = " " + node.getName() + "=\"" +
      UDOM.escapeAttrQuot(node.getValue()) + "\"";
    buffer_.insert(buffer_.length() - 1, str);
  }

  public void leave(JTAttr node) {
    // do nothing
  }
  
  public void enter(JTElement node) {
    flg_ = true;
    if (firstflg_ == false) {
      buffer_.append("\n");
    }
    firstflg_ = false;
    _makeIndent();
    
    String tag = node.getName();

    if (tag.equals("P") || tag.equals("LI")) {

      buffer_.append("<" + tag + ">");
      buffer_.append(UDOM.escapeCharData(node.getValue()));
      
    } else {
      
      buffer_.append("<" + tag + ">");
    }
    up();
  }

  public void leave(JTElement node) {
    down();
    if (flg_ == false) {
      buffer_.append("\n");
      _makeIndent();
    }
    String tag = node.getName();
    buffer_.append("</" + tag + ">");
    flg_ = false;
  }

  public void enter(JTNode node) {
    throw (new InternalError());
  }

  public void leave(JTNode node) {
    throw (new InternalError());
  }
}
