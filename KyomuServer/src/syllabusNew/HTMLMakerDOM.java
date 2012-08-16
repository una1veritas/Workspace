package syllabusNew;
import org.w3c.dom.*;

public class HTMLMakerDOM implements IDOMVisitor {
  protected StringBuffer buffer;
  protected String encoding = "EUC-JP";

  public HTMLMakerDOM() {
    buffer = new StringBuffer();
  }

  public void setEncoding(String encoding) {
    encoding = encoding;
  }

  public String getText() {
    return (new String(buffer));
  }
  
  public boolean enter(Element element) {
    String tag = element.getTagName();

    if (tag.equals("�ֵ�")) {
      buffer.append("<html><body>");
    }
    
    if (tag.equals("�ֵ�����") || tag.equals("�����դ�") ||
        tag.equals("�ֵ�����") || tag.equals("�ʤ���") ||
        tag.equals("ɾ����ˡ") || tag.equals("����") ||
        tag.equals("���ʽ�") || tag.equals("���ͽ�") || 
	tag.equals("�������") ||
	tag.equals("���Ȥγ���") ||
	tag.equals("���ꥭ����ˤ����뤳�μ��Ȥΰ����դ�") ||
	tag.equals("���ȹ���") ||
	tag.equals("���Ȥοʤ���") ||
	tag.equals("���Ȥ�ã����ɸ") ||
	tag.equals("����ɾ���δ�प���ɾ����ˡ")) {
      
      if (element.hasChildNodes()) {
        buffer.append("<h2 align=\"left\"><font color=\"blue\">");
        buffer.append(tag);
        buffer.append("</font></h2>");
      }
      
    } else if (tag.equals("UL")) {
      
      buffer.append("<ul>");
      
    } else if (tag.equals("OL")) {
      
      buffer.append("<ol>");
      
    } else if (tag.equals("LI")) {
      
      buffer.append("<li>");
      
    } else if (tag.equals("P")) {

      buffer.append("<p>");

    }      

    NamedNodeMap attrs = element.getAttributes();
    int nAttrs = attrs.getLength();
    for (int i = 0; i < nAttrs; i++) {
      Attr attr = (Attr)attrs.item(i);
      if (attr.getSpecified()) {
        enter(attr);
        leave(attr);
      }
    }
    return (true);
  }

  public void leave(Element element) {
    String tag = element.getTagName();

    if (tag.equals("�ֵ�")) {
      buffer.append("</body></html>");
    }
    
    if (tag.equals("OL")) {
      buffer.append("</ol>");
    } else if (tag.equals("UL")) {
      buffer.append("</ul>");
    } else if (tag.equals("P")) {
      buffer.append("</p>");
    }
  }
  
  public boolean enter(Attr attr) {
    return (true);
  }
  
  public void leave(Attr attr) {
    // do nothing
  }
  
  public boolean enter(Text text) {
    String str = UDOM.escapeCharData(text.getData());
    if ("".equals(str.trim()) == false)
      buffer.append(str.trim());
    return (true);
  }

  public void leave(Text text) {
    // do nothing
  }
  
  public boolean enter(CDATASection cdata) {
    return (true);
  }
  
  public void leave(CDATASection cdata) {
    // do nothing
  }
  
  public boolean enter(EntityReference entityRef) {
    return (false);
  }

  public void leave(EntityReference entityRef) {
    // do nothing
  }
  
  public boolean enter(Entity entity) {
    return (false);
  }

  public void leave(Entity entity) {
    // do nothing
  }
  
  public boolean enter(ProcessingInstruction pi) {
    return (true);
  }
  
  public void leave(ProcessingInstruction pi) {
    // do nothing
  }
  
  public boolean enter(Comment comment) {
    return (true);
  }
  
  public void leave(Comment comment) {
    // do nothing
  }
  public boolean enter(Document doc) {
    return (true);
  }

  public void leave(Document doc) {
    // do nothing
  }
  
  public boolean enter(DocumentType doctype) {
    return (true);
  }
  
  public void leave(DocumentType doctype) {
    // do nothing
  }
  
  public boolean enter(DocumentFragment docfrag) {
    // do nothing
    return (true);
  }

  public void leave(DocumentFragment docfrag) {
  }
  
  public boolean enter(Notation notation) {
    return (true);
  }
  
  public void leave(Notation notation) {
    // do nothing
  }
  
  public boolean enter(Node node) {
    throw (new InternalError());
  }

  public void leave(Node node) {
    throw (new InternalError());
  }
}
