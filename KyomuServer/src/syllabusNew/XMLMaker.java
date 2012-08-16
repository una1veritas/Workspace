package syllabusNew;
import org.w3c.dom.*;

public class XMLMaker implements IDOMVisitor {
  protected StringBuffer buffer;
  protected String encoding = "EUC-JP";

  public XMLMaker() {
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

    if (tag.equals("¹ÖµÁ")) {
      buffer.append("<?xml version=\"1.0\" encoding=\"EUC-JP\"?>\n");
    }
    
    buffer.append("<");
    buffer.append(tag);

    NamedNodeMap attrs = element.getAttributes();
    int nAttrs = attrs.getLength();
    for (int i = 0; i < nAttrs; i++) {
      Attr attr = (Attr)attrs.item(i);
      if (attr.getSpecified()) {
	buffer.append(' ');
	enter(attr);
	leave(attr);
      }
    }
    buffer.append(">");
    return (true);
  }

  public void leave(Element element) {
    String tag = element.getTagName();
    buffer.append("</" + tag + ">");
  }
  
  public boolean enter(Attr attr) {
    buffer.append(attr.getName());
    buffer.append("=\"");
    buffer.append(UDOM.escapeAttrQuot(attr.getValue()));
    buffer.append("\"");
    return (true);
  }

  public void leave(Attr attr) {
    // do nothing
  }
  
  public boolean enter(Text text) {
    buffer.append(UDOM.escapeCharData(text.getData()));
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
