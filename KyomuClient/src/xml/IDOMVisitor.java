package xml;
import org.w3c.dom.*;

public interface IDOMVisitor {
  boolean enter(Element element) throws DOMVisitorException;
  boolean enter(Attr attr) throws DOMVisitorException;
  boolean enter(Text text) throws DOMVisitorException;
  boolean enter(CDATASection cdata) throws DOMVisitorException;
  boolean enter(EntityReference entityRef) throws DOMVisitorException;
  boolean enter(Entity entity) throws DOMVisitorException;
  boolean enter(ProcessingInstruction pi) throws DOMVisitorException;
  boolean enter(Comment comment) throws DOMVisitorException;
  boolean enter(Document doc) throws DOMVisitorException;
  boolean enter(DocumentType doctype) throws DOMVisitorException;
  boolean enter(DocumentFragment docfrag) throws DOMVisitorException;
  boolean enter(Notation notation) throws DOMVisitorException;
  boolean enter(Node node) throws DOMVisitorException;
  void leave(Element element) throws DOMVisitorException;
  void leave(Attr attr) throws DOMVisitorException;
  void leave(Text text) throws DOMVisitorException;
  void leave(CDATASection cdata) throws DOMVisitorException;
  void leave(EntityReference entityRef) throws DOMVisitorException;
  void leave(Entity entity) throws DOMVisitorException;
  void leave(ProcessingInstruction pi) throws DOMVisitorException;
  void leave(Comment comment) throws DOMVisitorException;
  void leave(Document doc) throws DOMVisitorException;
  void leave(DocumentType doctype) throws DOMVisitorException;
  void leave(DocumentFragment docfrag) throws DOMVisitorException;
  void leave(Notation notation) throws DOMVisitorException;
  void leave(Node node) throws DOMVisitorException;
}
