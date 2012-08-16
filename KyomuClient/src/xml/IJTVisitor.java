package xml;

public interface IJTVisitor {
  void enter(JTAttr node);
  void leave(JTAttr node);
  void enter(JTElement node);
  void leave(JTElement node);
  void enter(JTNode node);
  void leave(JTNode node);
}
