package xml;

public class JTAttr extends JTNode {
  public JTAttr(String name, String value) {
    super(name, value, false);

    /*
    super(name, value, true);
     */
  }

  public void enter(IJTVisitor visitor) {
    visitor.enter(this);
  }

  public void leave(IJTVisitor visitor) {
    visitor.leave(this);
  }
}
