package xml;
//import java.util.List;
//import java.util.ArrayList;

public class JTElement extends JTNode {
  public JTElement(String name, String value, boolean leaf) {
    super(name, value, leaf);
  }

  public void enter(IJTVisitor visitor) {
    visitor.enter(this);
  }

  public void leave(IJTVisitor visitor) {
    visitor.leave(this);
  }
}
