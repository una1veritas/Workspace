package xml;
import javax.swing.tree.*;
import java.util.*;

public final class UJTVisitor {
  public static void traverse(DefaultMutableTreeNode node, IJTVisitor visitor) {

    JTNode jtnode = (JTNode)node.getUserObject();
    jtnode.enter(visitor);
    Enumeration<DefaultMutableTreeNode> enu = node.children();

    while (enu.hasMoreElements()) {
      DefaultMutableTreeNode mutable =
        (DefaultMutableTreeNode)enu.nextElement();
      traverse(mutable, visitor);
    }
    jtnode.leave(visitor);
  }
}
