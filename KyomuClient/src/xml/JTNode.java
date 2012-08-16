package xml;

public abstract class JTNode {
  private String name_;
  private String value_;
  private boolean leaf_;

  protected JTNode(String name, String value, boolean leaf) {
    name_ = name;
    value_ = value;
    leaf_ = leaf;
  }
  
  public final String getName() {
    return (name_);
  }

  public final void setName(String name) {
    name_ = name;
  }
  
  public final String getValue() {
    return (value_);
  }

  public final void setValue(String value) {
    value_ = value;
  }

  public final boolean isJTLeafNode() {
    return (leaf_);
  }
  
  public String toString() {
    if (value_.length() > 25) {
      String str = value_.substring(0, 24) + ".....";
      return ("<" + name_ + "> " + str);
    } else {
      return ("<" + name_ + "> " + value_);
    }
  }

  public abstract void enter(IJTVisitor visitor);
  public abstract void leave(IJTVisitor visitor);
}
