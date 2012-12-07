import SuffixTree;
import java.util.*;

class SuffixTreeEnumerator extends SuffixTree implements Enumeration {
    SuffixTree current;
    
    public SuffixTreeEnumerator(SuffixTree root) {
        parent = root;
        child = root.child;
        current = root;
        if (!current.isLeaf()) {
            nsibling = root.child.nsibling;
        } else {
            nsibling = null;
        }
    }
    
    private SuffixTree mostLowestNonLastAncestor(SuffixTree node) {
        SuffixTree trialNode;
        for (trialNode = node; !trialNode.isRoot(); ) {
            trialNode = trialNode.parent;
            if (!trialNode.isLastSibling()) {
                return trialNode;
            }
        }
        return trialNode;
    }
    
    public boolean hasMoreNodes() {
        return (nsibling != null);
    }
    
    public boolean hasMoreElements() {
        return hasMoreNodes();
    }
    
    public SuffixTree current() {
        return current;
    }
    
    public SuffixTree next() {
        current = nsibling;
        /*
        if (current == null) {
            return current;
        }
        */
        if (!nsibling.isLeaf()) {
            nsibling = nsibling.child.nsibling;
            return current;
        }
        if (!nsibling.isLastSibling()) {
            nsibling = nsibling.nsibling;
            return current;
        }
        nsibling = mostLowestNonLastAncestor(nsibling);
        if (nsibling.isRoot()) {
            //System.out.println("no more node");
            nsibling = null;
        } else {
            nsibling = nsibling.nsibling;
        }
        return current;
    }
    
    public Object nextElement() {
        return (Object) next();
    }
    
    public void skipChildren() {
        if (nsibling == null) {
            return;
        }
        if (!nsibling.isLastSibling()) {
            nsibling = nsibling.nsibling;
            return;
        }
        nsibling = mostLowestNonLastAncestor(nsibling);
        if (nsibling.isRoot()) {
            nsibling = null;
        } else {
            nsibling = nsibling.nsibling;
        }
        return;
    }
    
    public SuffixTree nextSibling() {
        current = nsibling;
        nsibling = nsibling.nsibling;
        return current;
    }
    
    public void setFirstChild() {
        parent.child = child;
        return;
    }
    
/*
  public static void main(String[] args) {
    String[] str = new String[10];
    str[0] = new String("abbabb");
    str[1] = new String("abbabb");
    SuffixTree tree = new SuffixTree(str);
    System.out.println(tree);
    SuffixTreeEnumerator e = new SuffixTreeEnumerator(tree);
    SuffixTree node, node1;
    for ( ;e.hasMoreNodes(); ) {
      SuffixTreeEnumerator e1 = new SuffixTreeEnumerator(tree);
      node = e.nextNode();
      for (;e1.hasMoreNodes();) {
	node1 = e1.nextNode();
	System.out.println(node.getStrings()+" "+node1.getStrings());
      }
    }
  }
  */
}
