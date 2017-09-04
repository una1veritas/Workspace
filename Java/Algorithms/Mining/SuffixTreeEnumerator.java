import SuffixTree;
import java.util.*;

class SuffixTreeEnumerator implements Enumeration {
    SuffixTree current, next, child, parent;
    
    public SuffixTreeEnumerator(SuffixTree root) {
        parent = root;
        child = root.child;
        current = root;
        if (!current.isLeaf()) {
            //next = root.firstChild();
            next = root.child;
        } else {
            next = null;
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
        return (next != null);
    }
    
    public boolean hasMoreElements() {
        return hasMoreNodes();
    }
    
    public SuffixTree current() {
        return current;
    }
    
    public SuffixTree next() {
        current = next;
        /*
        if (current == null) {
            return current;
        }
        */
        if (!next.isLeaf()) {
            //next = next.firstChild();
            next = next.child;
            return current;
        }
        for ( ; next.isEmptyLeaf() && (!next.isLastSibling()) ; next = next.sibling) ;
        if (!next.isLastSibling()) {
            next = next.sibling;
            return current;
        }
        next = mostLowestNonLastAncestor(next);
        if (next.isRoot()) {
            //System.out.println("no more node");
            next = null;
        } else {
            next = next.sibling;
        }
        return current;
    }
    
    public Object nextElement() {
        return (Object) next();
    }
    
    public void skipChildren() {
        if (next == null) {
            return;
        }
        if (!next.isLastSibling()) {
            next = next.sibling;
            return;
        }
        next = mostLowestNonLastAncestor(next);
        if (next.isRoot()) {
            next = null;
        } else {
            next = next.sibling;
        }
        return;
    }
    
    public SuffixTree nextSibling() {
        current = next;
        next = next.sibling;
        return current;
    }
/*    
    public void setFirstChild() {
        parent.child = child;
        return;
    }
*/    
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
