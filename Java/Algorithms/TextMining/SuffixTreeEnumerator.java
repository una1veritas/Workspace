import SuffixTree;
import Occurrence;
import java.util.*;

class SuffixTreeEnumerator implements Enumeration {
    SuffixTree.Node current, next;
    HashMap occurrenceTable;
    
    public SuffixTreeEnumerator(SuffixTree tree) {
        occurrenceTable = new HashMap();
        current = tree.root();
        next = lowestFirstDescendantOf(current);
        if (!next.isRoot()) {
            if ( next.isEmptyLeaf() ) {
	      //next = current.parent();
	      next();
            }
        } else {
            next = null;
        }
    }
    
    public SortedSet occurrences() {
        SortedSet result;
        SuffixTree.Node aChild;
        
        if ( (result = (SortedSet) occurrenceTable.get(current)) != null ) {
            return result;
        }
        if (current.isLeaf()) {
            result = current.occurrences();
            occurrenceTable.put(current, result);
            return result;
        }
        result = new TreeSet();
        for (aChild = current.firstChild(); aChild != null; aChild = aChild.sibling() ) {
            if (occurrenceTable.containsKey(aChild)) {
                result.addAll((SortedSet) occurrenceTable.get(aChild));
                occurrenceTable.remove(aChild);
            } else {
                result.addAll(aChild.occurrences());
            }
        }
        occurrenceTable.put(current, result);
        return result;
    }
        
    public HashSet hits() {
        Occurrence occ;
        HashSet set = new HashSet();
        
        for (Iterator i = occurrences().iterator(); i.hasNext(); ) {
            occ = (Occurrence) i.next();
            set.add(new Integer(occ.identifier));
        }
        return set;        
    }
        
    private SuffixTree.Node lowestFirstDescendantOf(SuffixTree.Node node) {
        SuffixTree.Node t;
        for (t = node; ! t.isLeaf() ; t = t.firstChild()) ;
        return t;
    }
    
    
    
    /*
    private SuffixTree.Node nextOfMostLowestNonLastAncestor(SuffixTree.Node node) {
        SuffixTree.Node trialNode;
        for (trialNode = node; !trialNode.isRoot(); ) {
            trialNode = trialNode.parent;
            if ( !trialNode.isLastSibling() ) {
	            for ( trialNode = trialNode.sibling;
	                trialNode.isEmptyLeaf() && (! trialNode.isLastSibling()) ;
	                trialNode = trialNode.sibling ) ;
	            if (! trialNode.isEmptyLeaf() ) {
	            return trialNode;
	            }
            }
        }
        return null; // reached to the root
    }
    */
    
    public boolean hasMoreNodes() {
        return (next != null);
    }
    
    public boolean hasMoreElements() {
        return hasMoreNodes();
    }

  public boolean hasNext() {
    return hasMoreNodes();
  }
    
    private SuffixTree.Node current() {
        return current;
    }
    
    public String pathLabel() {
        return current().pathLabel();
    }
    
  private SuffixTree.Node next() {
        // current is always the result value;
        current = next;
        if ( next.isLastSibling() ) {
            next = next.parent();
            return current;
        }
        next = next.sibling();
        if ( next.isEmptyLeaf() ) {
            next = next.parent();
            return current;
        }
        next = lowestFirstDescendantOf(next);
        return current;
    }

    /*
    private SuffixTree.Node next() {
        // current is always the result value;
        current = next;
        if (!next.isLeaf()) {
            next = next.firstChild();
            return current;
        }
	    for ( ;! next.isLastSibling(); ) {
	        next = next.sibling();
	        if ( ! next.isEmptyLeaf() ) {
	            return current;
	        } else {
	            break;
	        }
	    }
	    // if next was the last one, or there is no succeeding non empty leaf. 
        next = nextOfMostLowestNonLastAncestor(next);
        // if (next == null) then no more nodes.
        return current;
    }
    */
    
    public Object nextElement() {
        return (Object) next();
    }
    /*
    public void skipChildren() {
        if (next == null) {
            return;
        }
        if (!next.isLastSibling()) {
            next = next.sibling();
            return;
        }
        next = nextOfMostLowestNonLastAncestor(next);
        return;
    }
    
    public SuffixTree.Node nextSibling() {
        current = next;
        next = next.sibling();
        return current;
    }
    */
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
