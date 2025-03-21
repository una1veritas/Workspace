import java.lang.*;
import java.util.*;
import java.io.*;
import Occurrence;

class SuffixTree implements Serializable {
    private String texts[];
    private Node root;
    private HashMap occurrenceTable;
    
    public SuffixTree(Vector strs) {
        texts = new String[strs.size()];
        strs.copyInto(texts);
        root = new Node(texts);
    }
    
    public SuffixTree(String strs[]) {
        texts = strs;
        root = new Node(texts);
    }
    
    public SuffixTree(String strs[], String delims) {
        texts = strs;
        root = new Node(texts, delims);
    }

    public SuffixTree(SuffixTree tree) {
        texts = tree.texts;
        root = tree.root;
    }
    
    public Node root() {
        return root;
    }

  public Node findLocus(String w) {
    return root.findLocus(w);
  }
    
    public SuffixTreeEnumerator enumeration() {
        return new SuffixTreeEnumerator(this);
    }
        
    public synchronized String toString() {
        return root.toString();
    }
        
    protected class Node implements Serializable, Comparable {
        private int textid, index;
        private int labelStart, labelEnd;
        private Node parent, sibling, child;
    
        protected Node() {
            labelStart = labelEnd = -1;
            textid = index = -1;
            parent = null;
            child = null;
            sibling = null;
        }
        
        private Node(Node parentNode, int id, 
                    int originIndex, int beginIndex, int endIndex){
            this();
            textid = id;
            index = originIndex;
            parent = parentNode;
            labelStart = beginIndex;
            labelEnd = endIndex;
        }
        
        public Node(int id, int idx) {
            this();
            textid = id;
            index = idx;
        }
      
        public Node(String strs[]){
            this();
            for (int i = 0; i < strs.length; i++){
                for (int j = 0; j < strs[i].length(); j++){
		  this.insertSuffix(i, j);
                }
            }
        }
        
        public Node(String strs[], String delims){
  	    this();
	    int i, j, k;
	    for (i = 0; i < strs.length; i++){
	        j = 0;
		while (j < strs[i].length()) {
		    // skip delimiters given in `delims.'
		    for ( ; j < strs[i].length(); j++) {
		        for (k = 0; k < delims.length(); k++) {
			    if (strs[i].charAt(j) == delims.charAt(k)) {
			        break;
			    }
			}
			if (! (k < delims.length())) {
			    break; 
			}
		    }
		    if (! (j < strs[i].length() )) {
		      break;
		    }
		    // j points non-delimiter symbol.
		    this.insertSuffix(i, j);
		    // skip non-delimiter symbols. 
		    for ( ; j < strs[i].length(); j++) {
		        for (k = 0; k < delims.length(); k++) {
			    if (strs[i].charAt(j) == delims.charAt(k)) {
			        break;
			    }
			}
			if ( k < delims.length() ) {
			    break; 
			}
		    }
		}
	    
	    }
	}
        
        public int textid() {
            return textid;
        }
        
        public int index() {
            return index;
        }
        
        public boolean equals(Object obj) {
            if (! (obj instanceof SuffixTree.Node)) {
                return false;
            }
            return (textid == ((SuffixTree.Node) obj).textid) 
                    && (index == ((SuffixTree.Node) obj).index);
        }
        
        public int compareTo(Object o) {
            int res;
            if (! (o instanceof SuffixTree.Node)) {
                return -1;
            }
            if ((res = textid - ((SuffixTree.Node) o).textid) == 0) {
                return index - ((SuffixTree.Node) o).index;
            } else {
                return res;
            }
        }
        /*
        public int hashCode() {
	        long bits = (long) index;
	        bits ^= ((long) textid) * 31;
	        return (((int) bits) ^ ((int) (bits >> 32)));
        }
        */
        
        private char charAt(int id, int pos){
            if( pos == texts[id].length() ){
                return '\255';
            } else {
                return texts[id].charAt(pos);
            }
        }

        private char charAt(int i){
            return charAt(textid, i);
        }
        
        protected Node firstChild() {
            return child;
        }
          
        protected Node firstChild(Node node) {
            child = node;
            return child;
        }
        
        protected Node sibling() {
            return sibling;
        }
        
        protected Node parent() {
            return parent;
        }
        
        public boolean isRoot() {
            if (parent == null) {
                return true;
            }
            return false;
        }
          
        public boolean isLeaf(){
            return (child == null);
        }
        
        public boolean isLastSibling() {
            return (sibling == null);
        }
        
        public boolean isEmptyLeaf() {
            return (isLeaf() && (labelStart == labelEnd));
        }
        
        public SortedSet occurrences() {
            SortedSet set = new TreeSet();
            //System.out.print("#");
            if ( isLeaf() ) {
                set.add(new Occurrence(textid, index));
                return set;
            }
            for (Node aChild = child; aChild != null; aChild = aChild.sibling) {
                set.addAll(aChild.occurrences());
            }
            return set;
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

        
        protected boolean isDummyLeader() {
            return (this == child);
        }
         
        protected Node childrensDummyLeader() {
            Node leader;
            leader = new Node();
            leader.parent = this;
            leader.sibling = firstChild();
            leader.child = leader;
            return leader;
        }

        public String pathLabel() {
            if (isRoot()) {
                return "";
            }
            return texts[textid].substring(index, labelEnd);
        }

        public String edgeLabel() {
	  if (isRoot()) {
	    return "";
	  }
	  return texts[textid].substring(labelStart, labelEnd);
        }

        private Node beforeEqualOrGreaterChild(int id, int pos){
            Node prev;
            for (prev = childrensDummyLeader(); prev.sibling != null; prev = prev.sibling){
                if (prev.sibling.charAt(prev.sibling.labelStart) >= charAt(id, pos)){
	                return prev;
                }
            }
            return prev;
        }

        private Node insertBreakNode(int breakpt, int id, int org) {
            Node newnode; 
            newnode = new Node(this.parent, sibling.textid, sibling.index, 
			            sibling.labelStart, sibling.labelStart + breakpt);
            newnode.sibling = sibling.sibling;
            newnode.firstChild(sibling);
            newnode.firstChild().labelStart = newnode.labelEnd;
            newnode.firstChild().parent = newnode;
            newnode.firstChild().sibling = null;
            if (isDummyLeader()) {
                // this is a dummpy leader
                parent.firstChild(newnode);
                return newnode;
            }
            sibling = newnode;
            return newnode;
        }

        private void addInsertSibling(int id, int start, int idx){
            Node newnode;
            newnode = new Node(this.parent, id, idx, start, texts[id].length());
            newnode.sibling = sibling;
            if (isDummyLeader()) {
                // this is a dummy leader
                parent.firstChild(newnode);
                return;
            }
            sibling = newnode;
            return;
        }

        private int extendVal(int id, int index) {
            int i;
            for (i = 1; (index + i != texts[id].length())&&(labelStart + i != labelEnd); i++){
                if (charAt(labelStart + i) != charAt(id, index + i)){
                    break;
                }
            }
            return i;
        }
        
        private void insertSuffix(int id, int idx) {
            Node locus, prev, newnode;
            int skip, org;
            locus = this;
            org = idx;
            while (true) {
                prev = locus.beforeEqualOrGreaterChild(id, idx);
                if (prev.sibling != null){
    	            if (charAt(id,idx) != '\255' 
    	                && prev.sibling.charAt(prev.sibling.labelStart) != '\255' 
    	                && prev.sibling.charAt(prev.sibling.labelStart) == charAt(id,idx) ){
    	            skip = prev.sibling.extendVal(id, idx);
    	            if ((! prev.sibling.isLeaf()) && 
    	                (skip == (prev.sibling.labelEnd - prev.sibling.labelStart)) ) {
    	                idx = idx + skip;
    	                locus = prev.sibling;
    	                continue;
    	            }
    	            newnode = prev.insertBreakNode(skip, id, org);
    	            newnode.beforeEqualOrGreaterChild(id, idx + skip).addInsertSibling(id, idx + skip, org);
    	            return;
    	            }
                }
                break;
            }
            prev.addInsertSibling(id, idx, org);
            return;
        }

      protected Node findLocus(String w) {
	Node locus, prev;
	int skip, pos; 

	locus = this;
	pos = 0;
	while (true) {
	  for (prev = locus.childrensDummyLeader(); prev.sibling != null; prev = prev.sibling){
	    if (prev.sibling.charAt(prev.sibling.labelStart) >= w.charAt(pos)){
	      break;
	    }
	  }
	  if (prev.sibling == null) {
	    break;
	  }
	  if (prev.sibling.charAt(prev.sibling.labelStart) != w.charAt(pos) ){
	    break;
	  }
	  for (skip = prev.sibling.labelStart; 
	       skip < prev.sibling.labelEnd && pos < w.length(); 
	       skip++, pos++) {
	    if (prev.sibling.charAt(skip) == w.charAt(pos)) {
	      //
	      continue;
	    }
	  }
	  if ( pos == w.length() ) {
	    //System.out.print(prev.sibling.pathLabel()+";");
	    return prev.sibling;
	  }
	  if ( skip == prev.sibling.labelEnd ) {
	    //System.out.print(locus.pathLabel()+"("+pos+">");
	    locus = prev.sibling;
	    continue;
	  }
	  break;
	}
	return null;
      }

        public synchronized String toString(){
            StringBuffer strbuf = new StringBuffer();
            //String tmp;
            Node aChild;
            if (isLeaf()){
	      return texts[textid].substring(labelStart, labelEnd); //+":"+ String.valueOf(index);
            }
            if (labelStart == labelEnd){
                strbuf.append("a SuffixTree.Node");
            }else{
                strbuf.append(texts[textid].substring(labelStart, labelEnd));
            }
            strbuf.append("(");
            for (aChild = firstChild(); aChild != null; aChild = aChild.sibling){
                strbuf.append(aChild.toString());
                if (aChild.sibling != null){
	                strbuf.append(",");
                }
            }
            strbuf.append(")");
            return strbuf.toString();
        }

    }


  public static void main(String args[]){
    SuffixTree tree;
    Vector strs = new Vector();
    tree = new SuffixTree(args," \t\n\r");
    System.out.println(tree);
  }
  
}

