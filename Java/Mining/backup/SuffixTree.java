import java.lang.*;
import java.util.*;
import SuffixTreeEnumerator;

class SuffixTree extends Object {
	private String referString;
	private String processingStr;
	private int start, end, count, origin;
	protected SuffixTree parent, nsibling, child;
  
    protected SuffixTree() {
		referString = null;
		processingStr = null;
		start = end = count = origin = -1;
		parent = null;
		child = null;
		nsibling = null;
    }
    
    private SuffixTree(SuffixTree parentTree, String str, 
                        int originIndex, int beginIndex, int endIndex){
		parent = parentTree;
		referString = str;
		processingStr = str;
		count = 1;
		origin = originIndex;
		start = beginIndex;
		end = endIndex;
		child = new SuffixTree();
		child.parent = this;
		nsibling = null;
	}
  
	public SuffixTree(Vector strs){
	    this();
		int j;
		start = 0;
		end = 0;
		count = strs.size();
		child = new SuffixTree();
		child.parent = this;
		for (Enumeration e = strs.elements(); e.hasMoreElements(); ){
			referString = (String) e.nextElement();
			for (j = 0; j < referString.length(); j++){
				insertSuffix(referString, j);
			}
		}
		removeEmptyLeaves();
		System.gc();
	}
	    
    public int count() {
        return count;
    }

    public boolean isLeaf(){
        return (child.nsibling == null);
    }
    
    public boolean isLastSibling() {
        return (nsibling == null);
    }

	public boolean isRoot() {
		//if ((start == 0)&&(end == 0)) {
		if (parent == null) {
			return true;
		}
		return false;
	}
  
	public boolean isEmptyLeaf() {
		return (isLeaf() && (start == end));
	}
  
	public SuffixTree nearSibling() {
		SuffixTree trialNode;
		for (trialNode = this; trialNode != null; trialNode = trialNode.nsibling) {
			if (!(trialNode.isEmptyLeaf())) {
				return trialNode;
			}
		}
		trialNode = null;
		return trialNode;
	}
  
	private char charAt(int i){
		if(!(i < referString.length()) ){
			return '@';
		}
		return referString.charAt(i);
	}
  
	private char charAt(String str, int i){
		if(!(i < str.length()) ){
			return '$';
		}
		return str.charAt(i);
	}
	
	public String representingString() {
		int start;
		if (isRoot()) {
			return "";
		}
		return referString.substring(origin, end);
	}

  
  public void removeEmptyLeaves() {
    if (isLeaf()) {
      return;
    }
    for (; (!isLeaf())&& child.nsibling.isEmptyLeaf() ;) {
      child.nsibling = child.nsibling.nsibling;
    }
    SuffixTree tmp;
    for (tmp = child.nsibling; tmp != null; tmp = tmp.nsibling) {
      tmp.removeEmptyLeaves();
    }
  }
    
  private int extendVal(String str, int index) {
    int i;
    for (i = 1; (index + i != str.length())&&(start + i != end); i++){
      if (charAt(start + i) != charAt(str, index + i)){
        break;
      }
    }
    return i;
  }
  
  private SuffixTree insertBreakNode(int breakpt, String procstr, int org) {
    SuffixTree newNode; 
    newNode = new SuffixTree(this.parent, nsibling.referString, nsibling.origin, 
                                nsibling.start, nsibling.start + breakpt);
    //newNode.origin = prev.nsibling.origin;
    newNode.countup(procstr, nsibling); 
    newNode.nsibling = nsibling.nsibling;
    newNode.child.nsibling = nsibling;
    newNode.child.nsibling.start = newNode.end;
    newNode.child.nsibling.parent = newNode;
    newNode.child.nsibling.nsibling = null;
    nsibling = newNode;
    return newNode;
  }
  
  private SuffixTree beforeEqualOrGreaterChild(String str, int index){
    SuffixTree prev;
    for (prev = child; prev.nsibling != null; prev = prev.nsibling){
      if (prev.nsibling.charAt(prev.nsibling.start) >= charAt(str,index)){
	    return prev;
      }
    }
    return prev;
  }
  
  private void addInsertSibling(String str, int index, int org){
    SuffixTree newnode;
    newnode = new SuffixTree(this.parent, str, org, index, str.length());
    newnode.nsibling = nsibling;
    //newnode.origin = org;
    nsibling = newnode;
    return;
  }

  private void countup(String str) {
		if (processingStr != str) {
			count++;
			processingStr = str;
		}
	}

  private void countup(String str, SuffixTree tree) {
    if (tree.processingStr != str) {
      count = tree.count + 1;
      processingStr = str;
    } else {
      count = tree.count;
    }
  }
  
  private void insertSuffix(String str, int idx){
    SuffixTree locus, prev, newNode;
    int skip, org;
    locus = this;
    org = idx;
    while (true) {
      prev = locus.beforeEqualOrGreaterChild(str, idx);
      if (prev.nsibling != null){
    	if (prev.nsibling.charAt(prev.nsibling.start) == charAt(str,idx)){
    	  skip = prev.nsibling.extendVal(str, idx);
    	  if ((! prev.nsibling.isLeaf()) && 
    	      (skip == (prev.nsibling.end - prev.nsibling.start)) ) {
    	    idx = idx + skip;
    	    locus = prev.nsibling;
    	    locus.countup(str);
    	    continue;
    	  }
    	  newNode = prev.insertBreakNode(skip, str, org);
    	  newNode.beforeEqualOrGreaterChild(str,idx).addInsertSibling(str, idx + skip, org);
    	  return;
    	}
      }
      break;
    }
    prev.addInsertSibling(str, idx, org);
    return;
  }
  
  public SuffixTreeEnumerator enumeration() {
    return new SuffixTreeEnumerator(this);
  }
  
  public String toString(){
    String tmp;
    SuffixTree aChild;
    if (isLeaf()){
      return new String(referString.substring(start, end));
    }
    if (start == end){
      tmp = new String("a SuffixTree");
    }else{
      tmp = new String(referString.substring(start, end));
    }
    tmp = tmp + "(";
    for (aChild = child.nsibling; aChild != null; aChild = aChild.nsibling){
      tmp = tmp + aChild.toString();
      if (aChild.nsibling != null){
	tmp = tmp + ",";
      }
    }
    tmp = tmp + ")";
    return tmp;
  }
  

  public static void main(String args[]){
    SuffixTree tree;
    Vector strs = new Vector();
    strs.addElement(new String("abbabb"));
    strs.addElement(new String("ababba"));
    tree = new SuffixTree(strs);
    System.out.println(tree);
  }
  
}

