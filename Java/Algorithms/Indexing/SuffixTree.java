import java.lang.*;
import java.util.*;
import java.io.*;
import SuffixTreeEnumerator;

public class SuffixTree extends Object implements Serializable {
	private String referString;
	private String processingStr;
	private int start, end, count, origin;
	protected SuffixTree parent, sibling, child;
  
    protected SuffixTree() {
		referString = null;
		processingStr = null;
		start = end = count = origin = -1;
		parent = null;
		child = null;
		sibling = null;
    		//{{INIT_CONTROLS
		//}}
}
    
    private SuffixTree(SuffixTree parentTree, String str, 
                        int originIndex, int beginIndex, int endIndex){
		this();
		parent = parentTree;
		referString = str;
		processingStr = str;
		count = 1;
		origin = originIndex;
		start = beginIndex;
		end = endIndex;
	}
  
	public SuffixTree(Vector strs){
	    this();
		int j;
		start = 0;
		end = 0;
		count = strs.size();
		for (Enumeration e = strs.elements(); e.hasMoreElements(); ){
			referString = (String) e.nextElement();
			for (j = 0; j < referString.length(); j++){
				insertSuffix(referString, j);
			}
		}
		//removeEmptyLeaves();
		System.gc();
	}
/*	
	protected void finalize() throws Throwable {
	    for (SuffixTree tmp = child; tmp != null; tmp = tmp.sibling) {
	        tmp.finalize();
	    }
	    super.finalize();
	}
*/	
    public int count() {
        return count;
    }

    protected SuffixTree firstChild() {
        return child;
    }
    
    protected SuffixTree firstChild(SuffixTree node) {
        child = node;
        return child;
    }
    
    protected boolean isDummyLeader() {
        return (this == child);
    }
    
    protected SuffixTree childrensDummyLeader() {
        SuffixTree leader;
        leader = new SuffixTree();
        leader.parent = this;
        leader.sibling = firstChild();
        leader.child = leader;
        return leader;
    }
    
    public boolean isLeaf(){
        return (child == null);
    }
    
    public boolean isLastSibling() {
        return (sibling == null);
    }

	public boolean isRoot() {
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
		for (trialNode = this; trialNode != null; trialNode = trialNode.sibling) {
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
    for (; (!isLeaf())&& firstChild().isEmptyLeaf() ;) {
      firstChild(firstChild().sibling);
    }
    SuffixTree tmp;
    for (tmp = firstChild(); tmp != null; tmp = tmp.sibling) {
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
    newNode = new SuffixTree(this.parent, sibling.referString, sibling.origin, 
                                sibling.start, sibling.start + breakpt);
    newNode.countup(procstr, sibling); 
    newNode.sibling = sibling.sibling;
    newNode.firstChild(sibling);
    newNode.firstChild().start = newNode.end;
    newNode.firstChild().parent = newNode;
    newNode.firstChild().sibling = null;
    if (isDummyLeader()) {
        // this is a dummpy leader
        parent.firstChild(newNode);
        return newNode;
    }
    sibling = newNode;
    return newNode;
  }
  
  private SuffixTree beforeEqualOrGreaterChild(String str, int index){
    SuffixTree prev;
    for (prev = childrensDummyLeader(); prev.sibling != null; prev = prev.sibling){
      if (prev.sibling.charAt(prev.sibling.start) >= charAt(str,index)){
	    return prev;
      }
    }
    return prev;
  }
  
  private void addInsertSibling(String str, int index, int org){
    SuffixTree newnode;
    newnode = new SuffixTree(this.parent, str, org, index, str.length());
    newnode.sibling = sibling;
    if (isDummyLeader()) {
        // this is a dummy leader
        parent.firstChild(newnode);
        return;
    }
    sibling = newnode;
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
      if (prev.sibling != null){
    	if (prev.sibling.charAt(prev.sibling.start) == charAt(str,idx)){
    	  skip = prev.sibling.extendVal(str, idx);
    	  if ((! prev.sibling.isLeaf()) && 
    	      (skip == (prev.sibling.end - prev.sibling.start)) ) {
    	    idx = idx + skip;
    	    locus = prev.sibling;
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
    for (aChild = firstChild(); aChild != null; aChild = aChild.sibling){
      tmp = tmp + aChild.toString();
      if (aChild.sibling != null){
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
  
	//{{DECLARE_CONTROLS
	//}}
}

