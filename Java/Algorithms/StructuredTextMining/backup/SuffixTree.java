import java.lang.*;
import java.util.*;
import java.io.*;

class SuffixTree extends Object implements Serializable {
  private String referString;
  private int start, end, origin;
  protected SuffixTree parent, sibling, child;
  
  protected class StringReference {
    String string;
    
    StringReference(String ref) {
      string = ref;
    }
    
    public int hashCode() {
      return string.hashCode();
    }
    
    public synchronized String toString() {
      return "["+string+"]";
    }
    
    public boolean equals(Object anObject) {
      if (this == anObject) {
	return true;
      }
      if ((anObject != null) && (anObject instanceof StringReference)) {
	return (string == ((StringReference) anObject).string);
      }
      return false;
    }
  }
  
  protected SuffixTree() {
    referString = null;
    start = end = origin = -1;
    parent = null;
    child = null;
    sibling = null;
  }
  
  private SuffixTree(SuffixTree parentTree, String str, 
		     int originIndex, int beginIndex, int endIndex){
    this();
    parent = parentTree;
    referString = str;
    origin = originIndex;
    start = beginIndex;
    end = endIndex;
  }
  
  public SuffixTree(Vector strs){
    this();
    int j;
    start = 0;
    end = 0;
    for (Enumeration e = strs.elements(); e.hasMoreElements(); ){
      referString = (String) e.nextElement();
      for (j = 0; j < referString.length(); j++){
	insertSuffix(referString, j);
      }
    }
  }
  
  public SuffixTree(String strs[]){
    this();
    start = 0;
    end = 0;
    for (int i = 0; i < strs.length ; i++){
      for (int j = 0; j < strs[i].length() ; j++){
	insertSuffix(strs[i], j);
      }
    }
  }
  
  public int childrenCount() {
    int sum = 0;
    if ( isLeaf() ) {
      return 1;
    }
    for (SuffixTree tmp = child; tmp != null; tmp = tmp.sibling) {
      sum = sum + tmp.childrenCount();
    }
    return sum;
  }
  
  public HashSet occurrences() {
    StringReference tmp;
    HashSet set = new HashSet();
    if ( isLeaf() ) {
      set.add(new StringReference(referString));
      return set;
    }
    for (SuffixTree aChild = child; aChild != null; aChild = aChild.sibling) {
      for (Iterator i = aChild.occurrences().iterator(); i.hasNext(); ) {
	tmp = (StringReference) i.next();
	//tmp.equals(tmp);
	set.add(tmp);
      }
    }
    return set;
  }
  
  public int count() {
    return (occurrences()).size();
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
    if ( i == referString.length() ) {
      return '\255';
    }
    return referString.charAt(i);
  }
  
  private char charAt(String str, int i){
    if( i == str.length() ){
      return '\255';
    }
    return str.charAt(i);
  }
  
  public String representingString() {
    if (isRoot()) {
      return "";
    }
    return referString.substring(origin, end);
  }
  
  
  public void removeEmptyLeaves() {
    if (isLeaf()) {
      return;
    }
    for ( ; (!isLeaf())&& firstChild().isEmptyLeaf() ; ) {
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
  
  
  private void insertSuffix(String str, int idx) {
    SuffixTree locus, prev, newNode;
    int skip, org;
    //System.out.println(this);
    locus = this;
    org = idx;
    while (true) {
      prev = locus.beforeEqualOrGreaterChild(str, idx);
      if (prev.sibling != null){
    	if (charAt(str,idx) != '\255' && prev.sibling.charAt(prev.sibling.start) != '\255' &&
    	    prev.sibling.charAt(prev.sibling.start) == charAt(str,idx)){
    	  skip = prev.sibling.extendVal(str, idx);
    	  if ((! prev.sibling.isLeaf()) && 
    	      (skip == (prev.sibling.end - prev.sibling.start)) ) {
    	    idx = idx + skip;
    	    locus = prev.sibling;
    	    continue;
    	  }
    	  newNode = prev.insertBreakNode(skip, str, org);
    	  // ????
    	  //newNode.beforeEqualOrGreaterChild(str,idx).addInsertSibling(str, idx + skip, org);
    	  newNode.beforeEqualOrGreaterChild(str,idx + skip).addInsertSibling(str, idx + skip, org);
    	  //
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
  
  public synchronized String toString(){
    StringBuffer strbuf = new StringBuffer();
    //String tmp;
    SuffixTree aChild;
    if (isLeaf()){
      return referString.substring(start, end)+":"+ String.valueOf(origin);
    }
    if (start == end){
        strbuf.append("a SuffixTree");
    }else{
        strbuf.append(referString.substring(start, end));
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
  

  public static void main(String args[]){
    SuffixTree tree;
    Vector strs = new Vector();
    strs.addElement(new String("abbabb"));
    strs.addElement(new String("ababba"));
    tree = new SuffixTree(strs);
    System.out.println(tree);
  }
  
}

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
    
  private SuffixTree nextOfMostLowestNonLastAncestor(SuffixTree node) {
    SuffixTree trialNode;
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
    
    public boolean hasMoreNodes() {
        return (next != null);
    }
    
    public boolean hasMoreElements() {
        return hasMoreNodes();
    }
    
    public SuffixTree current() {
        return current;
    }
    
  private SuffixTree next() {
    // current is always the result value;
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
	  /*
	    for ( ; next.isEmptyLeaf() && (!next.isLastSibling()) ; next = next.sibling) ;
	    if ( !next.isLastSibling() ) {
            next = next.sibling;
            return current;
	    }
	    */
	
	if ( ! next.isLastSibling() ) {
	  // find the next non empty leaf, or the last (possibly empty) leaf. 
	  for ( ; ! next.isLastSibling() ; ) {
	    next = next.sibling;
	    if ( ! next.isEmptyLeaf() ) {
	      break;
	    }
	  }
	  if ( !next.isEmptyLeaf() ) {
	    return current;
	  }
	}
	// if next was the last one, or there is no succeeding non empty leaf. 
        next = nextOfMostLowestNonLastAncestor(next);
        // if (next == null) then no more nodes.
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
        next = nextOfMostLowestNonLastAncestor(next);
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
