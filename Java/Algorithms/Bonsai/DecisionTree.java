import java.lang.*;
import java.util.*;

/*
                query(yes -> leftchild, no -> rightchild)
*/

public class DecisionTree {
    private DecisionTree child, parent, sibling;
    private String label;

    private static double infValue(double p, double n) {
        //#define i_p_n(p,n) (( p + n ) ? ( -1.0 / (p + n) * ((p ? p * log2(p/(p+n))
        //: 0.0) + (n ? n * log2(n/(p+n)) : 0.0))) : 0.0 )
        return (( p + n > 0) ?
            ( -1.0 / (p + n) * ((p > 0 ? p * Math.log((double)p/(p+n)) : 0.0) + 
                                (n > 0 ? n * Math.log((double)n/(p+n)) : 0.0)))
            : 0.0 );
    }

    public DecisionTree() {
        label = null;
        parent = null;
        child = null;
        sibling = null;
        //pmm = null;
    		//{{INIT_CONTROLS
		//}}
}

    public DecisionTree(String myLabel, DecisionTree left, DecisionTree right) {
        this();
        label = myLabel;
        child = left;
        child.sibling = right;
    }

    public DecisionTree(String myLabel) {
        this();
        label = myLabel;
    }

    public boolean isLeaf() {
        return child == null;
    }

    String decide(String str) {
        if (isLeaf()) {
            return label;
        }
        /*if (pmm == null) {
            pmm = new KnuthMorrisPratt(label);
        }*/
        //if ( pmm.findIndex(str) >= 0) {
        if ( str.indexOf(label) >= 0) {
            return child.decide(str);
        } else {
            return child.sibling.decide(str);
        }
    }
    
    public String toString() {
        StringBuffer tmp;
        if (isLeaf()) {
            return label;
        }
        tmp = new StringBuffer(label);
        tmp.append('(');
        tmp.append("yes:");
        tmp.append(child.toString());
        tmp.append(", ");
        tmp.append("no :");
        tmp.append(child.sibling.toString());
        tmp.append(')');
        return tmp.toString();
    }

    public DecisionTree(Vector pos, String posLabel, Vector neg, String negLabel) {
        this();
        String pattern;

        if (pos.size() == 0) {
            label = negLabel;
            return;
        }
        if (neg.size() == 0) {
            label = posLabel;
            return;
        }
        Vector posm = new Vector(), posu = new Vector(),
               negm = new Vector(), negu = new Vector();
        pattern = findPattern(pos,neg);
        if (pattern != null) {
            partition(pattern,pos,neg,posm,negm,posu,negu);
            label = pattern;
            child = new DecisionTree(posm,posLabel,negm,negLabel);
            child.sibling = new DecisionTree(posu,posLabel,negu,negLabel);
            return;
        } else {
            /* Can not find patterns any more. */
            label = negLabel;
            return;
        }
    }
    
    String findPattern(Vector pos, Vector neg) {
        double gain = 0, bestGain = 0;
        String pattern, bestPattern = null;
        String data, exstr;
        int pmc, puc,nmc, nuc;
        Vector examples; 
        Enumeration enu;
        SuffixTree sftree;
        
        examples = new Vector(pos.size() + neg.size());
        for (enu = pos.elements(); enu.hasMoreElements(); ) {
            examples.addElement(enu.nextElement());
        }
        for (enu = neg.elements(); enu.hasMoreElements(); ) {
            examples.addElement(enu.nextElement());
        }
        sftree = new SuffixTree(examples);
        //sftree.removeEmptyLeaves();

        for (enu = sftree.enumeration(); enu.hasMoreElements(); ) {
            pattern = ((SuffixTree) enu.nextElement()).representingString();
            pmc = 0; puc = 0;
            for (Enumeration ex = pos.elements(); ex.hasMoreElements(); ) {
                exstr = (String) ex.nextElement();
                if (exstr.indexOf(pattern) >= 0) {
                  pmc++;
                } else {
                  puc++;
                }
            }
            nmc = 0; nuc = 0;
            for (Enumeration ex = neg.elements(); ex.hasMoreElements(); ) {
                exstr = (String) ex.nextElement();
                if (exstr.indexOf(pattern) >= 0) {
                  nmc++;
                } else {
                  nuc++;
                }
            }
            if ((pmc > 0 || nmc > 0) && (puc > 0 || nuc > 0)) {
                gain = ((pmc+nmc)*infValue(pmc,nmc)
                        + (puc+nuc)*infValue(puc,nuc))
                        /(pmc+puc+nmc+nuc) + 1.0f;
                if ((bestPattern == null) || gain < bestGain ||
                    (gain == bestGain && pattern.length() < bestPattern.length())) {
                    bestGain = gain;
                    bestPattern = pattern;
                }
            }
        }
        return bestPattern;
    }

    void partition(String str, Vector pos, Vector neg,
                    Vector posm, Vector negm, Vector posu, Vector negu) {
        String data;
        for (Enumeration e = pos.elements(); e.hasMoreElements(); ) {
            data = (String) e.nextElement();
            if (data.indexOf(str) >= 0) {
                posm.addElement(data);
            } else {
                posu.addElement(data);
            }
        }
        for (Enumeration e = neg.elements(); e.hasMoreElements(); ) {
            data = (String) e.nextElement();
            if (data.indexOf(str) >= 0) {
                negm.addElement(data);
            } else {
                negu.addElement(data);
            }
        }
    }

    public int size() {
        if (isLeaf()) {
            return 1;
        } else {
            return 1 + child.size() + child.sibling.size();
        }
    }
    
    private Vector structureString(Vector strs, int level) {
        StringBuffer buf;
        buf = new StringBuffer();
        for (int i = 0; i < level; i++) {
            buf.append(" ");
        }
        buf.append(label);
        strs.addElement(buf.toString());
        if (isLeaf()) {
            return strs;
        }
        child.structureString(strs, level+1);
        child.sibling.structureString(strs, level+1);
        return strs;
    }
    
    public String[] structureString() {
        Vector tmpStrs = new Vector();
        structureString(tmpStrs, 0);
        String[] tmp = new String[tmpStrs.size()];
        for (int i = 0; i < tmpStrs.size(); i++) {
            tmp[i] = (String) tmpStrs.elementAt(i);
        }
        return tmp;
    }
    
    public static void main(String args[]) {
        Vector pos = new Vector();
        Vector neg = new Vector();
        DecisionTree test;
        pos.addElement("aabaaba");
        pos.addElement("abbababa");
        pos.addElement("abaabba");
        neg.addElement("bbaababa");
        neg.addElement("babbaba");
        neg.addElement("bbabab");
        test = new DecisionTree(pos, "+", neg, "-");
        System.out.println(test);
    }
    

    private class SuffixTree {
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
		    //System.gc();
	    }
	    
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
        
        /*
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
        */
        
    }


    public class SuffixTreeEnumerator implements Enumeration {
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
            while ( next.isEmptyLeaf() && (!next.isLastSibling()) ) {
                next = next.sibling ;
            }
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
    }
}
