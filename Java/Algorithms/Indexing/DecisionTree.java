import java.lang.*;
import java.util.*;

/*
                aaaa
                 /\
             no /  \ yes           aaaa(yes[,],bbb(yes[,],no[,]))
               /    \
              1     bbb
                     /\
                 no /  \ yes
                   /    \
                  1      0
*/

public class DecisionTree {
    DecisionTree child, parent, sibling;
    private String label;
    private KnuthMorrisPratt pmm;

    private static double log2(double d) {
        //#define log2(n) (log(n)/log(2.))
        return Math.log(d)/Math.log(2.0f);
    }

    private static double infValue(double p, double n) {
        //#define i_p_n(p,n) (( p + n ) ? ( -1.0 / (p + n) * ((p ? p * log2(p/(p+n))
        //: 0.0) + (n ? n * log2(n/(p+n)) : 0.0))) : 0.0 )
        return (( p + n > 0) ?
            ( -1.0 / (p + n) * ((p > 0 ? p * log2(p/(p+n)) : 0.0) + 
                                (n > 0 ? n * log2(n/(p+n)) : 0.0)))
            : 0.0 );
    }

    public DecisionTree() {
        label = null;
        parent = null;
        child = null;
        sibling = null;
        pmm = null;
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
        if (pmm == null) {
            pmm = new KnuthMorrisPratt(label);
        }
        if ( pmm.findIndex(str) >= 0) {
        //if ( str.indexOf(label) >= 0) {
            return child.sibling.decide(str);
        } else {
            return child.decide(str);
        }
    }
    
    public String toString() {
        StringBuffer tmp;
        if (isLeaf()) {
            return label;
        }
        tmp = new StringBuffer(label);
        tmp.append('(');
        tmp.append("n:");
        tmp.append(child.toString());
        tmp.append(", ");
        tmp.append("y:");
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
            child = new DecisionTree(posu,posLabel,negu,negLabel);
            child.sibling = new DecisionTree(posm,posLabel,negm,negLabel);
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
        child.sibling.structureString(strs, level+1);
        child.structureString(strs, level+1);
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
}
