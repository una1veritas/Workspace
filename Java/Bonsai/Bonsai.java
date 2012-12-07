import java.util.*;
import java.io.*;
import AlphabetIndexing;
import DecisionTree;

public class Bonsai {
    private DecisionTree dtree;
    private AlphabetIndexing indexing;
    private Vector posex, negex, postr, negtr;

    private static String positiveLabel() {
        return "+";
    }
    
    private static String negativeLabel() {
        return "-";
    }
    
    private Bonsai(String indices) {
        indexing = new AlphabetIndexing(indices);
        dtree = null;
        posex = new Vector();
        negex = new Vector();
        postr = new Vector();
        negtr = new Vector();
    }
    
    public Bonsai(String indices, BufferedReader preader, BufferedReader nreader, double sampratio, int trsize) {
        this(indices);
        String str;
        Random rnd = new Random();

		try {
			//System.out.print("Reading examples... ");
			while ((str = preader.readLine()) != null) {
				if (rnd.nextDouble() > sampratio) {
					continue;
				}
	            indexing.addToDomain(str);
	            posex.addElement(str);
			}
			preader.close();
			for (int i = 0; i < posex.size() && postr.size() < trsize; i++) {
    		    if ( ((trsize - postr.size()) /(posex.size() - i)) > rnd.nextDouble() ) {
				    postr.addElement(posex.elementAt(i));
				}
			}
			//System.out.println("Got "+posex.size()+" examples and "+postr.size()+" training examples;");
			//System.out.print("Reading counter-examples... ");
			while ((str = nreader.readLine()) != null) {
	            if (rnd.nextDouble() > sampratio) {
					continue;
				}
				indexing.addToDomain(str);
	            negex.addElement(str);
			}
			nreader.close();
			for (int i = 0; i < negex.size() && negtr.size() < trsize; i++) {
    		    if ( ((trsize - negtr.size())/ (negex.size() - i)) > rnd.nextDouble() ) {
				    negtr.addElement(negex.elementAt(i));
				}
			}
			//System.out.println("Got "+negex.size()+" counter-examples and "+negtr.size()+" traning counter-examples.");
		} catch(IOException e) {
			System.out.println("Regarded Error: \n"+ e.toString());
		}
		indexing.randomize();
		//System.out.println("Finished making indexing. ");
		dtree = makeDecisionTree(indexing);
    }
    
    public boolean hasExamples() {
        return (posex != null) && (negex != null);
    }
        
    public DecisionTree makeDecisionTree(AlphabetIndexing ix) {
		Vector transedp = new Vector();
		Vector transedn = new Vector();
		for (Enumeration e = postr.elements(); e.hasMoreElements(); ) {
		    transedp.addElement(ix.translate((String)e.nextElement()));
		}
		for (Enumeration e = negtr.elements(); e.hasMoreElements(); ) {
		    transedn.addElement(ix.translate((String)e.nextElement()));
		}
		//System.out.println("examples:\n"+transedp.toString());
		//System.out.println("counter-examples: \n"+transedn.toString());
		return new DecisionTree(transedp,positiveLabel(),transedn,negativeLabel());
    }
    
    public double verifyDecisionTree() {
        return verifyDecisionTree(dtree, indexing);
    }
    
    public double verifyDecisionTree(DecisionTree tree, AlphabetIndexing ix) {
        String str;
        int p, n;
        double accr;
        char carray[];
        
		p = 0; 
		n = 0;
		for (Enumeration e = posex.elements(); e.hasMoreElements(); ) {
		    //
		    carray = ((String) e.nextElement()).toCharArray();
		    //if (positiveLabel().equals(tree.decide(ix.translate((String)e.nextElement())))) {
		    ix.translate(carray);
		    if (positiveLabel().equals(tree.decide(new String(carray)))) {
		        p++;
		    } else {
		        n++;
		    }
		}
		accr = ((double) p) / (p+n);
	    //System.out.print("(" + (p*100/(p+n)) + "%, ");
		p = 0; 
		n = 0;
		for (Enumeration e = negex.elements(); e.hasMoreElements(); ) {
		    //
		    carray = ((String) e.nextElement()).toCharArray();
		    //if (positiveLabel().equals(tree.decide(ix.translate((String)e.nextElement())))) {
		    ix.translate(carray);
		    if (negativeLabel().equals(tree.decide(new String(carray)))) {
		    //if (negativeLabel().equals(tree.decide(ix.translate((String)e.nextElement())))) {
		        n++;
		    } else {
		        p++;
		    }
		}
		accr = accr * (((double) n) / (p+n));
	    //System.out.print((n*100/(p+n)) + "%) ");
	    return accr;
    }

    public AlphabetIndexing findBetterNeighbor() {
        // first encountered better neighborhood implementation
        // with cyclic displacedment search structrure
        
        AlphabetIndexingNeighbor n;
        AlphabetIndexing neighbor;
        double currEval, neighborEval;
        DecisionTree tree;
        
        currEval = verifyDecisionTree();
        //System.out.println();
        for (n = new AlphabetIndexingNeighbor(indexing); n.hasNext(); ) {
            neighbor = n.next();
            tree = makeDecisionTree(neighbor);
            neighborEval = verifyDecisionTree(tree, neighbor);
            tree = null;
            //System.out.print(neighbor.translate(neighbor.domainString()) 
			//					+ " " + Double.toString(neighborEval).substring(0,4));
            if (neighborEval > currEval) {
                return neighbor;
            }
            //System.out.println();
        }
        return null;
    }
    
    public AlphabetIndexing indexing() {
        return indexing;
    }
    
    public AlphabetIndexing setIndexing(AlphabetIndexing ix) {
        indexing = ix;
        return ix;
    }
    
    public DecisionTree decisionTree() {
        return dtree;
    }
    
    public DecisionTree renewDecisionTree() {
        dtree = makeDecisionTree(indexing);
        return dtree;
    }
    
    public synchronized String toString() {
        StringBuffer tmp = new StringBuffer();
        tmp.append("Bonsai with ");
        tmp.append(posex.size());
        tmp.append(" examples and ");
        tmp.append(negex.size());
        tmp.append(" counter examples, ");
        tmp.append(postr.size());
        tmp.append(" training examples and counter examples.");
        return tmp.toString();
    }
    
    public static void main(String args[]) {
        BufferedReader pr, nr;
        Bonsai b;
        AlphabetIndexing neighbor;
        DecisionTree tree;
        double currEval, neighborEval;
        
        if (args.length != 2) {
            System.out.println("Not expected number of args.");
            return;
        }
        try {
            pr = new BufferedReader(new FileReader(args[0]));
            nr = new BufferedReader(new FileReader(args[1]));
        } catch(IOException ex) {
            System.out.println("Error :" + ex);
            return;
        }
        b = new Bonsai("12", pr, nr, 0.5f, 12);
        System.out.println(b.indexing);
        System.out.println(b.dtree);

        while (true) {
            neighbor = b.findBetterNeighbor();
            if (neighbor == null) {
                break;
            }
            b.indexing = neighbor;
            b.dtree = b.makeDecisionTree(b.indexing);
            System.out.println(" updated: "+b.dtree);
        }
        System.out.println("Result: \n"+b.indexing.domainString()+"\n"+b.indexing.translate(b.indexing.domainString()));
        System.out.println(b.dtree+" with accuracy "+b.verifyDecisionTree());
        System.out.println("finished all.");
		try {
		    (new BufferedReader(new InputStreamReader(System.in))).readLine();
		} catch(IOException e) {
		    
		}
    }
}
