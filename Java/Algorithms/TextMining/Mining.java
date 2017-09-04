import java.util.*;
import java.io.*;
import SuffixTree;
import Occurrence;

class AssociationPattern {
	String s1, s2;
	int gap;
	
	public AssociationPattern(String s, String t, int g) {
		s1 = s;
		s2 = t;
		gap = g;
	}
	
	public synchronized String toString() {
	    return "\""+s1+"\".("+gap+").\""+s2+"\" ";
	}
}

class Mining {
    String texts[];
	
	public Mining(String strs[], double confidence, double support, int gapsize) {
	    Vector patterns;
	    
	    texts = strs;
		SuffixTree tree = new SuffixTree(texts);
		System.out.println("Finished the Construction of SuffixTree");
		patterns = makeList(tree, confidence, support, gapsize);
		//System.out.println(strs);
		//System.out.println(tree);
	}

	public int compare(Vector strs, String t1, String t2) {
		int counter = 0;
		int tmpInt1, tmpInt2, i;
		
		for (i = 0; i < strs.size(); i++) {
			tmpInt1 = ((String) strs.elementAt(i)).indexOf(t1);
			tmpInt2 = ((String) strs.elementAt(i)).indexOf(t2, tmpInt1+t1.length());
			if ((tmpInt1 != -1 && tmpInt2 != -1) ) {
			    counter++;
			    //System.out.print("Hit: "+tmpInt1+", "+tmpInt2+" in "+strs.elementAt(i)+"  ");
			}
		}
		return counter;
	}

	public int countOccurrences(AssociationPattern pat, SortedSet set1, SortedSet set2) {
		int counter = 0;
		int id, lastid = -1, pos, posmax;
		Occurrence occ;
		
		if (pat.s1.length() == 0 || pat.s2.length() == 0) {
		    return 0;
		} // ignores patterns with empty string
        for (Iterator i = set1.iterator(); i.hasNext(); ) {
            occ = (Occurrence) i.next();
            id = occ.identifier;
		    pos = occ.position;
            if (id == lastid) {
                continue;
            }
		    pos = pos + pat.s1.length();
		    //posmax = Math.max(texts[id].length(), pos + pat.gap);
		    posmax = pos + pat.gap;
		    if (! set2.subSet(new Occurrence(id, pos), new Occurrence(id, posmax)).isEmpty()) {
		        counter++;
		        lastid = id;
		    }
		}
		return counter;
	}

	private Vector makeList(SuffixTree tree, double confidence, double support, int gp) {
		SuffixTreeEnumerator enum1, enum2;
		HashSet hits1, hits2;
		int match;  // A ... B
		AssociationPattern pat;
		Vector results = new Vector();
		
		for ( enum1 = tree.enumeration(); enum1.hasMoreNodes(); ) {
		    enum1.nextElement();
			hits1 = enum1.hits();
			//System.out.println(hits1);
			if ( hits1.size() < texts.length * support ) {
				continue;
			}
			
			for ( enum2 = tree.enumeration(); enum2.hasMoreNodes(); ) {
				enum2.nextElement();
				hits2 = enum2.hits();
				if ( hits2.size() < texts.length * support ) {
					continue;
				}
				pat = new AssociationPattern(enum1.pathLabel(), enum2.pathLabel(), gp);
				match = countOccurrences(pat, enum1.occurrences(), enum2.occurrences());
				       // compare(strs, enum1.current().representingString(), 
				       //         enum2.current().representingString());
				
				if ( (match < hits1.size() * confidence) 
				    || (match < texts.length * support) ) {
					//enum2.skipChildren();
					continue;
				}
				//System.out.println(enum2.occurrences());
				results.addElement(pat);
				System.out.println(pat +" \t"+ (100*match/hits1.size()) + " \t" + (100*match/texts.length));
					  //  +" \t pattern matched "+ match + ", pre matched " + node1.count() + ", & rule holds " + match + " among " + strs.size());
			}
		}
		return results;
	}

	public static void main(String[] args) {
		double confidence, support;
		int gapsize;
		Vector tmptexts = new Vector();
		String str, texts[];
		BufferedReader reader = null;
		long t;
		//DataInputStream inStream = new DataInputStream(System.in);

		if (args.length == 4) {
			try {
				System.out.println("Trying to read the file "+args[3]);
				reader = new BufferedReader(new FileReader(args[3]));
			} catch(IOException e) {
				System.out.println("File does not found: "+ e.toString());
			}
		}
		if (reader == null) {
			reader = new BufferedReader(new InputStreamReader(System.in));
		}    
		
		confidence = (double)Integer.parseInt(args[0]) / 100;
		support = (double)Integer.parseInt(args[1]) / 100;
		gapsize = (int)Integer.parseInt(args[2]);
		System.out.println("Condidence = " + confidence*100 + " %");
		System.out.println("Support = " + support*100 + " %"); 
		System.out.println("Max gap size = " + gapsize); 

		try {
			System.out.print("Reading strings");
			while ((str = reader.readLine()) != null) {
				tmptexts.addElement(str);
				System.out.print(".");
			}
			System.out.println();
		} catch(IOException e) {
			System.out.println("Regarded Error: \n"+ e.toString());
			System.exit(0);
		}
		texts = new String[tmptexts.size()];
		for (int i = 0; i < tmptexts.size(); i++) {
		    texts[i] = (String) tmptexts.elementAt(i);
		}
		t = System.currentTimeMillis();
		Mining m = new Mining(texts, confidence, support, gapsize);
		System.out.println("Completed in "+(System.currentTimeMillis() - t)+ " msec.\n");
		/*
		try {
		    (new BufferedReader(new InputStreamReader(System.in))).readLine();
		} catch(IOException e) {
		    
		}
		*/
	}
}
