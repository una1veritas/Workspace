import java.util.*;
import java.io.*;
import SuffixTree;

class MiningInfo {
	String t1, t2;
	int confidence, match, support;
	
	public MiningInfo(String tmp1, String tmp2, int a_to_b, int co, int sup) {
		t1 = tmp1;
		t2 = tmp2;
		match = a_to_b;
		confidence = co;
		support = sup;
	}
			//{{INIT_CONTROLS
		//}}

	//{{DECLARE_CONTROLS
	//}}
}

class Mining {
	Vector answer = new Vector();  // list_of_MiningInfo
	
	public Mining(Vector strs, double confidence, double support) {
		SuffixTree tree = new SuffixTree(strs);
		//System.out.println(tree);
		System.out.println("Finish Making SuffixTree");
		makeList(strs, tree, confidence, support);
	}
			//{{INIT_CONTROLS
		//}}

	public int compare(Vector strs, String t1, String t2) {
		int counter = 0;
		int tmpInt1, tmpInt2, i;
		
		for (i = 0; i < strs.size(); i++) {
			tmpInt1 = ((String) strs.elementAt(i)).indexOf(t1);
			tmpInt2 = ((String) strs.elementAt(i)).lastIndexOf(t2);
			if ((tmpInt1 != -1 && tmpInt2 != -1) 
				&& (tmpInt2 > tmpInt1 + t1.length() - 1) ) {
					counter++;
			    }
		}
		return counter;
	}

	private void makeList(Vector strs, SuffixTree tree, double confidence, double support) {
		SuffixTreeEnumerator enum1, enum2;
		SuffixTree node1, node2;
		int match;  // A ... B

		for ( enum1 = tree.enumeration(); enum1.hasMoreNodes(); ) {
			node1 = enum1.next();
			if (node1.count() < strs.size() * support) {
				enum1.skipChildren();
				continue;
			}
			for ( enum2 = tree.enumeration(); enum2.hasMoreNodes(); ) {
				node2 = enum2.next();
				if (node2.count() < strs.size() * support) {
					enum2.skipChildren();
					continue;
				}
				match = compare(strs, node1.representingString(), node2.representingString());
				if ((match < node1.count() * confidence) || (match < strs.size() * support)) {
					enum2.skipChildren();
					continue;
				}
				if ((node1.representingString().length() != 0)&&(node2.representingString().length() != 0)) {
					answer.addElement(new MiningInfo(node1.representingString(), node2.representingString(), match , node1.count(), strs.size()));
					System.out.println("<" + node1.representingString() +"..."+ node2.representingString() + ">" +" \t"+ (100*match/node1.count()) + " \t" + (100*match/strs.size()));
				}
			}
		}
	}

	public String toString() {
		String tmp = new String("answer:\n");
		MiningInfo info;
		
		for (int i = 0; i < answer.size(); i++) {
			info = (MiningInfo)answer.elementAt(i);
			tmp = tmp + "(" + info.t1 +"  "+ info.t2 + ")" +"  "+ info.match +"/" + info.confidence + "  " + info.match + "/" + info.support + "\n";
		}
		return tmp;
	}

	public static void main(String[] args) {
		String str;
		double confidence, support;
		Vector strings = new Vector();
		BufferedReader reader = null;
		long t;
		//DataInputStream inStream = new DataInputStream(System.in);

		if (args.length == 3) {
			try {
				System.out.println("Trying for file "+args[2]);
				reader = new BufferedReader(new FileReader(args[2]));
			} catch(IOException e) {
				System.out.println("File does not found: "+ e.toString());
			}
		}
		if (reader == null) {
			reader = new BufferedReader(new InputStreamReader(System.in));
		}    
		
		confidence = (double)Integer.parseInt(args[0]) / 100;
		support = (double)Integer.parseInt(args[1]) / 100;
		System.out.println("Input condidence = " + args[0] + " %");
		System.out.println("Input support = " + args[1] + " %"); 

		try {
			System.out.print("Reading strings");
			while ((str = reader.readLine()) != null) {
				strings.addElement(str);
				System.out.print(".");
			}
			System.out.println();
		} catch(IOException e) {
			System.out.println("Regarded Error: \n"+ e.toString());
		}
		t = System.currentTimeMillis();
		Mining m = new Mining(strings, confidence, support);
		System.out.println("Completed in "+(System.currentTimeMillis() - t)+ " msec.\n");
		try {
		    (new BufferedReader(new InputStreamReader(System.in))).readLine();
		} catch(IOException e) {
		    
		}
	}
	//{{DECLARE_CONTROLS
	//}}
}
