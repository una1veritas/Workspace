import java.util.*;
import java.io.*;
import SuffixTree;
import Occurrence;

class Words {
  
  public static void main(String[] args) {
    String texts[];
    SuffixTree stree;
    long timewatch;

    texts = readTexts(args);

    timewatch = System.currentTimeMillis();

    stree = new SuffixTree(texts, " \n\t\r,.:;)(\"\'`-+=\\/!?&%$#@0123456789{}[]~<>");

    System.out.println("Construction of SuffixTree completed in " 
		       + ((System.currentTimeMillis() - timewatch)/1000) 
		       + " secs.");

    if(stree.findLocus("exchange") != null) {
      System.out.println(stree.findLocus("exchange").occurrences() +"\n");
      Set indices = stree.findLocus("exchange").hits();
      indices.retainAll(stree.findLocus("money").hits());
      System.out.println("exchange ... money occurs in " + indices + ". ");
    }


    for (SuffixTreeEnumerator w1 = stree.enumeration(); w1.hasNext(); ) {
      w1.nextElement();
      if (w1.pathLabel().length() < 3 || w1.pathLabel().indexOf(" ") != -1) 
	continue;
      
      Set w1hits, w2hits; 

      for (SuffixTreeEnumerator w2 = stree.enumeration(); w2.hasNext(); ) {
	w2.nextElement();
	if (w2.pathLabel().length() < 3 || w2.pathLabel().indexOf(" ") != -1) 
	  continue;

	w1hits = w1.hits();
	w2hits = w2.hits();
	w1hits.retainAll(w2hits);
	if (w1hits.size() > 1) {
	  System.out.println("\""+w1.pathLabel()+"\" ... \"" + w2.pathLabel() 
			     + "\" \t" + w1hits.size());
	}
      }

    }

    System.out.println("Completed in "+(System.currentTimeMillis() - timewatch)+ " msec.\n");
  }


  static String [] readTexts(String args[]) {
    Vector tmptexts = new Vector();
    String str, texts[];
    BufferedReader reader = null;

    if (args.length == 1) {
      try {
	System.out.println("Trying to read the file "+args[0]);
	reader = new BufferedReader(new FileReader(args[0]));
      } catch(IOException e) {
	System.out.println("File does not found: "+ e.toString());
      }
    }
    if (reader == null) {
      reader = new BufferedReader(new InputStreamReader(System.in));
    }    
    
    try {
      System.out.print("Reading strings");
      while ((str = reader.readLine()) != null) {
	tmptexts.addElement(str.toLowerCase());
	System.out.print(".");
      }
      System.out.println();
    } catch(IOException e) {
      System.out.println("Error: \n"+ e.toString());
      System.exit(0);
    }
    texts = new String[tmptexts.size()];
    tmptexts.toArray((Object[]) texts);
    return texts;
  }
}
