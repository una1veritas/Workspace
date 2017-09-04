import SuffixTree;
import java.util.*;

class Test {
  
  public static void main(String args[]) {
    SuffixTree sf;

    if (args.length == 0) {
      System.exit(0);
    }

    sf = new SuffixTree(args);
    /*
    for (SuffixTreeEnumerator e = sf.enumeration();
	 e.hasMoreElements(); ) {
	    node = (SuffixTree.Node)e.nextElement();
      System.out.println(node.representingString()+" "+node.occurrences()+" ");
    }
    */
    System.out.println();
    //System.out.println(sf + "\n");
    
    for ( SuffixTreeEnumerator e = sf.enumeration(); 
        e.hasMoreElements() ; ) {
        e.nextElement();
        System.out.println(e.pathLabel() + " " 
                            + e.occurrences());
    }
  }
    
}
