import SuffixTree;
import java.util.*;

class Test {
  
  public static void main(String args[]) {
    SuffixTree sf;

    if (args.length == 0) {
      System.exit(0);
    }

    sf = new SuffixTree(args);
    
    System.out.println(sf + "\n");

    for (SuffixTreeEnumerator e = sf.enumeration();
	 e.hasMoreElements(); ) {
      System.out.println(e.nextElement());
    }
    System.out.println();

  }
    
}
