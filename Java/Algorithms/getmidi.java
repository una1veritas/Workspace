//
import java.io.*;
import java.util.*;
import java.net.*;

class getmidi {
  public static void main(String args[]) 
  throws Exception {
    
    DataInputStream input;
    FileOutputStream fout = new FileOutputStream(args[1]);
    byte c;
    int i, n;
    URL u;
    
    if (args.length <= 0){
      System.err.println("No, file given");
      System.exit(1);
    }
    
    u = new URL(args[0]);
    input = new DataInputStream(u.openStream());
    try {
      while (true) {
	  c = input.readByte();
	  fout.write(c);
      }
    } catch (EOFException e) {
	System.out.println("Done.\n");
    } catch(IOException e) {
      System.err.println("IO Error");
      System.exit(1);
    } finally {
	input.close();
	}
    fout.close();
  }
}

/*

this site was updated on the 26th of November.
We are very sorry that we couldn't update our
site since our teacher who takes care of this has been sick. 
she has to be in a hospital in December again. 
So we cannot change anything until she comes back.
Still enjoy the history of our
hometown. 

*/
