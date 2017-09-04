//
import java.io.*;
import java.util.*;
import java.net.*;

class tango{
  public static void main(String args[]) 
  throws Exception {
    
    StringBuffer sb = new StringBuffer(1024);
    DataInputStream dis;
    char c;
    URL u;
    
    if (args.length <= 0){
      System.err.println("No, file given");
      System.exit(1);
    }
    
    u = new URL(args[0]);
    dis = new DataInputStream(u.openStream());
    try {
      while (true) {
	c = (char)dis.readByte();
	sb.append(c);
      }
    } catch (EOFException e) {
	//System.out.println("Done.\n");
    } catch(IOException e) {
      System.err.println("IO Error");
      System.exit(1);
    } finally {
      dis.close();
    }

    /*    
    String s2 = sb.toString();
    String s3 = " \" ";
    StringTokenizer st = new StringTokenizer(s2,s3);
    LinkedList list = new LinkedList();
    //list.add(st.nextToken());
    
    while (st.hasMoreTokens()){
      String s1 = st.nextToken();
      if(! list.contains(s1)){
	list.add(s1);
      }
      
    }
    for(Iterator i = list.iterator();i.hasNext();){
      System.out.println(i.next());
    }
    */
    System.out.println(sb);
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
