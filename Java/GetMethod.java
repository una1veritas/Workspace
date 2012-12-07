//
import java.io.*;
import java.util.*;
import java.net.*;

class GetMethod {

    public static void main(String args[]) 
	throws Exception {
	
	StringBuffer sb = new StringBuffer(1024);
	DataInputStream in;
	char c;
	URL url;
	String query;
	
	if (args.length != 2){
	    System.err.println("Supply URL and string for messg.");
	    System.exit(1);
	}
	
	query = new String("?messg=\"") + URLEncoder.encode(args[1], new String("UTF-8"));
	System.out.println(query);
	url = new URL(args[0]+query);

	in = new DataInputStream(url.openStream());
	try {
	    while (true) {
		c = (char) in.readByte();
		sb.append(c);
	    }
	} catch (EOFException e) {
	    System.out.println("Done.\n");
	    in.close();
	} catch(IOException e) {
	    System.err.println("IO Error");
	    System.exit(1);
	} 
	
	System.out.println("The answer is: \n" + sb);
	
    }
    
}
