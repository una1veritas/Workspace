import java.io.*;
import java.net.URL;

class URLDemo {
    public static void main(String args[]) {
	try {
	    URL url = new URL(args[0]);
	    InputStream in = url.openStream();
	    InputStreamReader sr = new InputStreamReader(in);
	    BufferedReader br = new BufferedReader(sr);
	    
	    String s;
	    while ((s = br.readLine()) != null) {
		System.out.println(s);
	    }
	    br.close();
	} catch (IOException e) {
	    System.err.println("Exception: " + e);
	}
    }
}
