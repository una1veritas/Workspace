import java.io.*;
import java.net.*;

class URLDemo2 {
  public static void main(String args[]) {
    try {
	//      URL url = new URL("http://search.yahoo.com/bin/search?p=" + args[0]);
	//      URL url = new URL("http://www.google.co.jp/search?p=" + args[0]);
      URL url = new URL("http://search.yahoo.co.jp/bin/search?p=" + args[0]);
      InputStream in = url.openStream();
      InputStreamReader sr = new InputStreamReader(in);
      BufferedReader br = new BufferedReader(sr);

      String s;
      while ( (s = br.readLine()) != null )
        System.out.println(s);
      
      sr.close();
    } catch (Exception e) {
      System.out.println("Exception " + e + " occurred.");
    }
  }
}
