/*
 * Created on 2004/12/18
 *
 * TODO To change the template for this generated file go to
 * Window - Preferences - Java - Code Style - Code Templates
 */

/**
 * @author sin
 *
 * TODO To change the template for this generated type comment go to
 * Window - Preferences - Java - Code Style - Code Templates
 */

import java.net.URL;
import java.io.*;

public class URLReader {
	
	public static void main(String argv[]) throws IOException {
		URL conn = new URL(argv[0]);
		BufferedReader reader 
			= new BufferedReader(new InputStreamReader(conn.openStream()));
		StreamTokenizer st = new StreamTokenizer(reader);
		int val;
		
		while (StreamTokenizer.TT_EOF != (val = st.nextToken())) {
			if ( val == StreamTokenizer.TT_WORD)
				System.out.println(st.sval);
			else if ( val == StreamTokenizer.TT_NUMBER)
				System.out.println(st.nval);
		}
		
		reader.close();

	}
}
