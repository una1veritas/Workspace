/*
 * Created on 2004/11/03
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

import java.net.*;
import java.io.*;

public class TravelingSalesmanProblem {
	Permutation tour;
	
	public TravelingSalesmanProblem(String urlstr) {
		tour = new Permutation(sz);
        URL url = new URL( urlstr );
        Object content = url.getContent();
        if( content instanceof InputStream ) {
        	BufferedReader reader = 
        		new BufferedReader(new InputStreamReader( (InputStream)content ) );
           	String line;
            while( ( line = reader.readLine() ) != null )
                System.out.println( line );
            reader.close();
        } else
            System.out.println( "Content is " + content.toString() );
        }
        catch( ArrayIndexOutOfBoundsException e ){
            System.err.println( "Usage: java URLContent urlname" );
            System.exit(-1);
        }
        catch( IOException e ){
            System.err.println( "IO Error" );
            System.exit(-1);
        }
	}

	public static void main(String[] args) {
	}
}
