/*
 *  Hello7Client.java
 *
 *  FSProver uses RMI for methods in Hello7 class.
 */

// package samples.rmi1;

import java.rmi.*;

public class FSProver
{
    public static void main( String args[] )
    {
        String hostName = "localhost:21";

        if( args.length > 0 )
            hostName = args[0];

        System.setSecurityManager( new RMISecurityManager() );

        try
        {
            FS remote = (FS) Naming.lookup( "rmi://" + hostName + "/FSVerifier" );
            System.out.println( remote.printHello() + "    It's now " + (remote.getTime()).toString() + " at the remote object." );
        }
        catch( Exception e )
        {
            System.out.println( e.toString() );
        }
    }
}