/*
 *  Hello7Server.java
 *
 *  class Hello7Server implements the remote interface
 *  Hello7 to identify it as a remote object.
 *  Methods printHello() and getTime() are remote methods
 *  for clients to call.
 */

// package samples.rmi1;

import java.rmi.*;
import java.rmi.registry.*;
import java.rmi.server.UnicastRemoteObject;
import java.util.Date;

public class FSVerifier extends UnicastRemoteObject implements FS
{
    public FSVerifier() throws RemoteException
    {
        super();
    }

    public String printHello() throws RemoteException
    {
        return( "Howdy!" );
    }

    public Date getTime() throws RemoteException
    {
        return( new Date( System.currentTimeMillis() ) );
    }

    public static void main( String args[] )
    {
        try
        {
            System.setSecurityManager( new RMISecurityManager() );

            FSVerifier server = new FSVerifier();
            Naming.rebind( "rmi://" + "localhost:21" + "/FSVerifier", server );
            System.out.println( "FSVerifier :  bound in registry" );
        }
        catch( Exception e )
        {
            System.out.println( e.toString() );
        }
    }
}