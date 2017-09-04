/*
 *  FS.java
 *
 *  FS interface is a Remote interface that
 *  gets implemented by the remote object, in
 *  this case the FSServer.
 */

// package samples.rmi1;

import java.rmi.*;
import java.util.Date;

public interface FS extends Remote
{
    String printHello() throws RemoteException;
    Date getTime() throws RemoteException;
}