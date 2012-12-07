// TcpServer.java
import java.net.*;
import java.io.*;
import java.math.*;

class TcpServer {
    public static void main(String args[]) {
        ServerSocket server;
        Socket socket;
        BufferedReader reader;
        String aLine;

        try {
            server = new ServerSocket(5234);
            socket = server.accept();
            reader = new BufferedReader(new InputStreamReader(socket.getInputStream()));
        
            while ( (aLine = reader.readLine()) != null) {
                System.out.println(aLine);
                try {
                    System.out.println(new BigInteger(aLine));
                } catch (NumberFormatException x) {
                    System.out.println(new BigInteger(aLine.getBytes()));
                }
            }
            
            reader.close();
            server.close();
        } catch (IOException x) {
            System.err.println("Caught an IOException: "+ x);
            System.exit(1);
        }
        
        //Wait for return key typed
        try {
		    (new BufferedReader(new InputStreamReader(System.in))).readLine();
		} catch(IOException x) { }

    }
}

