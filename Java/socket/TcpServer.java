// TcpServer.java
import java.net.*;
import java.io.*;

class TcpServer {
    public static void main(String args[]) {
        ServerSocket server = null;
        Socket soc;
        InputStream in;
        int c;

        try {
            server = new ServerSocket(5234,31);
            while (true) {
                soc = server.accept();
                in = soc.getInputStream();
                while ( (c = in.read()) != -1) {
                    System.out.print((char)c);
                }
                System.out.println();
                in.close();
                server.close();
            }
        } catch (IOException x) {
            System.err.println("Caught an IOException: "+ x);
            System.exit(1);
        }
    }
}

