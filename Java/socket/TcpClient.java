// TcpServer.java
import java.net.*;
import java.io.*;


class TcpClient {
    public static void main(String args[]) {
        int c;
        Socket soc;
        String hname;
        String str ="Hello Net World!!\n";
        int slength;
        OutputStream out;


        if (args.length != 1) {
            hname = "localhost";
        } else {
            hname = args[0];
        }

        try {
            soc = new Socket(hname,5234);
            out = soc.getOutputStream();
            while ((c = System.in.read()) != -1) {
                out.write(c);
            }
            out.close();
        } catch (IOException x) {
            System.err.println("Caught IOException: "+x);
            System.exit(1);
        }
    }
}