// TcpServer.java
import java.net.*;
import java.io.*;


class TcpClient {
    public static void main(String args[]) {
        Socket soc;
        String hname;
        BufferedWriter writer;
        BufferedReader reader;
        String line;

        if (args.length != 1) {
            hname = "localhost";
        } else {
            hname = args[0];
        }

        try {
            soc = new Socket(hname,5234);
            writer = new BufferedWriter(new OutputStreamWriter(soc.getOutputStream()));
            reader = new BufferedReader(new InputStreamReader(System.in));
            while ((line = reader.readLine()) != null) {
                //System.out.println(line);
                writer.write(line);
                writer.newLine();
                writer.flush();
            }
            writer.close();
            soc.close();
        } catch (IOException x) {
            System.err.println("Caught IOException: "+x);
            System.exit(1);
        }
    }
}