import java.io.*;

class PrintWriterDemo {
    public static void main(String args[]) {
	try {
	    FileOutputStream os = new FileOutputStream(args[0]);
	    PrintWriter pw = new PrintWriter(os);
	    
	    pw.println(true);
	    pw.println("There are " + 18 + " pencils.");
	    pw.println();
	    pw.println(new Double(40.22));
	    pw.println('a');

	    pw.close();
	} catch (FileNotFoundException e) {
	    System.err.println("File not found: " + e);
	} catch (IOException e) {
	    System.err.println("Exception: " + e);
	}
    }
}
