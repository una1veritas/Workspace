import java.io.*;

class FileInputDemo {
    public static void main(String args[]) {
	try {
	    FileReader reader = new FileReader(args[0]);
	    BufferedReader buffered = new BufferedReader(reader);
	    
	    String s;
	    while ((s = buffered.readLine()) != null) {
		System.out.println("> " + s);
	    }
	    reader.close();
	} catch (FileNotFoundException e) {
	    System.out.println("Correct File Name. " + args[0] + " could not be found.");
	} 
    }
}
