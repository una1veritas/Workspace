import java.io.*;

class StandardInputDemo {
    public static void main(String args[]) throws Exception {
	InputStreamReader stream = new InputStreamReader(System.in);
	BufferedReader buffered = new BufferedReader(stream);
	
	String s;
	while ((s = buffered.readLine()) != null) {
	    System.out.println("Got: " + s);
	}
    }
}
