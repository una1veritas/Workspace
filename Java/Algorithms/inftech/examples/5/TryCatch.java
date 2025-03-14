import java.io.*;

class TryCatch {
  public static void main(String args[]) {
    try {
      FileReader reader = new FileReader(args[0]);
      BufferedReader buffered = new BufferedReader(reader);
      
      String s;
      while ( (s = buffered.readLine()) != null )
        System.out.println(s);
      
      reader.close();
    } catch (Exception e) {
      System.out.println("Exception " + e + " occurred.");
    }
  }
}
