import java.lang.*;
import java.math.*;
import java.util.*;

class test {
  public static void main(String[] argv) {
    
    BigInteger bz = BigInteger.ONE;
    BigInteger a,p;
    Random rnd = new Random(11);
    int i;
    for (i = 0; i < 3; i++) {
      BigInteger bx = new BigInteger(6,rnd);
      System.out.println("Orginal:\t"+bx + "\tNegation:\t" + bx.negate());
    }
    a = new BigInteger("20");
    p = new BigInteger("41");
    System.out.println("Result: "+a.modPow(p.subtract(BigInteger.ONE).divide(new BigInteger("2")),p));

  }
}
