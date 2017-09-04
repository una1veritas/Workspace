
import java.math.BigInteger;
import java.util.*;

class test {
  
  public static void main(String argv[]) {
    HashMap m = new HashMap();
    double d = 2.0040302;
    int i = 9;

    m.put("apple", "Ringo.");
    System.out.println(m.keySet());
    System.out.println(m.get("apple"));

    System.out.println(new BigInteger("1"));
    System.out.println(BigInteger.ONE);
    System.out.println(BigInteger.ZERO);
    System.out.println((new BigInteger("1")).equals(BigInteger.ONE));
    System.out.println((new BigInteger("1")) == BigInteger.ONE);
    System.out.println(d);
    System.out.println((int) d);
    System.out.println(new BigInteger(32,new Random()));
    return;
  }
}
