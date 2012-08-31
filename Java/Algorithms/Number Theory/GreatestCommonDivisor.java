import java.lang.*;
import java.util.Random;


class GreatestCommonDivisor {

  public static long euclidean(long n, long m) {
    long r;
    while ((r = n % m) !=0 ) {
	System.out.println(n + ", " + m);
      n = m;
      m = r;
    }
    return m;
  }

	public static long naive(long n, long m) {
		long r;
		r = Math.min(n, m);
		while (!((n % r) == 0 && (m % r) == 0) ) {
			//System.out.println("" + r + ", \r");
			r--;
		}
		return r;
	}
	
  public static void main(String args[]) {
    long n1, n2, ans, in;
    Random rnd = new Random();

    if (args.length == 2) {
      n1 = Integer.parseInt(args[0]);
      n2 = Integer.parseInt(args[1]);
      System.out.println("Look for " + n1 + " and " + n2 + ".");
    } else {
      System.out.println("Seems too many or few input parameters. Bye.");
      return;
    }

    //for (long i = n1; i < n2; i++) {
    in = System.currentTimeMillis();
    //  for (long j = i+1; j <= n2; j++) {
    ans = euclidean(n1,n2);
    //  }
    System.out.println("GCD for "+n1+" and "+n2+" is "+ans+". Found in time "
	+(System.currentTimeMillis() - in)+" milli secs.");
    //}
	  in = System.currentTimeMillis();
	  //  for (long j = i+1; j <= n2; j++) {
	  ans = naive(n1,n2);
	  //  }
	  System.out.println("Naive alg. for "+n1+" and "+n2+" is "+ans+". Found in time "
						 +(System.currentTimeMillis() - in)+" milli secs.");
	  
  }
}
