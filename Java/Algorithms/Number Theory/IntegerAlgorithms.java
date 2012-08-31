//
//  IntegerAlgorithms.java
//  IntegerAlgorithms
//
//  Created by 下薗 真一 on 07/11/05.
//  Copyright 2007 __MyCompanyName__. All rights reserved.
//

public class IntegerAlgorithms {

	public static long euclidean(long n, long m) {
		long r;
		while ((r = n % m) !=0 ) {
			System.out.println(n + ", " + m);
			n = m;
			m = r;
		}
		return m;
	}
	
	public static void main(String arg[]) {
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
		
	}

}
