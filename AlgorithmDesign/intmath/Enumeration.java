import java.lang.Integer;
import java.lang.Math;

class Enumeration {

    static long gcd(long a, long b) {
	long c;

	for (c = Math.min(a, b); c > 1; c--) {
	    if ( (a % c == 0) && (b % c == 0) ) {
		return c;
	    }
	}
	return 1;
    }

    static public void main(String argv[]) {
	long a, b, divisor;

	if (argv.length != 2) {
	    System.err.println("Arguments are too few or many.");
	    return;
	}
	a = Integer.parseInt(argv[0]);
	b = Integer.parseInt(argv[1]);

	divisor = gcd(a, b);

	System.out.println("gcd of " + a + " and " + b + " is: " 
			   + divisor + ".");

	return;
    }
};
