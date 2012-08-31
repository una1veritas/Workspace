import java.lang.Integer;

class Euclidean {

    static long gcd(long a, long b) {
	long c;
	
	do { // System.err.print("a = " + a + ", b = " + b);
	    c = a % b; // System.err.println(", a mod b = " + c + ";");
	    a = b;
	    b = c;
	} while ( c != 0 );
	return a;
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
