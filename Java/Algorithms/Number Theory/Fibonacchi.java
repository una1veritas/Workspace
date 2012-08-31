import java.math.*;

class Fibonacchi {

    public static BigInteger fibonacchi(int n) {
	BigInteger f_2, f_1, tmp;
	int i;

	if (n == 0) {
	    return BigInteger.ZERO;
	} 
	if (n == 1) {
	    return BigInteger.ONE;
	}

	f_2 = BigInteger.valueOf(0);
	f_1 = BigInteger.valueOf(1);
	for ( i = 2; ! (i >  n); i++ ) {
	    tmp = f_2.add(f_1);
	    f_2 = f_1;
	    f_1 = tmp;
	}
	return f_1;
    }


    public static BigInteger rlfibonacchi(int n) {
	long f_2, f_1;

	if (n == 0) {
	    return BigInteger.ZERO;
	} 
	if (n == 1) {
	    return BigInteger.ONE;
	}
	return fibonacchi(n, 2, BigInteger.ONE, BigInteger.ZERO);
    }

    static BigInteger fibonacchi(int n, int curr, BigInteger f1, BigInteger f2) {
	if (n == curr) {
	    return f1.add(f2);
	}
	return fibonacchi(n, curr + 1, f1.add(f2), f1);
    }

/*
    public static long rfibonacchi(int n) {
	if (n == 0) {
	    return 0;
	} 
	if (n == 1) {
	    return 1;
	}
	return rfibonacchi(n-1) + rfibonacchi(n-2);
    }

*/
    
    public static void main(String[] args) throws Exception {
	long t;
	int n = 2;

	if (args.length > 0) {
	    n = Integer.parseInt(args[0]);
	}

	t = System.currentTimeMillis();
	System.out.println("Fl("+n+") = "+ fibonacchi(n) + ", computed in " 
			   + (System.currentTimeMillis() - t) + " msec.");

	t = System.currentTimeMillis();
	System.out.println("Frl("+n+") = "+ rlfibonacchi(n) + ", computed in " 
			   + (System.currentTimeMillis() - t) + " msec.");
/*
	t = System.currentTimeMillis();
	System.out.println("Fr("+n+") = "+ rfibonacchi(n) 
			   + ", computed in " + (System.currentTimeMillis()-t) 
			   + " msec.");
*/
	return;


    }
}
