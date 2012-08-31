package Algorithms;

class Fibo {
    static long cnt = 0;

    static long fibo(int n) {
	if (n == 3)
	    cnt++;
	if (n == 0 || n == 1 || n == 2)
	    return 1;
	return fibo(n-1) + fibo(n-2);
    }

    public static void main(String args[]) {
	int n = Integer.parseInt(args[0]);

	System.out.println(n);
	System.out.println(fibo(n));
	System.out.println("cnt = " + cnt);
    }

}
//nebbiolo:Algorithms% time java Fibo 50
//50
//12586269025
//cnt = 4807526976
//500.790u 0.021s 8:22.37 99.6%   0+0k 0+0io 1503pf+0w
