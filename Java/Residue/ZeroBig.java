import java.io.*;
import java.lang.*;
import java.math.BigInteger;
import java.util.*;

 class ZeroBig {
    
    static BigInteger zero = new BigInteger("0");
    
    static BigInteger readBigInt() {
        BigInteger f;
        String line;
      	BufferedReader d = new BufferedReader(new InputStreamReader(System.in));
     
       	try {
         	line = d.readLine();
            f = new BigInteger(line);
        } catch(NumberFormatException e) {
            try {
                f = (new BigInteger(line.getBytes())).abs();
           	} catch (Exception e) {
           	    f = zero;
           	}
        }
       	return f;
   	}
    
  	public static void main(String args[]){  	
		BigInteger id,n,S;
        boolean kotae = false;
		
		while(true){
		    System.out.println("N = ?");
	 	    n = readBigInt();
		    System.out.println("S = ?");
            S = readBigInt();
            System.out.println("I = ?");
            id = readBigInt();
		    if(!id.equals(zero) && !n.equals(zero) && !S.equals(zero)){
		        break;
		    }
		    System.out.println("NO ZERO!!");
		}
		
		long chrono = System.currentTimeMillis();
		kotae = taiwa(id,S,n);
		chrono = chrono - System.currentTimeMillis();
		System.out.println("After "+chrono+" millisecs, ");
        if(kotae == true){
            System.out.println("you are the real person.");
		} else {
            System.out.println("you aren't the real person!!.");
		}
		readBigInt();
    }


 	static boolean taiwa(BigInteger id, BigInteger S, BigInteger n){
//	ZeroBig x = new ZeroBig();
//	ZeroBig y = new ZeroBig();
	    BigInteger x1,x2,y1,y2,z1,z2;
    	int k,f,N;
    	f=0;
        long e;
        N = n.bitLength();

		Random rnd = new Random();
 		for (k=1; k <= N; k++){
		    BigInteger r = new BigInteger(N,rnd);
		    x1 = r.multiply(r).mod(n);
       		if (x1.equals(zero)) {
                k--;
	    	    continue;
	    	}
     	    // System.out.println("x = "+x1);
            e = (long) Math.rint(Math.random());
            //  System.out.println("e = "+e);
            
    		if(e == 0){
                y1 = r.mod(n);
	    	} else {
                y1 = r.multiply(S).mod(n);
		    }
         	// System.out.println("y = "+y1);
     		y2 = y1.multiply(y1);
	    	x2 = x1.multiply(id);
        	
		    if(e == 0){
     		z1 = x1.mod(n);
            z2 = y2.mod(n);
			if(z1.equals(z2)){
			    // System.out.println("OK");
			    f++;
			} else {
			    f=0;
			    break ;
			}
		}
		else if (e == 1){
		    z1 = x2.mod(n);
		    z2 = y2.mod(n);
		    if(z1.equals(z2)){
		        // System.out.println("OK");
			    f++;
		    } else {
			    f=0;
 			    break;
 		    } 
	    } else {
            // System.out.println("error");
            return false;
		}
	}

	if (f > 0) {
	    return true; 
	} else {
	    return false;
	}
}
   

}
