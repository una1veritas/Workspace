
import java.math.BigInteger;
import java.util.Random;
import java.io.*;

class Residue implements Serializable {
    
    private BigInteger residue, modulus;
    
    //
    
    static BigInteger Zero = new BigInteger("0");
    static BigInteger One = new BigInteger("1");
    static BigInteger Two = new BigInteger("2");
    
    //
    
    private static BigInteger big(long x) {
        return BigInteger.valueOf(x);
    }
    
    static BigInteger readBigInt() {
        BigInteger f;
        String line = "";
      	BufferedReader d = new BufferedReader(new InputStreamReader(System.in));
     
       	try {
         	line = d.readLine();
        } catch(IOException e1) {
            return Zero;
        }
        try {
            return new BigInteger(line);
        } catch (NumberFormatException e2) {
            try {
                return (new BigInteger(line.getBytes())).abs();
           	} catch(Exception e3) {
           	}
        }
       	return Zero;
   	}
        
    public Residue(BigInteger val, BigInteger mod) {
        modulus = mod.abs();
        residue = val.mod(modulus).add(modulus).mod(modulus);
        if (! residue.gcd(modulus).equals(One) ) {
            residue = Zero;
        }
    }
    
    public Residue(long val, long mod) {
        this(BigInteger.valueOf(val), BigInteger.valueOf(mod));
    }
    
    // Instance methods
    
    public Residue multiply(Residue x) {
        if (modulus == x.modulus) {
            return new Residue(residue.multiply(x.residue),modulus);
        } else {
            return new Residue(residue.multiply(x.residue),modulus.multiply(x.modulus));
        }
    }
    
    public boolean congruent(Residue r) {
        return r.residue.equals(residue) && r.modulus.equals(modulus);
    }
    
    public boolean congruent(BigInteger x) {
        return residue.equals(x.mod(modulus).add(modulus).mod(modulus));
    }
    
    public boolean congruent(BigInteger x, BigInteger y) {
        return x.mod(modulus).add(modulus).mod(modulus).equals(x.mod(modulus).add(modulus).mod(modulus));
    }
    
    public int legendre() {
        return legendre(residue);
    }
    
    public int legendre(BigInteger a) {
        BigInteger legval;
        
        legval = a.modPow(modulus.subtract(One).divide(Two), modulus);
        // System.out.println("Debug: "+new Residue(legval,modulus));
        if (legval.equals(modulus.subtract(One))) {
            return -1;
        }
        return legval.intValue();
    }
    
    public BigInteger quadraticNonresidue(int len) {
        Random rnd = new Random();
        BigInteger n;
        
        while (true) {
            n = new BigInteger(len, rnd);
            if (legendre(n) == -1) {
                return n;
            }
        }
    }
    
    //
    
    public synchronized String toString() {
        return residue.toString()+" (mod "+modulus.toString() + ") ";
    }
    
    public synchronized void binaryExpression() {
        try {
            java.io.ObjectOutputStream os = new java.io.ObjectOutputStream(System.out);
            os.writeObject(residue);
            os.flush();
        } catch (java.io.IOException e) {
        }
        
        //return residue.toByteArray();
    }
    
    // test routine
    /*
    public static void main(String argv[]) {
        Residue q ;
        Random rnd = new Random();
        
        Residue r = new Residue(new BigInteger("オッペルときたら，たいしたもんだ．".getBytes()) , new BigInteger("のんのんのんのんのんのんと，おおおそろしない音を".getBytes()));
        System.out.println(r);
        
        while (true) {
            q = new Residue(new BigInteger(512,rnd), Residue.prime(512,rnd));
            System.out.println(q+",\nLegendre symbol: "+q.legendre()+"\n");
            if (q.legendre() == 1) {
                break;
            }
        }
        
        System.out.println("Non-residue: "+q.quadraticNonresidue(q.modulus.bitCount()));
        System.out.println("終わったで.");
        
        q.binaryExpression();
        System.out.println("\n"+q.residue.toString(2));
        
        try {
		    (new java.io.BufferedReader(new java.io.InputStreamReader(System.in))).readLine();
		} catch(java.io.IOException e) { }

    }
    */


  	public static void main(String args[]){  	
		BigInteger n,p,q,r, eu,epasswd;
		Residue S, I;
		long chrono;
        boolean kotae = false;
		Random rnd = new Random();
	    
		while(true){
		    //System.out.println("N = ?");
	 	    //n = readBigInt();
	 	    p = new BigInteger(256, 256*2, rnd);
	 	    q = new BigInteger(256, 256*2, rnd);
	 	    n = p.multiply(q);
		    System.out.println("S = ?");
		    S = new Residue(readBigInt(), n);
            //System.out.println("I = ?");
            //id = readBigInt();
            I = S.multiply(S);
		    if( I.residue.signum() == 0 || n.signum() == 0 || S.residue.signum() == 0){
		        System.out.println("NO ZERO!!");
		        continue;
		    }
		    System.out.println("Public key: "+n+",\n Public Id: "+I+",\n Private password: "+S);
		    break;
		}
		readBigInt();
		
		chrono = -System.currentTimeMillis();
		kotae = FScheck(I,S);
		chrono += System.currentTimeMillis();
		System.out.println("After "+chrono+" millisecs, ");
        if(kotae == true){
            System.out.println("you are identified correctly.");
		} else {
            System.out.println("you are denied!!");
		}
		
		//
		eu = p.subtract(One).multiply(q.subtract(One));
		S.residue = S.residue.divide(S.residue.gcd(eu));
		System.out.println("Private key: "+S+",\nEuler's function = "+eu);
		//
		try {
    		I.residue = S.residue.modInverse(eu);
    	} catch (Exception e) {
    	    System.out.println("gcd(S.residue,n) = "+S.residue.gcd(n));
    	}
		System.out.println("Public key: "+n+",\nPublic id: "+I);
		
		chrono = -System.currentTimeMillis();
		r = (new BigInteger(512,rnd)).mod(n);
		epasswd = r.modPow(I.residue, n);
		kotae = epasswd.modPow(S.residue, n).equals(r);
		chrono += System.currentTimeMillis();
		
		System.out.println("One-time password: "+r+",\nUser's response: "+epasswd.modPow(S.residue, n));
		System.out.println("Result: "+kotae+" in "+chrono+" millisecs.");
		
		// (new Residue(10,10)).FSVerifier(n, System.in);
		readBigInt();
    }


 	static boolean FScheck(Residue qresidue, Residue root){
	    Residue r, x1, x2, y1, y2;
    	int e, N, f = 0;
        
        if (!qresidue.modulus.equals(root.modulus)) {
            return false;
        }
        N = root.modulus.bitLength();

		Random rnd = new Random();
 		while (N > 0){
		    r = new Residue(new BigInteger(N,rnd), root.modulus);
		    x1 = r.multiply(r);
       		if (x1.residue.signum() == 0) {
	    	    continue;
	    	}
     	    // System.out.println("x = "+x1);
            e = (int) Math.rint(Math.random());
            //  System.out.println("e = "+e);
            
    		if(e == 0){
                y1 = r;
	    	} else {
                y1 = r.multiply(root);
		    }
         	// System.out.println("y = "+y1);
     		y2 = y1.multiply(y1);
	    	x2 = x1.multiply(qresidue);
        	
		    if(e == 0){
    			if(x1.congruent(y2)){
	    		    // System.out.println("OK");
		    	    N--;
    			} else {
			        return false ;
			    }
    		} else /* if (e == 1) */ {
    		    if(x2.congruent(y2)){
    		        // System.out.println("OK");
	    		    N--;
    		    } else {
	    		    return false;
     		    }
    	    }
    	}

	    return true; 
    }


}
