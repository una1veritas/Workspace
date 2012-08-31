import java.util.Random;
import java.math.BigInteger;

class PrimeGen {
    
    public static void main(String args[]) {

	System.out.println(BigInteger.probablePrime(16, new Random()));

	return;
    }
    
};
