
import java.io.*;
import java.lang.*;
import java.math.BigInteger;
import java.util.*;


public class Heihou{
	
	public static void main(String args[]){
		Heihou i = new Heihou();
		Heihou p = new Heihou();
		Heihou q = new Heihou();

		
		BigInteger X1,X2,Y;
		
 		BigInteger I = new BigInteger("0");
		BigInteger P = new BigInteger("0");
		BigInteger Q = new BigInteger("0");
		BigInteger N = new BigInteger("0");

		while (I.equals(BigInteger.ZERO)||P.equals(BigInteger.ZERO)
						||Q.equals(BigInteger.ZERO)){
			System.out.println("I = ?");
			I = i.getvalue4();
			System.out.println("p = ?");		
			P = p.getvalue4();
			System.out.println("q = ?");		
			Q = q.getvalue4();
  			if (I.equals(BigInteger.ZERO)||
			P.equals(BigInteger.ZERO)||Q.equals(BigInteger.ZERO)){
		 	 System.out.println("NO zero");
			}
		}
		X1 = Heihou.getheihou(I,P);
 		System.out.println("I,P=" +X1);
		X2 = Heihou.getheihou(I,Q);
     		  System.out.println("I,Q=" +X2);
		Y = Heihou.getchina(X1,X2,P,Q);
		System.out.println("kotaeha="+Y);
		if((Y.multiply(Y).mod(P.multiply(Q))).
					equals(I.mod(P.multiply(Q)))){
		System.out.println("OK");
		}
		else{
		System.out.println("NO");
		}
        }
		


  	static BigInteger getheihou(BigInteger i,BigInteger x ){
	BigInteger kotae,h,m,A,B,large,alp,bea,J;
 	int k,f,a1,h1,h2,sw,sw2,sw3;
	double p,f1,d,amari;
	Boolean hantei,hantei3,hantei4;
	boolean boo,boo2,boo3,hantei1,hantei2;
	String D;
	J = BigInteger.ZERO;
 	kotae = BigInteger.ZERO;
	large = BigInteger.ZERO;
        k=0;

	BigInteger Two = new BigInteger("2");
	BigInteger Three = new BigInteger("3");
	BigInteger Four = new BigInteger("4");
	BigInteger Five = new BigInteger("5");
	BigInteger Eight = new BigInteger("8");

	Random ram = new Random(); 
	BigInteger b;

	p = x.doubleValue();
	f1 = Math.log(p)/Math.log(2);
	f = (int)f1;
	while (true) {
		b = new BigInteger(f,ram);
		if (b.equals(BigInteger.ZERO)) {
			continue;
		}
		m = b.modPow((x.subtract(BigInteger.ONE)).divide(Two),x);
		if (!m.equals(BigInteger.ONE)) {
			break;
		}
	};
	System.out.println("Quardratic Non-residue: "+b);
	
        if(Three.equals(x.mod(Four))){
	 h = (x.subtract(Three)).divide(Four);
	 h1 = (h.add(BigInteger.ONE)).intValue();
	 kotae = i.pow(h1).mod(x);
	}
	else if(Five.equals(x.mod(Eight))){
	 h = (x.subtract(Five)).divide(Eight);
	 h1 = (h.multiply(Two).add(BigInteger.ONE)).intValue();
	 A =  i.pow(h1).mod(x);
		if (BigInteger.ONE.equals(A.mod(x))){
		 h2 = (h.add(BigInteger.ONE)).intValue(); 
		 kotae =  i.pow(h2).mod(x);
		}
		else {
		 h2 = (h.add(BigInteger.ONE)).intValue();
		 kotae = (i.pow(h2).multiply(b.pow(h1))).mod(x);
 		}
	}
	else if(BigInteger.ONE.equals(x.mod(Eight))){
	hantei = Boolean.FALSE;
	hantei3 = Boolean.FALSE;
	hantei4 = Boolean.FALSE ;
	sw = 0;
	sw2 = 0;
	sw3 = 0;
	h = (x.subtract(BigInteger.ONE)).divide(Eight);
	Random r1 = new Random();
	Random R1 = new Random();
	BigInteger r;
	BigInteger R; 
		do{
		p = x.doubleValue();
		f1 = Math.log(p)/Math.log(2);
		f = (int)f1;
		r = new BigInteger(f,r1);
	        R = new BigInteger(f,R1);
		boo = h.equals(Two.pow(r.intValue()).multiply(R));
		Boolean Boo = new Boolean(boo);
		hantei1 = hantei.equals(Boo);
		if(hantei1==false){
			if (BigInteger.ZERO.equals(R.subtract(BigInteger.ONE).
 					            mod(Two))){
	 	  	sw = 1;
	 
			}
			else{
		 	hantei1= true;
		 	sw = 0;
			} 
		}
		}while(hantei1&&sw==0);
	    	
		if(i.pow(R.intValue()).mod(x).equals(BigInteger.ONE)){
		 kotae = i.pow(((R.add(BigInteger.ONE)).divide(Two)).
							intValue()).mod(x);
		 return kotae; 
		}
		for(BigInteger j = BigInteger.ONE;j.equals(j.min(r.add(Two)));
						j.add(BigInteger.ONE)){
		 boo2 = BigInteger.ONE.equals((i.pow(Two.pow(j.intValue()).
					multiply(R).intValue())).mod(x));
		 Boolean Boo2 = new Boolean(boo2);
		 hantei2 = hantei3.equals(Boo2);
		 if(hantei2==true){
		 	 	if(j.equals(j.max(large))){
				 large = j;
				}
			 
		 }
		}
		J = large;
		alp = Two.pow(J.intValue()).multiply(R);
		bea = Two.pow(r.add(Two).intValue()).multiply(R);
		A = i.pow(alp.intValue()).mod(x);
		B = b.pow(bea.intValue()).mod(x);
		boo3 = alp.equals(R);
		Boolean Boo3 = new Boolean(boo3);
		while(sw3==0){
		 while((((A.multiply(B)).mod(x)).equals(BigInteger.ONE))&&
			(hantei4.equals(Boo3))){
		     alp = alp.divide(Two);
		     bea = bea.divide(Two);
		     A = i.pow(alp.intValue()).mod(x);
	 	     B = b.pow(bea.intValue()).mod(x);
		     boo3 = alp.equals(R);
		     Boolean Boo4 = new Boolean(boo3);
		     Boo3 = Boo4;
		 }
		     if(alp.equals(R)){
		     d = Math.sqrt((i.multiply(A).multiply(B)).doubleValue());
		     amari = Math.IEEEremainder(d,x.doubleValue());
		     D = Double.toString(amari);	
		     BigInteger kotae1 = new BigInteger(D);
		     kotae = kotae1;
		     return kotae;
		     }
		     else {
			bea = bea.add(Two.pow((r.add(Two)).intValue()).
							multiply(R));
			B = b.pow(bea.intValue()).mod(x);
			
		     }
		}
		
	}
	else {
	    System.out.println(" error ");
	    kotae =BigInteger.ZERO;
	}
	return kotae;
	
	}

	static BigInteger getchina(BigInteger y1,BigInteger y2,BigInteger p,
				BigInteger q ){
	BigInteger X,x1,x2,eulerp,eulerq;
	
	eulerp = p.multiply(BigInteger.ONE.subtract(BigInteger.ONE.divide(p)));
	eulerq = q.multiply(BigInteger.ONE.subtract(BigInteger.ONE.divide(q)));
	x1 = (q.pow(p.subtract(BigInteger.ONE).intValue())).multiply(y1).mod(p.multiply(q));
	x2 = (p.pow(q.subtract(BigInteger.ONE).intValue())).multiply(y2).mod(p.multiply(q));
	X = x1.add(x2);
	return X;
	}


	BigInteger getvalue4(){
        BigInteger f;
      	InputStreamReader c = new InputStreamReader(System.in);
      	BufferedReader d = new BufferedReader(c);
     
       	try{
         	String line = d.readLine();
                f = new BigInteger(line);
         	} catch(NumberFormatException e){
           	f = BigInteger.ZERO;
         	} catch(IOException e){
           	f = BigInteger.ZERO;
         	}
	
       	return f;
   	}
 }
