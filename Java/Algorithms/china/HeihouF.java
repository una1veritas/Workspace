
import java.io.*;
import java.lang.*;
import java.math.BigInteger;
import java.math.BigDecimal;
import java.util.*;


public class HeihouF{

	static BigInteger Two = new BigInteger("2");
	static BigInteger Three = new BigInteger("3");
	static BigInteger Four = new BigInteger("4");
	static BigInteger Five = new BigInteger("5");
	static BigInteger Eight = new BigInteger("8");

	public static void main(String args[]){
		HeihouF i = new HeihouF();
		HeihouF p = new HeihouF();
		HeihouF q = new HeihouF();

		
		BigInteger g1,g2,S;
		
 		BigInteger I = new BigInteger("0");
		BigInteger P = new BigInteger("0");
		BigInteger Q = new BigInteger("0");
		BigInteger N = new BigInteger("0");
	
     		 while (true){
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
	
		g1 = I.modPow((P.subtract(BigInteger.ONE)).divide(Two),P);
		g2 = I.modPow((Q.subtract(BigInteger.ONE)).divide(Two),Q);
		if (g1.equals(BigInteger.ONE)){
		     if(g2.equals(BigInteger.ONE)){
			System.out.println("quadratic P and Q residue = OK");
		     	break;
			} 
		     else{
			System.out.println("quadratic P residue = OK");
		        System.out.println("quadratic Q residue = NO!!");
			
		     }
 		}
		else{
		    if(!g2.equals(BigInteger.ONE)){
		      System.out.println("quadratic P and Q residue = NO!!");
			
		    }
		    else{
			System.out.println("quadratic P residue = NO!!");
			System.out.println("quadratic Q residue = OK");
			
		    }
		}
		}
		S = passward(I,P,Q);
		System.out.println("kotaeha="+S);
		if((S.multiply(S).mod(P.multiply(Q))).
					equals(I.mod(P.multiply(Q)))){
		System.out.println("OK");
		}
		else{
		System.out.println("NO!");
		}
        }
		


  	static BigInteger getheihou(BigInteger i,BigInteger x ){
	BigInteger kotae,h,m,A,B,alp,bea,j,J,k;
	BigDecimal root,root_new;
 	int f,a1,h1,h2;
	double p,f1;
	J = BigInteger.ZERO;
 	kotae = BigInteger.ZERO;

	
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
	//  kotae = i.pow(h.add(BigInteger.ONE).intValue()).mod(x);
	 kotae = i.modPow((h.add(BigInteger.ONE)),x);
	}
	else if(Five.equals(x.mod(Eight))){
	 h = (x.subtract(Five)).divide(Eight);
	 A =  i.modPow(h.multiply(Two).add(BigInteger.ONE),x);
		if (BigInteger.ONE.equals(A.mod(x))){
		 kotae =  i.modPow(h.add(BigInteger.ONE),x);
		}
		else {
		 h2 = (h.add(BigInteger.ONE)).intValue();
		 kotae = (i.pow(h2).multiply(b.pow(h.multiply(Two)
			.add(BigInteger.ONE).intValue()))).mod(x);
 		}
	}
	else if(BigInteger.ONE.equals(x.mod(Eight))){

	h = (x.subtract(BigInteger.ONE)).divide(Eight);
	System.out.println("k = "+h);
	BigInteger r;
	BigInteger R; 
	BigInteger ha;
   
	if(BigInteger.ZERO.equals(h.subtract(BigInteger.ONE).mod(Two))){
		r = BigInteger.ZERO;
	        R = h;
	}
	else{ 
	    ha = h;
	     for(k = BigInteger.ONE;true;k=k.add(BigInteger.ONE)){
		ha = ha.divide(Two);
		      if(BigInteger.ZERO.equals(ha.subtract(
			   		     BigInteger.ONE).mod(Two))){
		 	  break ;
		      }
	     }
	     r = k;
	     R = ha;
	}
	
System.out.println("r = "+r);
System.out.println("R = "+R);
		if(i.modPow(R,x).equals(BigInteger.ONE)){
		 kotae = i.modPow((R.add(BigInteger.ONE)).divide(Two),x);
		 return kotae; 
		}
	
		for(j = r.add(BigInteger.ONE);true;j=j.subtract(
							BigInteger.ONE)){

		 if(!(BigInteger.ONE.equals(i.modPow(Two.pow(
					j.intValue()).multiply(R),x)))){
		      break;
		 }
		}

		J = j;
System.out.println("J = "+J);
		alp = Two.pow(J.intValue()).multiply(R);
		bea = Two.pow(r.add(Two).intValue()).multiply(R);
		A = i.modPow(alp,x);
		B = b.modPow(bea,x);
System.out.println("alp = "+alp);
System.out.println("bea = "+bea);
		while(true){
System.out.println("A = "+A);
System.out.println("B = "+B);
		 while((((A.multiply(B)).mod(x)).equals(BigInteger.ONE))&&
			(! alp.equals(R))){
		     alp = alp.divide(Two);
		     bea = bea.divide(Two);
		     A = i.modPow(alp,x);
		     B = b.modPow(bea,x);
System.out.println("Yes4");	   
		 }
System.out.println("A' = "+A);
System.out.println("B' = "+B);
//System.out.println("Yes5");
		 if(alp.equals(R)){
System.out.println("Yes6");
		   
		    BigInteger Ten20 = new BigInteger("100000000000000000000");
		    BigDecimal ROOT= new BigDecimal("0.00000000000000000001");
		    BigDecimal Two_D= new BigDecimal("2.00000000000000000000");
		    BigDecimal heihou = new BigDecimal(i.multiply(A)
			.multiply(B).multiply(Ten20),20);
		     
System.out.println("heihou = "+heihou);
		     root = heihou;
		     root_new = (root.add(heihou.divide(root,20,
 BigDecimal.ROUND_UNNECESSARY))).divide(Two_D,20,BigDecimal.ROUND_UNNECESSARY);

System.out.println("root_new = "+root_new);

		     while((root.subtract(root_new).min(ROOT)).equals(ROOT)){
			root = root_new;
//System.out.println("Yes while");

			root_new = (root.add(heihou.divide(root,20,
 BigDecimal.ROUND_HALF_EVEN))).divide(Two_D,20,BigDecimal.ROUND_HALF_EVEN);
//System.out.println("root="+root);	
//System.out.println("root_new="+root_new);		     
		     }
System.out.println("root="+root);
System.out.println("root_new="+root_new);
		     kotae = root_new.toBigInteger().mod(x);
		     if (! i.mod(x).equals(kotae.pow(2).mod(x))) {
		       System.err.println("Err!!");
		       System.err.println("+1 "+i.mod(x).equals(kotae.add(BigInteger.ONE).pow(2).mod(x)));
		     }
		     return kotae;
		 }
		 else {
System.out.println("Yes7");
			bea = bea.add(Two.pow((r.add(Two)).intValue()).
							multiply(R));
			B = b.modPow(bea,x);
	
		 }
		}
		
	}
	else {
	    System.out.println(" error ");
	    kotae =BigInteger.ZERO;
	}
	return kotae;
	
	}

	static BigInteger getchina(BigInteger x1,BigInteger x2,BigInteger p,
				BigInteger q ){
	BigInteger X,N,y1,y2,y3,y4,eulerp,eulerq,euler1,euler2;
	N = p.multiply(q);
//	eulerp = p.multiply(BigInteger.ONE.subtract(BigInteger.ONE.divide(p)));
//	eulerq = q.multiply(BigInteger.ONE.subtract(BigInteger.ONE.divide(q)));
	y1 = q.modPow(p.subtract(BigInteger.ONE),N);
	y2 = x1.mod(N);
	euler1 = y1.multiply(y2).mod(N);
	y3 = p.modPow(q.subtract(BigInteger.ONE),N);
	y4 = x2.mod(N);
	euler2 = y3.multiply(y4).mod(N);
	X = euler1.add(euler2);
//System.out.println("X="+X);
	return X.mod(N);
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

 	static BigInteger passward(BigInteger i,BigInteger p,BigInteger q){
		BigInteger x1,x2,s;


		x1 = HeihouF.getheihou(i,p);
 		System.out.println("I,P=" +x1);
		x2 = HeihouF.getheihou(i,q);
     		  System.out.println("I,Q=" +x2);
		s = HeihouF.getchina(x1,x2,p,q);
		return s;
	}
	
 }

