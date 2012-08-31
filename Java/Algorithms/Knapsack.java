import java.lang.*;
import java.util.*;

class Knapsack {
    int pricelist[];
    int budget;

    Knapsack(int b, int p[]) {
	pricelist = new int[p.length];
	for (int i = 0; i < p.length; i++)
	    pricelist[i] = p[i];
	budget = b;
    }

    int bestbuy(int l, int b) {
	int c1, c2;
	if ( l < 0 ) // No more item
	    return 0;
	if ( pricelist[l] > b )  // Too few money
	    return bestbuy(l-1, b);
	if ( (c1 = bestbuy(l-1, b)) > 
	     (c2 = pricelist[l] + bestbuy(l-1, b - pricelist[l])) ) {
	    return c1;  // Don't buy.
	} else {
	    return c2;	 // Buy the lth item.
	}
    }

    int dp() {
	int table[][] = new int[pricelist.length][budget+1];

	for (int i = 0; i < pricelist.length; i++)
	    table[i][0] = 0;
	for (int b = 0; b <= budget; b++)
	    if ( pricelist[0] > b )
		table[0][b] = 0;
	    else
		table[0][b] = pricelist[0];
	
	for (int i = 1; i < pricelist.length; i++)
	    for (int b = 1; b <= budget; b++)
		if ( pricelist[i] > b  ||
		     (table[i-1][b] > pricelist[i] + table[i-1][b-pricelist[i]]) )
		    table[i][b] = table[i-1][b];
		else
		    table[i][b] = table[i-1][b - pricelist[i]] + pricelist[i];
	
	return table[pricelist.length-1][budget];
    }


    public static void main(String args[]) {
	int p[];
	int b;
	Knapsack inst;

	p = new int[args.length - 1];
	b = Integer.parseInt(args[0]);
	for (int i = 1; i < args.length; i++) 
	    p[i-1] = Integer.parseInt(args[i]);

	System.out.println("Budget = " + b + " yen,\n" + p.length + " items,\nPrices: ");
	for (int i = 0; i < p.length; i++)
	    System.out.print("" + p[i] + ", ");
	System.out.println();

	inst = new Knapsack(b,p);

	System.out.println();
	System.out.println("\nSpend " + inst.bestbuy(p.length-1, inst.budget) + " yen.");
	//System.out.println("\nSpend " + inst.dp() + " yen.");

	return;
    }
}