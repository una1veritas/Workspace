//
//  BMHMachine.java
//  BMHMachine
//
//  Created by ?? ?? on 05/12/25.
//  Copyright (c) 2005 __MyCompanyName__. All rights reserved.
//
import java.util.*;

public class BMHMachine {
	char pattern[], alphabet[];
	int skip[];
	
	BMHMachine(char a[], char p[]) {
		alphabet = new char[a.length];
		pattern = new char[p.length];
		skip = new int[alphabet.length];
		for (int i = 0; i < alphabet.length; i++) {
			alphabet[i] = a[i];
			skip[i] = pattern.length;
		}
		for (int i = 0; i < pattern.length; i++) {
			pattern[i] = p[i];
		}
		for (int i = 0; i < pattern.length - 1; i++) {
			skip[p[i] - alphabet[0]] = pattern.length - i - 1;
		}
		return;
	}
	
	public synchronized String toString() {
		StringBuffer buf = new StringBuffer("BMH([");
		for (int i = 0; i < pattern.length; i++) {
			buf.append(pattern[i]);
			buf.append(", ");
		}
		buf.append("], [");
		for (int i = 0; i < alphabet.length; i++) {
			buf.append(alphabet[i]);
			buf.append(": ");
			buf.append(skip[i]);
			buf.append(", ");
		}
		buf.append("]) ");
		return buf.toString();
	}
	
	int performInnerLoop(char txt[], int pos) {
		int i = pos + pattern.length - 1;
		int j = pattern.length - 1;
		int count = 0;
		while (! (j < 0) ) {
			count++;
			if ( pattern[j] != txt[i] )
				break;
			j--;
			i--;
		}
		// if j == -1 then an occurrence has been found!
		return count;
	}
		
	int amountOfSkip(char t) {
		return skip[t - alphabet[0]];
	}
	
	public static char[] randomText(char[] alphabet, int length, double p) {
		Random rnd1, rnd2;
		char txt[] = new char[length];
		rnd1 = new Random();
		rnd1.setSeed(System.currentTimeMillis());
		rnd2 = new Random();
		rnd2.setSeed(rnd1.nextInt());
		for (int i = 0; i < length; i++) {
			txt[i] = alphabet[0];
			if (rnd1.nextDouble() < p) {
				txt[i] = alphabet[rnd2.nextInt(alphabet.length)];
			}
		}
		return txt;
	}
	
	public static long sum(long array[]) {
		long tsum = 0;
		for (int i = 0; i < array.length; i++)
			tsum += array[i];
		return tsum;
	}
	
	public static long weightedsum(long array[]) {
		long tsum = 0;
		for (int i = 0; i < array.length; i++)
			tsum += array[i] * i;
		return tsum;
	}
	
	public static double average(long array[]) {
		double tsum = 0, dsum = 0;
		for (int i = 0; i < array.length; i++) {
			tsum += array[i] * i;
			dsum += array[i];
		}
		return tsum / dsum;
	}
	
	public static void main (String args[]) {
		char txt[], pat[];
		
		char alphabet[]; // = {'a', 'b', 'c', 'd'};
		char pat_alphabet[];
		int tlen = 10000; 
		int plen = 14;
		double p = 0.1;
		int rep = 100000;
		
		alphabet = args[0].toCharArray();
		tlen = Integer.parseInt(args[1]); //7597; 
		plen = Integer.parseInt(args[2]); //13;
		p = Double.parseDouble(args[3]);  //0.1;
		rep = Integer.parseInt(args[4]); //92631;
		if (args.length == 6) {
			pat_alphabet = args[5].toCharArray();
		} else {
			pat_alphabet = args[0].toCharArray();
		}
		//
		long totalrunningtime = 0;
		long totalcount = 0;
		long totalpossiblepositions = 0;
		long stat_alphs[] = new long[alphabet.length];
		Arrays.fill(stat_alphs, 0);
		long stat_heads[] = new long[alphabet.length];
		Arrays.fill(stat_heads, 0);
		long stat_loops[] = new long[plen+1]; 
		Arrays.fill(stat_loops, 0);
		long stat_deltas[] = new long[plen+1];
		Arrays.fill(stat_deltas, 0);
		
		long stat_naive_loops[] = new long[plen+1];
		Arrays.fill(stat_naive_loops, 0);
		//
		for (int c = 0; c < rep; c++) {
			txt = randomText(alphabet, tlen, p);
			//pat = randomText(alphabet, plen, p);
			pat = randomText(pat_alphabet, plen, p);
			//
			for (int i = 0; i < tlen; i++) 
				stat_alphs[txt[i] - alphabet[0]]++;
			for (int i = 0; i < plen; i++)
				stat_alphs[pat[i] - alphabet[0]]++;
			//
			BMHMachine bmh = new BMHMachine(alphabet, pat);
			long tstamp = System.currentTimeMillis();
			int loops, delta;
			//
			for (int i = 0; i < txt.length - pat.length + 1; ) {
				loops = bmh.performInnerLoop(txt, i);
				delta = bmh.amountOfSkip(txt[i + pat.length - 1]);
				//
				totalcount++;
				stat_heads[txt[i+pat.length-1]-alphabet[0]]++;
				stat_loops[loops]++;
				stat_deltas[delta]++;
				//
				i += delta;
			}
			totalrunningtime += System.currentTimeMillis() - tstamp;
			totalpossiblepositions += txt.length - pat.length + 1;
			
			NaiveMachine naive = new NaiveMachine(alphabet, pat);
			for (int i = 0; i < txt.length - pat.length + 1; ) {
				loops = naive.performInnerLoop(txt, i);
				delta = naive.amountOfSkip();
				//
				stat_naive_loops[loops]++;
				//
				i += delta;
			}
		}
		
		System.out.println("alph. size " + alphabet.length + ", text len. " + tlen + ", pat. len. " + plen + ", prob. " + p + ", repeated " + rep + " times, avr. run. msec." + (totalrunningtime / (double) rep));
		//		System.out.println("Prbability mu = " + (mu_event_count / (double) mu_test_count));
		System.out.println();
		//System.out.println("statistics: ");
		System.out.println("symbol dist. ");
		for (int i = 0; i < alphabet.length; i++) {
			System.out.print("'" + alphabet[i] + "': " + (stat_alphs[i] /(double) sum(stat_alphs)) + ", ");
		}
		System.out.println();
		System.out.println();
		
		System.out.println("symbol dist. @ head ");
		for (int i = 0; i < alphabet.length; i++) {
			System.out.print("'" + alphabet[i] + "': " + (stat_heads[i] / (double) sum(stat_heads)) + ", ");
		}
		System.out.println();
		System.out.println();
		
		System.out.println("Ex[inner-loop] / Ex[delta] = " + average(stat_loops) + " / " +  average(stat_deltas) );
		//System.out.println("distribution: ");
		//for (int i = 0; i < stat_loops.length; i++) {
		//	System.out.println("[loops = " + i + "] " + (stat_loops[i] / (double) sum(stat_loops)));
		//}
		//System.out.println("distribution: ");
		//for (int i = 0; i < stat_deltas.length; i++) {
		//	System.out.println("[delta = " + i + "] " + (stat_deltas[i] / (double) sum(stat_deltas)));
		//}
		//System.out.println();
		
		System.out.println("Pr[i is head] = " + (totalcount/(double) totalpossiblepositions));
		System.out.println("Pr[loops/char] = " + (weightedsum(stat_loops) / (double) weightedsum(stat_deltas)) );
		System.out.println();
		
		System.out.println("Ex[naive inner-loop] " + average(stat_naive_loops) );
		//System.out.println("distribution: ");
		//for (int i = 0; i < stat_loops.length; i++) {
		//	System.out.println("[loops = " + i + "] " + (stat_naive_loops[i] / (double) sum(stat_naive_loops)));
		//}
		return;
	}
}
