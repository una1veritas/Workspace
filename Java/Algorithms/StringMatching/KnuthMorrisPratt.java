// package jp.ac.kit.ai.algorithm.stringpatternmatching;

import java.lang.*;

abstract class PatternMatchingMachine {
	char[] pattern;
	
	public PatternMatchingMachine(String target) {
		pattern = target.toCharArray();
		// subclass responsibility;
	}
	
	public char[] pattern() {
		return pattern;
	}
}

public class KnuthMorrisPratt extends PatternMatchingMachine {
    protected int next[];
	
    public KnuthMorrisPratt(String pattern) {
	super(pattern);
	next = new int[pattern.length()];
	initialize();
    }
    
    private void initialize() {
	int i, j;
	for (i = 1, j = 0, next[0] = 0; i < pattern.length; ) {
	    /*
	    System.out.print("(" + i + ", " + j + "), chars: "+pattern[i]+", "+pattern[j]);
	    System.out.println(); System.out.println(pattern);
	    for (int tmp = 0; tmp < i - j; tmp++) {
		System.out.print(" ");
	    }
	    System.out.println(String.copyValueOf(pattern,0,j+1));
	    */
	    if ( pattern[i] == pattern[j] ) {
		next[i] = j + 1;
		j++; 
		i++;
	    } else {
		if (j > 0) {
		    j = next[j-1];
		} else {
		    i++;
		}
	    }
	}
    }


    public int findIndex(String text, int i) {
	int j = 0;
	for (j = 0; i < text.length() ; ) {
	    //System.out.println(" " + i + ", " + j);
	    //
	    if (! (j < pattern.length) ) {
		return i-j;
	    }
	    if ( pattern[j] == text.charAt(i) ) {
		j++;
		i++;
	    } else {
		if (j > 0) {
		    j = next[j-1];
		} else {
		    i++;
		}
	    }
	}
	return -1;
    }

    public int findIndex(String text) {
	return findIndex(text, 0);
    }

    public synchronized String toString() {
	StringBuffer tmp = new StringBuffer("KnuthMorrisPratt(\"");
	tmp.append(pattern);
	tmp.append("\", [");
	for (int i = 0; i < next.length; i++) {
	    tmp.append(next[i]);
	    if ( i + 1 < next.length ) {
		tmp.append(", ");
	    }
	}
	tmp.append("]) ");
	return new String(tmp);
    }
	
    public static void main(String args[]) {
	int idx;
	String pat = new String(args[0]);
	String text = new String(args[1]);
	KnuthMorrisPratt kmp = new KnuthMorrisPratt(pat);
	System.out.println(kmp);
	idx = kmp.findIndex(text,0);
	System.out.println("pattern "+pat+" in text "+text + " at " + idx);
	System.out.println("pattern "+pat+" in text "+text + " at " + kmp.findIndex(text,idx+1));
    }
    
}
