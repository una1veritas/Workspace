// package jp.ac.kit.ai.algorithm.patternmatching;

import java.lang.*;

abstract class StringMatching {
	char[] pattern;
	
	public StringMatching(String target) {
		pattern = target.toCharArray();
		// subclass responsibility;
			//{{INIT_CONTROLS
		//}}
}
	
	public String pattern() {
		return new String(pattern);
	}
	//{{DECLARE_CONTROLS
	//}}
}

public class KnuthMorrisPratt extends StringMatching {
	protected int bord[];
	
	public KnuthMorrisPratt(String pattern) {
		super(pattern);
		bord = new int[pattern.length() + 1];
		computeBorders();
			//{{INIT_CONTROLS
		//}}
}
	
	private void computeBorders() {
		int t, j;
		bord[0] = -1;
		for (j = 0, t = -1; j < pattern.length;  ) {
			while (t >= 0 
				&& pattern[t] != pattern[j] ) {
				t = bord[t];
			}
			t++;
			j++;
			if ( (j == pattern.length) || pattern[t] != pattern[j] ) {
				bord[j] = t;
			} else {
				bord[j] = bord[t];
			}
			//System.out.println(this);
		}
	}
	
	public int findIndex(String text, int beginIndex) {
		int i, j = 0;
		i = beginIndex;
		while (i <= text.length() - pattern.length) {
			while (j < pattern.length && pattern[j] == text.charAt(i+j)) {
				j++;
			}
			if (j == pattern.length) {
				return i;
			}
			i = i+j - bord[j]; 
			j = (0 > bord[j])? 0 : bord[j];
		}
		return -1;
	}
	
	public int findIndex(char[] text, int beginIndex) {
		int i, j = 0;
		i = beginIndex;
		while (i <= text.length - pattern.length) {
			while (j < pattern.length && pattern[j] == text[i+j]) {
				j++;
			}
			if (j == pattern.length) {
				return i;
			}
			i = i+j - bord[j]; 
			j = (0 > bord[j])? 0 : bord[j];
		}
		return -1;
	}
	
	public int findIndex(String text) {
		return findIndex(text, 0);
	}
	
	public int findIndex(char[] text) {
		return findIndex(text, 0);
	}
	
	public int findLastIndex(String text, int beginIndex) {
		int i, j = 0;
		i = beginIndex;
		while (i <= text.length() - pattern.length) {
			while (j < pattern.length && pattern[j] == text.charAt(text.length()-(i+j+1))) {
				j++;
			}
			if (j == pattern.length) {
				return text.length()-(i+j+1);
			}
			i = i+j - bord[j]; 
			j = (0 > bord[j])? 0 : bord[j];
		}
		return -1;
/*		int i, j;
		i = beginIndex;
		j = -1;
		while (i < text.length()) {;
			if ( (i = findIndex(text, i)) == -1) {
				break;
			}
			//System.out.println("at: "+j+"\n");
			j = i;
			i = i + 1;
		}
		return j;
*/	}
	
	public int findLastIndex(String text) {
		return findLastIndex(text, 0);
	}
	
	public String toString() {
		StringBuffer tmp = new StringBuffer("kmp: ");
		for (int i = 0; i < bord.length; i++) {
			tmp.append(bord[i]);
			tmp.append(", ");
		}
		return new String(tmp);
	}
	
	public static void main(String args[]) {
		String pat = "abaab";
		String text = "ababacababbabababbabcaaba";
		KnuthMorrisPratt kmp = new KnuthMorrisPratt(pat);
		System.out.println("pattern "+pat+" in text "+text + ":\n" + kmp.findIndex(text));
		System.out.println("pattern "+pat+" in text "+text + ":\n" + kmp.findLastIndex(text));
	}

	//{{DECLARE_CONTROLS
	//}}
}
