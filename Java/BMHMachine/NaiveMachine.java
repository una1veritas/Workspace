//
//  NaiveMachine.java
//  NaiveMachine
//
//  Created by ?? ?? on 05/12/25.
//  Copyright (c) 2005 __MyCompanyName__. All rights reserved.
//
import java.util.*;

public class NaiveMachine {
	char pattern[], alphabet[];
	
	NaiveMachine(char a[], char p[]) {
		alphabet = new char[a.length];
		pattern = new char[p.length];
		for (int i = 0; i < alphabet.length; i++) {
			alphabet[i] = a[i];
		}
		for (int i = 0; i < pattern.length; i++) {
			pattern[i] = p[i];
		}
		return;
	}
	
	public synchronized String toString() {
		StringBuffer buf = new StringBuffer("Naive([");
		for (int i = 0; i < pattern.length; i++) {
			buf.append(pattern[i]);
			buf.append(", ");
		}
		buf.append("]) ");
		return buf.toString();
	}
	
	int performInnerLoop(char txt[], int pos) {
		int i = pos;
		int j = 0;
		int count = 0;
		while (j < pattern.length) {
			count++;
			if ( pattern[j] != txt[pos] )
				break;
			j++;
			pos++;
		}
		// if j == -1 then an occurrence has been found!
		return count;
	}
	
	int amountOfSkip() {
		return 1;
	}
	
	
}
