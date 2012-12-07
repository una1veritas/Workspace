package Nonogram;

import java.util.*;

public class Nonogram {
	Vector<Vector<Integer>> colseqs, rowseqs;
	
	public Nonogram(int args[]) {
		colseqs = new Vector<Vector<Integer>>();
		rowseqs = new Vector<Vector<Integer>>();
		//
		int cols, rows, p = 0;
		cols = args[p++];
		rows = args[p++];
		for(int i = 0; i < cols && p < args.length; i++, p++) {
			colseqs.add(new Vector<Integer>());
			for ( ; p < args.length; p++ ) {
				if ( args[p] == 0 )
					break;
				colseqs.lastElement().add(new Integer(args[p]));
			}
		}
		for(int i = 0; i < rows && p < args.length; i++, p++) {
			rowseqs.add(new Vector<Integer>());
			for ( ; p < args.length; p++ ) {
				if ( args[p] == 0 )
					break;
				rowseqs.lastElement().add(new Integer(args[p]));
			}
		}
	}
	
	public int rows() {
		return rowseqs.size();
	}
	
	public int columns() {
		return colseqs.size();
	}
	
	public synchronized String toString() {
		StringBuffer buf = new StringBuffer("");
		buf.append("Nonogram (");
		buf.append(colseqs.size() );
		buf.append(", ");
		buf.append(rowseqs.size());
		buf.append(", ");
		buf.append(colseqs);
		buf.append(", ");
		buf.append(rowseqs);
		buf.append(") ");
		return buf.toString();
	}
	
	public static Vector<Integer> parseSequence(String str) {
		Vector<Integer> seq = new Vector<Integer>();
		int i, cont;
		for (i = 0, cont = 0; i < str.length(); i++) {
			if ( str.charAt(i) == ' ' || str.charAt(i) == '0' ) {
				if ( cont > 0 ) {
					seq.add(new Integer(cont));
				}
				cont = 0;
			} else {
				cont++;
			}
		}
		if ( cont > 0 )
			seq.add(new Integer(cont));
		return seq;
	}
	
	public static void main(String args[]) {
		int argarray[] = new int[args.length];
		for (int i = 0; i < args.length; i++) {
			argarray[i] = Integer.parseInt(args[i]);
		}
		Nonogram myPuzzle = new Nonogram(argarray);
		System.out.println(myPuzzle);
		//
		char table[][];
		Random rnd = new Random();
		table = new char[myPuzzle.rows()][myPuzzle.columns()];
		for(int r = 0; r < myPuzzle.rows(); r++) {
			for(int c = 0; c < myPuzzle.columns(); c++) {
				if ( rnd.nextFloat() < 0.5 )
					table[r][c] = ' ';
				else
					table[r][c] = 'x';
			}
		}
	
		for(int r = 0; r < myPuzzle.rows(); r++) {
			for(int c = 0; c < myPuzzle.columns(); c++) {
				System.out.print(table[r][c]);
			}
			System.out.println();
		}
	}
};
