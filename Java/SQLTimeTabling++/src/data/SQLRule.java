package data;

import java.util.*;

public class SQLRule {
	private ArrayList<String> sqls;
	private int weight;
	String comment;
	
	public SQLRule(){
		sqls = new ArrayList<String>();
	}
	
	public void addSQL(String q){
		sqls.add(q);
	}
	
	public ArrayList<String> queries() {
		return sqls;
	}
	
	public void queries(ArrayList<String> q) {
		sqls = q;
	}
	
	public int weight() {
		return weight;
	}
	
	public void weight(int w) {
		weight = w;
	}

	public String comment(String s) {
		return comment = s;
	}
	
	public String comment() {
		return comment;
	}
	
	public String toString() {
		StringBuffer b = new StringBuffer();
		b.append("Weight: " + weight() + ", ");
		for (Iterator<String> i = sqls.iterator(); i.hasNext() ; ) {
			b.append(i.next() + "; ");
		}
		return b.toString();
	}
	
}
