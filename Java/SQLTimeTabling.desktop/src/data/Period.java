package data;

import java.io.*;
/**
 * @author masayoshi
 *
 * TODO 日付と作業時間の組(作業時間を日付ごとに分ける時使用)
 *
 */

public class Period implements Comparable, Serializable {
	private int id;
	private int day;
	private int order;
	private int nextid;
	
	private Period(){
	}
	
	public Period(int id){
		this.id = id;
		day = -1;
		order = 0;
	}
	
	public Period(int id, int day, int hint, int next){
		this.id = id;
		this.day = day;
		order = hint;
		nextid = next;
	}
	
	public int day() {
		return day;
	}
	
	public void day(int day) {
		this.day = day;
	}
	
	public int id() {
		return id;
	}
	
	public void id(int id) {
		this.id = id;
	}
	
	public int next() {
		return nextid;
	}
	
	public void next(int id) {
		this.nextid = id;
	}
	
	public boolean equals(Object obj){
		return ((Period) obj).id == id;
	}
	
	public int compareTo(Object obj) {
		Period p = (Period) obj;
		int diff = day - p.day;
		if ( diff == 0 )
			return order - p.order;
		return diff;
	}
	
	public int hashCode(){
		return id;
	}
	
	public String toString(){
		return "("+id+" {"+day+"; "+order+"})"; 
	}
}
