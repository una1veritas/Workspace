package data;

import java.io.*;
/**
 * @author masayoshi
 *
 * TODO 日付と作業時間の組(作業時間を日付ごとに分ける時使用)
 *
 */

public class Period implements Comparable, Serializable {
	private int period_id;
	private int day_id;
	private int order_hint;
	
	private Period(){
	}
	
	public Period(int id){
		this.period_id = period_id;
		this.day_id = -1;
		order_hint = -1;
	}
	
	public Period(int id, int day, int hint){
		this.period_id = id;
		this.day_id = day;
		order_hint = hint;
	}
	public int getDay_id() {
		return day_id;
	}
	public void setDay_id(int day_id) {
		this.day_id = day_id;
	}
	public int getPeriod_id() {
		return period_id;
	}
	public void setPeriod_id(int period_id) {
		this.period_id = period_id;
	}
	
	public boolean equals(Period p){
		return p.period_id == period_id;
	}
	
	public int compareTo(Period p) {
		int diff = day_id - p.day_id;
		if ( diff == 0 )
			return order_hint - p.order_hint;
		return diff;
	}
	
	public int hashCode(){
		return period_id;
	}
	
	public String toString(){
		return "("+period_id+" {"+day_id+", "+order_hint+"})"; 
	}
}
