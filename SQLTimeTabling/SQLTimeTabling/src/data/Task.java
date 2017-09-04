package data;

import java.util.ArrayList;
import java.util.Iterator;

import java.io.*;


public class Task implements Serializable {
	//講義のid()
	private int id;
	//講義の種類
	private int qualification_id;
	//講師の人数の下限，上限
	private int processors_lb;
	private int processors_ub;
	//時間の連続(講義時間が連続する時間の作成時使用)
	private int length;
	
	private Task() {
		//
	}
	
	public Task(int id) {
		this.id = id;
		processors_lb = 1;
		processors_ub = 1;
		qualification_id = 0;
		length = 1;
		//availPeriods = new ArrayList();
	}
	
	public Task(int id, int req_lb, int req_ub, int qid, int length) {
		this.id = id;
		processors_lb = req_lb;
		processors_ub = req_ub;
		qualification_id = qid;
		this.length = length;
		//availPeriods = new ArrayList();
	}
	
	public int qualification_id() {
		return qualification_id;
	}
	
	public void qualification_id(int id) {
		this.qualification_id = id;
	}
	public int getProcessors_lb() {
		return processors_lb;
	}
	public void setProcessors_lb(int lb) {
		this.processors_lb = lb;
	}
	public int getProcessors_ub() {
		return processors_ub;
	}
	public void setProcessors_ub(int ub) {
		this.processors_ub = ub;
	}

	public int id() {
		return id;
	}
	public void id(int id) {
		this.id = id;
	}
	
	public int length() {
		return length;
	}
	
	public void length(int length) {
		this.length = length;
	}
	public boolean equals(Object o) {
		if(!(o instanceof Task))
			return false;
		return ((Task) o).id == this.id;
	}
	
	public int hashCode(){
		return id;
	}
	
	
	//
	public String toString(){
		return "Lecture("+id+"; "+length+")";
	}
}
