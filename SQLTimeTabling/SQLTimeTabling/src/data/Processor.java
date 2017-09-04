package data;

import java.util.*;
import java.io.*;

public class Processor implements Serializable {
	private int id;
	
	private Processor() {
		//
	}
	
	
	public Processor(int id) {
		this.id = id;
	}
	
	
	public int id() {
		return id;
	}
	
	public void id(int id) {
		this.id = id;
	}
	
	
	public String toString(){
		StringBuffer a = new StringBuffer("Processor("+id+") ");
		return a.toString();
	}
	
	public boolean equals(Object o){
		if(!(o instanceof Processor))
			return false;
		return ((Processor)o).id == id;
	}
	
	public int hashCode(){
		return id;
	}
	
}
