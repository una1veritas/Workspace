import java.util.*;
/*
 * 
 * kopi-
 */
public class HashDp {
	private Map hdp;
	
	public HashDp(){
		hdp = new HashMap();
	}
	
	public HashDp(int init){
		hdp = new HashMap(init);
	}
	
	public HashDp(int init, float load){
		hdp = new HashMap(init,load);
	}
	
	Set keySet(){
		return hdp.keySet();
	}
	
	void add(int a, int b, int c, int d, int value){
		String keys = a + "," + b + "," + c + "," + d;
		Integer val = new Integer(value);
		hdp.put(keys, val);
	}
	
	void add(int a, int b, int c, int d, int value, int p, int t){
		String keys = a + "," + b + "," + c + "," + d;
		String val = value+","+ p + "," + t;
		hdp.put(keys, val);
	}
	
	int getint(int a, int b, int c, int d){
		String keys = a + "," + b + "," + c + "," + d;
		Integer v = (Integer)hdp.get(keys);
		if(v == null){
			return 10000000;
		}else{
			return v.intValue();
		}
	}
	
	String get(int a, int b, int c, int d){
		String keys = a + "," + b + "," + c + "," + d;
		return (String)hdp.get(keys);
	}
	
	void remove(int a, int b, int c, int d){
		String keys = a + "," + b + "," + c + "," + d;
		hdp.remove(keys);
	}
	
	int size(){
		return hdp.size();
	}
}
