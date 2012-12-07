import java.util.*;

/*
 * トレースの仕方がいまいちわからないので、仮
 * 
 */

public class HashTrace {
	private Map hdp;
	
	public HashTrace(){
		hdp = new HashMap();
	}
	
	void add(int a, int b, int c, int d, int value1, int value2, List l){
		String keys = a + "," + b + "," + c + "," + d;
		String value = value1 + "," + value2;
		l.add(0,value);
		hdp.put(keys, l);
	}
	
	List get(int a, int b, int c, int d){
		String keys = a + "," + b + "," + c + "," + d;
		ArrayList l = (ArrayList)hdp.get(keys);
		return (List)l.clone();
	}
	
	int get_last_p(int a, int b, int c, int d){
		String keys = a + "," + b + "," + c + "," + d;
		ArrayList l = (ArrayList)hdp.get(keys);
		StringTokenizer stk = new StringTokenizer((String)l.get(0),",");
		return new Integer(stk.nextToken()).intValue();
	}
	
	int get_last_t(int a, int b, int c, int d){
		String keys = a + "," + b + "," + c + "," + d;
		ArrayList l = (ArrayList)hdp.get(keys);
		StringTokenizer stk = new StringTokenizer((String)l.get(0),",");
		stk.nextToken();
		return new Integer(stk.nextToken()).intValue();
	}
	
	void remove(int a, int b, int c, int d){
		String keys = a + "," + b + "," + c + "," + d;
		hdp.remove(keys);

	}
}
