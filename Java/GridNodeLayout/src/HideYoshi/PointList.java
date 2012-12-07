
/**
 * @author hideaki
 */
public class PointList {
	private int i;
	private int j;
	private int second_top;//[i,j]-‚È‚Ç‚Ì—˜—p‚Ì‚½‚ß
	private int second_right;//[i,j]-‚È‚Ç‚Ì—˜—p‚Ì‚½‚ß

	
	public PointList(int k, int l){
		i = k;
		j = l;
	}
	
	public PointList(int k, int l, int s_k, int s_l){
		i = k;
		j = l;
		second_top = s_k;
		second_right = s_l;

	}
	
	public String toString(){
		String out = "";
		
		out += "(";
		out += i;
		out += ",";
		out += j;
		out += ")";
			
		out += "[";
		out += second_right;
		out += ",";
		out += second_top;
		out += "]";
		
		return out;
	}
	
	public int i(){
		return i;
	}
	
	public int j(){
		return j;
	}
	
	public int remove_top(){
		return second_top;
	}
	
	public int remove_right(){
		return second_right;
	}
}
