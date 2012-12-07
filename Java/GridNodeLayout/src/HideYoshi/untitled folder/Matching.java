
public class Matching {
	private Point p;
	private Point t;
	
	public Matching(Point pattern, Point text){
			p = new Point(pattern);
			t = new Point(text);
	}
	
	public void textrestore(int xmove, int ymove){
		t.liner(xmove, ymove);
	}

	public int t_x(){
		return t.x();
	}
	
	public int t_y(){
		return t.y();
	}
	
	public Point getp(){
		return p;
	}
	
	public Point gett(){
		return t;
	}
	
	public String toString(){
		String out = "";
		out += p.x();
		out += ",";
		out += p.y();
		out += ",";
		out += t.x();
		out += ",";
		out += t.y();

		return out;	
	}
}
