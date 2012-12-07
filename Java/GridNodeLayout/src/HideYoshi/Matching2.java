
public class Matching2 {
	private Point p[];
	private Point t[];
	private int use_size;
	private int txmin;
	private int tymin;
	private int txmax;
	private int tymax;
	
	
	public Matching2(int maxsize){
		p = new Point[maxsize];
		t = new Point[maxsize];
		use_size = 0;
		txmin = 100000;
		tymin = 100000;
		txmax = 0;
		tymax = 0;
	}

	//
	public void add(Matching2 m){
		for(int i = 0;i < m.size();i++){
			p[use_size+i] = new Point(m.getp(i));
			t[use_size+i] = new Point(m.gett(i));
			if(txmin > t[use_size+i].x()){
				txmin = t[use_size+i].x();
			}
			if(tymin > t[use_size+i].y()){
				tymin = t[use_size+i].y();
			}
			if(txmax < t[use_size+i].x()){
				txmax = t[use_size+i].x();
			}
			if(tymax < t[use_size+i].y()){
				tymax =t[use_size+i].y();
			}
		}
		use_size += m.size();
	}
	
	public void add(Matching2 m, int x, int y){
		for(int i = 0;i < m.size();i++){
			p[use_size+i] = new Point(m.getp(i));
			t[use_size+i] = new Point(m.gett(i));
			t[use_size+i].liner(x, y);
			if(txmin > t[use_size+i].x()){
				txmin = t[use_size+i].x();
			}
			if(tymin > t[use_size+i].y()){
				tymin = t[use_size+i].y();
			}
			if(txmax < t[use_size+i].x()){
				txmax = t[use_size+i].x();
			}
			if(tymax < t[use_size+i].y()){
				tymax =t[use_size+i].y();
			}
		}
		use_size += m.size();
	}
	
	public void add(Matching m[]){
		for(int i = 0;i < m.length;i++){
			p[use_size+i] = new Point(m[i].getp());
			t[use_size+i] = new Point(m[i].gett());
			if(txmin > t[use_size+i].x()){
				txmin = t[use_size+i].x();
			}
			if(tymin > t[use_size+i].y()){
				tymin = t[use_size+i].y();
			}
			if(txmax < t[use_size+i].x()){
				txmax = t[use_size+i].x();
			}
			if(tymax < t[use_size+i].y()){
				tymax =t[use_size+i].y();
			}
		}
		use_size += m.length;
	}
	
	public void add(Matching m[],int size){
		for(int i = 0;i < size;i++){
			p[use_size+i] = new Point(m[i].getp());
			t[use_size+i] = new Point(m[i].gett());
			if(txmin > t[use_size+i].x()){
				txmin = t[use_size+i].x();
			}
			if(tymin > t[use_size+i].y()){
				tymin = t[use_size+i].y();
			}
			if(txmax < t[use_size+i].x()){
				txmax = t[use_size+i].x();
			}
			if(tymax < t[use_size+i].y()){
				tymax =t[use_size+i].y();
			}
		}
		use_size += size;
	}
	
	public void reset(){
		use_size = 0;
		txmin = 100000;
		tymin = 100000;
		txmax = 0;
		tymax = 0;
	}
	
	public int size(){
		return use_size;
	}
	
	public Point getp(int index){
		return p[index];
	}
	
	public Point gett(int index){
		return t[index];
	}
	
	public int gettxmin(){
		return txmin;
	}
	public int gettymin(){
		return tymin;
	}
	public int gettxmax(){
		return txmax;
	}
	public int gettymax(){
		return tymax;
	}

	
	public String toString(int index){
		String out = "";
		out += p[index].x();
		out += ",";
		out += p[index].y();
		out += ",";
		out += t[index].x();
		out += ",";
		out += t[index].y();

		return out;	
	}
}
