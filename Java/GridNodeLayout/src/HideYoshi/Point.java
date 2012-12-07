//2次元の点

/**
 * @author hideaki
 */

public class Point {
	private int x;
	private int y;
	private int number;//ソート前の添え字
	
	public Point(Point p){
		x = p.x();
		y = p.y();
		number = p.num();
	}
	public Point(int a,int b){
		x = a;
		y = b;
	}
	
	public Point(int a,int b,int n){
		x = a;
		y = b;
		number = n;
	}	
	
	//x出力
	public int x(){
		return x;
	}
	
	//y出力
	public int y(){
		return y;
	}
	
	//元の配列の添え字を返す
	public int num(){
		return number;
	}
	//座標更新
	public void renew(int a, int b){
		x = a;
		y = b;
	}
	
	public void liner(int a, int b){
		x += a;
		y += b;
	}
	
	public void renew(Point a){
		x = a.x();
		y = a.y();
		number = a.num();
	}
	
	public void renewnumber(int a){
		number = a;
	}
	
	public boolean equals(Point obj){
		if(number == obj.num()){
			return true;
		}
		return false;
	}
	
	public String toString(){
		String out = "";
		//out += "(";
		out += x;
		out += ",";out += y;
		//out += ")";
		
		return out;	
	}
}
