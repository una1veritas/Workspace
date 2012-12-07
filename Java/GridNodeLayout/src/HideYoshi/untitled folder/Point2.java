//2ŽŸŒ³‚Ì“_

/**
 * @author hideaki
 */

public class Point2 {
	private Point p[];
	
	public Point2(int size){
		p = new Point[size];
	}
	
	public Point2(Point[] array){
		p = new Point[array.length];
		for(int i=0;i < p.length;i++){
			p[i] = new Point(array[i]);
		}
	}

	public Point[] get(int start,int size){
		Point getp[] = new Point[size];
		if(start+size-1 >= p.length){
			System.out.println("Out");
		}
		for(int i = 0;i < size;i++){
			getp[i] = new Point(p[start+i]);
		}
		
		return getp;
	}
	
	public String toString(){
		String out = "(";
		
		return out;	
	}
}
