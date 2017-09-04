import java.util.*;

public class VectorTest{
	Vector r;
	Vector c;
	Vector a;
	Vector array[];
	
	public static void main(String args[]){
		Vector r[] = new Vector[10];
		Vector c = new Vector();
		Random rand = new Random();
/*		r.addElement(new Integer(0));
		r.addElement(new Integer(3));
		r.addElement(new Integer(1));
		System.out.println(r.toString());
		c.addElement(r.clone());
		System.out.println(c.toString());
		
		r.removeAllElements();
		r.addElement(new Integer(1));
		r.addElement(new Integer(1));
		r.addElement(new Integer(1));
		System.out.println(r.toString());
		c.addElement(r.clone());
		Vector a = new Vector();
		a.addElement(c.elementAt(0));
		System.out.println(a.toString());
		a.removeElementAt(0);
		System.out.println(a.toString());
		//System.out.println(c.toString());
		Vector array[] = new Vector[9];
		for (int i=0;i<9;i++){
			array[i] = new Vector();
		}
		for (int i=0;i<9;i++){
			array[i].addElement(new Integer(i));
		}
		for (int i=0;i<9;i++){
			System.out.println(array[i].toString());
		}
*/		
		for (int i=0;i<10;i++){	/* c‚Ì”z—ñ‚Ì—Ìˆæ‚ðŠm•Û */
			r[i] = new Vector();
		}
		
		System.out.println(rand.nextInt(5));
		for (int i=0;i<10;i++){
			for(int rCount=0;rCount<(rand.nextInt(2)+1);rCount++){
				System.out.println(rCount);
				r[i].addElement(new Integer(rand.nextInt(2)+1));
			}
		}
		String row = "c‚ÌƒL[ : \n";
		for(int rCount=0;rCount<10;rCount++){
			row += rCount + "—ñ–Ú : ";
			row += r[rCount].toString();
			row += "\n";
		}
		System.out.println(row);
	}
}