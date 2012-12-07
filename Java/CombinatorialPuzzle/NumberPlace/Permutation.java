/*
 * Created on 2004/11/02
 *
 * TODO To change the template for this generated file go to
 * Window - Preferences - Java - Code Style - Code Templates
 */

/**
 * @author sin
 *
 * TODO To change the template for this generated type comment go to
 * Window - Preferences - Java - Code Style - Code Templates
 */
public class Permutation {
	int perm[];

	public Permutation(int sz) {
		perm = new int[sz];
		int i, val;
		for(i=0, val = 1; i < perm.length; i++, val++)
			perm[i] = val;
	}
	
	public void next() /* throws Exception */ {
		int e;
		int x, l, r;

		for(x = perm.length - 1; (x != 0) && (perm[x-1] > perm[x]); x--) ;

		// reverse from x to perm.length - 1.
		for(l = x, r = perm.length - 1; l < r; l++, r--){
			e = perm[l];
			perm[l] = perm[r];
			perm[r] = e;
		}
		if ( x == 0 )
			return;

		// 
		for (l = x; perm[x-1] > perm[l]; l++);
		e = perm[x-1];
		perm[x-1] = perm[l];
		perm[l] = e;

		return;
	}
	
	public boolean hasNext() {
		for (int i = perm.length - 1; i > 0; i--){
			if (perm[i-1] < perm[i])
				return true;
		}
		return false;
	}
		
	public synchronized String toString() {
		StringBuffer tmp = new StringBuffer();
		int i;
		for(i = 0; i+1 < perm.length; i++) {
			tmp.append(perm[i]);
			tmp.append(", ");
		}
		tmp.append(perm[i]);
		return tmp.toString();
	}
	
	public static void main(String[] args) {
		Permutation p;
		
		for (p = new Permutation(Integer.parseInt(args[0])); ; p.next()) {
			System.out.println(p.toString());
			if (! p.hasNext())
				break;
		}
	}
}
