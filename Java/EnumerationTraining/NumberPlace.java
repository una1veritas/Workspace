class NumberPlace {

    class Block {
	int item[][];

	Block(int c11, int c12, int c13, int c21, int c22, int c23, int c31, int c32, int c33) {
	    item = new int[3][3];

	    item[1][1] = c11;

	    return;
	}
	
	public int value(int r, int c) {
	    return item[r][c];
	}

	public int value(int r, int c, int v) {
	    return item[r][c] = v;
	}
    }


	Block bl[][];

	NumberPlace() {
		bl = new Block[3][3];
		bl[1][1] = new Block(1,2,3,4,5,6,7,8,9);
		return;
	}	

	public static void main() {
		NumberPlace ex1 = new NumberPlace();
		System.out.println(ex1);
	}

}
