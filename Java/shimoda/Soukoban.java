import java.io.*;

class Soukoban{
	char map[][] = new char[10][10];
	int mx, my;

	Soukoban(){
		int x,y;
		char ch;
		String str = "";
		
		try{
			BufferedReader br = new BufferedReader(new FileReader("map1.txt"));
			
			String a;
			while((a = br.readLine()) != null)
				str = str + a;
			
			br.close();
		}catch(IOException e){
			System.out.println("“ü—ÍƒGƒ‰[‚Å‚·B");
		}
		
		for(y=0;y<10;y++){
			for(x=0;x<10;x++){
				if((ch=str.charAt(x+y*10)) != 'M')
					map[x][y] = ch;
				else{
					map[x][y] = ch;
					mx = x;
					my = y;
				}
			}
		}
	}
	
	public static void main(String[] args){
		Soukoban s = new Soukoban();
		
		s.printMap();
		s.moveDown();
		s.printMap();
		/*System.out.println(s);*/
	}

	public char getElm(int x, int y){
		return map[x][y];
	}

/*	public String toString(){
		String str = "hello";
		return str;
	}*/

	public void printMap(){
		int x, y;
	
		for (y=0;y<10;y++){
			for(x=0;x<10;x++){
				System.out.print(map[x][y]);
			}
			System.out.println();
		}
	}

	public void moveRight(){
		if(map[mx+1][my] != 'W'){
			if(map[mx+1][my] != 'B'){
				map[mx+1][my] = 'M';
				map[mx][my] = ' ';
				mx = mx + 1;
			}else{
				if((map[mx+2][my] != 'W')&&(map[mx+2][my] != 'B')){
					map[mx+2][my] = 'B';
					map[mx+1][my] = 'M';
					map[mx][my] = ' ';
					mx = mx + 1;
				}
			}
		}
	}

	public void moveLeft(){
		if(map[mx-1][my] != 'W'){
			if(map[mx-1][my] != 'B'){
				map[mx-1][my] = 'M';
				map[mx][my] = ' ';
				mx = mx - 1;
			}else{
				if((map[mx-2][my] != 'W')&&(map[mx-2][my] != 'B')){
					map[mx-2][my] = 'B';
					map[mx-1][my] = 'M';
					map[mx][my] = ' ';
					mx = mx - 1;
				}
			}
		}
	}

	public void moveUp(){
		if(map[mx][my-1]!='W'){
			if(map[mx][my-1]!='B'){
				map[mx][my] = ' ';
				map[mx][my-1] = 'M';
				my = my - 1;
			}else{
				if((map[mx][my-2]!='W')&&(map[mx][my-2]!='B')){
					map[mx][my-2] = 'B';
					map[mx][my-1] = 'M';
					map[mx][my] = ' ';
					my = my - 1;
				}
			}
		}
	}

	public void moveDown(){
		if(map[mx][my+1] != 'W'){
			if(map[mx][my+1] != 'B'){
				map[mx][my+1] = 'M';
				map[mx][my] = ' ';
				my = my + 1;
			}else{
				if((map[mx][my+2] != 'W')&&(map[mx][my+2] != 'B')){
					map[mx][my+2] = 'B';
					map[mx][my+1] = 'M';
					map[mx][my] = ' ';
					my = my + 1;
				}
			}
		}
	}
}
