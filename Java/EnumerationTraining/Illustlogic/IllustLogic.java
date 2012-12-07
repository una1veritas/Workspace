import java.util.*;

public class IllustLogic{
	int rSize=5;
	int cSize=5;
	private Vector r[] ;		/* 縦の数列 */
	private Vector c[] ;		/* 横の数列 */
	private int graph[][];	/* グラフの配列 */

	public IllustLogic(){
		rSize=5;
		cSize=5;
		r = new Vector[rSize];		/* 縦の数列 */
		c = new Vector[cSize];		/* 横の数列 */
		graph = new int[rSize][cSize];	/* グラフの配列 */
		
		Random rand = new Random();	/* 乱数作成用変数 */	
		int randomNumber;
		
		for (int i=0;i<rSize;i++){	/* 縦の配列の領域を確保 */
			this.r[i] = new Vector();
		}
		
		for (int j=0;j<cSize;j++){	/* 横の配列の領域を確保 */
			this.c[j] = new Vector();
		}

/* 適当な値を縦と横の配列に代入 */
		
		for (int i=0;i<rSize;i++){
			for(int rCount=0;rCount<rand.nextInt(2)+1;rCount++){
				this.r[i].addElement(new Integer(rand.nextInt(3)+1));
			}
		}
		
		for (int j=0;j<cSize;j++){
			for(int cCount=0;cCount<rand.nextInt(2)+1;cCount++){
				this.c[j].addElement(new Integer(rand.nextInt(3)+1));
			}
		}
/* 代入終わり */

/* 表の初期化 */		
//		int this.graph[][] = new int[rSize][cSize];
		
		for (int rCount=0;rCount<rSize;rCount++){
			for (int cCount=0;cCount<cSize;cCount++){
				this.graph[rCount][cCount] = 0;
			}
		}
/* 初期化終わり */

	}
	
	public String toString(){
		String row;			/*縦のキーの出力を入れる変数*/
		String colum;		/*横のキーの出力を入れる変数*/
		String graphString;	/*表の出力を入れる変数*/
		
/* 縦のキーを出力 */
		row = "縦のキー : \n";
		for(int rCount=0;rCount<rSize;rCount++){
			row += rCount + "列目 : ";
			row += this.r[rCount].toString();
			row += "\n";
		}
		
/* 横のキーを出力 */
		colum = "横のキー : \n";
		for(int cCount=0;cCount<cSize;cCount++){
			colum += cCount + "行目 : ";
			colum += this.c[cCount].toString();
			colum += "\n";
		}
		
/* 表を出力 */
		graphString = "表 : \n";
		for (int rCount=0;rCount<rSize;rCount++){
			for (int cCount=0;cCount<cSize;cCount++){
				if(this.graph[rCount][cCount] == 0){
					graphString += "□";
				} else {
					graphString += "■";
				}
			}
			graphString += "\n";
		}
		
/* 縦、横、表の出力をつなげて戻り値とする */
		return row+colum+graphString;
	}
}
		