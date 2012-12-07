import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/*
 * ２次元点集合の近似照合本体
 */

/*
 * @author hideaki
 */

public class Hashapm {
	boolean L1flag = true;
	
	Point t[];//tがテキスト
	Point p[];//pがパターンh
	int m,n;//mがテキストサイズ、nがパターンサイズ
	Point x_order_t[], y_order_t[];// ソートされたテキスト
	Point x_order_p[], y_order_p[];// ソートされたパターン
	LinkedList t_list[],p_list[];//list
	HashDp DP;//DP表
	HashTrace HT;//トレース用
	
	//ここからは計算実験用の変数なので動作には関係ありません。
	long loop_count;
	long patternList_count;
	long textList_count;
	int p_p[];

	public Hashapm(String patternFileName, int textgrid, int widthSize){
		System.out.println("  Instace");
		Vector pvec = new Vector();
		String line;
		int scale = 1;
		try{
			BufferedReader reader = new BufferedReader(new FileReader(patternFileName));
			int number = 0;
			while ((line = reader.readLine()) != null){
				StringTokenizer stk = new StringTokenizer(line,",");
				Integer dx = new Integer(stk.nextToken());
				Integer dy = new Integer(stk.nextToken());
				pvec.addElement(new Point(dx.intValue()*scale,dy.intValue()*scale,number));			
				number++;
			}
	    	reader.close();
		} catch(Exception e){
			System.out.println("Error:パターンデータの読み込みに失敗しました。\n      "+e);
		}
		

		n = pvec.size();
		
		if(widthSize != 0){
			m = widthSize*widthSize;
			t = new Point[m];
			for(int i = 0; i < widthSize;i++){
				for(int j = 0; j < widthSize;j++){
					t[i*widthSize + j] = new Point(i*textgrid, j*textgrid, i*widthSize + j);
				}
			}
		}else{
			m=n*n;
			t = new Point[n*n];
			for(int i = 0; i < n;i++){
				for(int j = 0; j < n;j++){
					t[i*n + j] = new Point(i*textgrid, j*textgrid, i*n + j);
				}
			}
		}
		
		//パターンデータの配列への格納
		p = new Point[n];
		Enumeration e = pvec.elements();
		for(int i = 0; i < p.length;i++){
			p[i] = new Point((Point)e.nextElement());
		}

		x_order_t = new Point[m];
		y_order_t = new Point[m];
		x_order_p = new Point[n];
		y_order_p = new Point[n];
		DP = new HashDp();
		HT = new HashTrace();
		//リストはp[右,上],t[右,上]の順！！

		
		//リストをインスタンス化
		t_list = new LinkedList[m];
		for(int i = 0; i < m;i++){
			t_list[i] = new LinkedList();
		}
		p_list = new LinkedList[n];
		for(int i = 0; i < n;i++){
			p_list[i] = new LinkedList();
		}
		
		//テキストの値を代入
		for(int i = 0; i < m;i++){
			x_order_t[i] = new Point(t[i]);
			y_order_t[i] = new Point(t[i]);
		}
		//ソート
		x_quickSort(x_order_t,0,m-1);
		y_quickSort(y_order_t,0,m-1);
		
		//パターンの代入
		for(int i = 0; i < n;i++){
			x_order_p[i] = new Point(p[i]);
			y_order_p[i] = new Point(p[i]);
		}
		//ソート
		x_quickSort(x_order_p,0,n-1);
		y_quickSort(y_order_p,0,n-1);
	}
	
	public Hashapm(Vector text, Vector pattern){
		//実験のためのカウントの初期化
		loop_count = 0;
		patternList_count = 0;
		textList_count = 0;
		//実験カウントの初期化ここまで
		
		//インスタンス化
		m = text.size();
		n = pattern.size();
		t = new Point[m];
		x_order_t = new Point[m];
		y_order_t = new Point[m];
		p = new Point[n];
		x_order_p = new Point[n];
		y_order_p = new Point[n];
		DP = new HashDp();
		HT = new HashTrace();
		//リストはp[右,上],t[右,上]の順！！

		
		//リストをインスタンス化
		t_list = new LinkedList[m];
		for(int i = 0; i < m;i++){
			t_list[i] = new LinkedList();
		}
		p_list = new LinkedList[n];
		for(int i = 0; i < n;i++){
			p_list[i] = new LinkedList();
		}
		
		//テキストの値を代入
		Enumeration e = text.elements();
		for(int i = 0; i < m;i++){
			t[i] = new Point((Point)e.nextElement());
			t[i].renewnumber(i);
			x_order_t[i] = new Point(t[i]);
			y_order_t[i] = new Point(t[i]);
		}
		//ソート
		x_quickSort(x_order_t,0,m-1);
		y_quickSort(y_order_t,0,m-1);
		
		//パターンの代入
		e = pattern.elements();
		for(int i = 0; i < n;i++){
			p[i] = new Point((Point)e.nextElement());
			p[i].renewnumber(i);
			x_order_p[i] = new Point(p[i]);
			y_order_p[i] = new Point(p[i]);
		}
		//ソート
		x_quickSort(x_order_p,0,n-1);
		y_quickSort(y_order_p,0,n-1);
	}
	
	public Hashapm(Point[] text, Point[] pattern){
		//実験のためのカウントの初期化
		loop_count = 0;
		patternList_count = 0;
		textList_count = 0;
		//実験カウントの初期化ここまで
		
		//インスタンス化
		m = text.length;
		n = pattern.length;
		t = new Point[m];
		x_order_t = new Point[m];
		y_order_t = new Point[m];
		p = new Point[n];
		x_order_p = new Point[n];
		y_order_p = new Point[n];
		DP = new HashDp();
		HT = new HashTrace();
		p_p = new int[p.length];
		//リストはp[右,上],t[右,上]の順！！

		
		//リストをインスタンス化
		t_list = new LinkedList[m];
		for(int i = 0; i < m;i++){
			t_list[i] = new LinkedList();
		}
		p_list = new LinkedList[n];
		for(int i = 0; i < n;i++){
			p_list[i] = new LinkedList();
		}
		
		//テキストの値を代入
		for(int i = 0; i < m;i++){
			t[i] = new Point(text[i]);
			t[i].renewnumber(i);
			x_order_t[i] = new Point(t[i]);
			y_order_t[i] = new Point(t[i]);
		}
		//ソート
		x_quickSort(x_order_t,0,m-1);
		y_quickSort(y_order_t,0,m-1);
		
		//パターンの代入
		for(int i = 0; i < n;i++){
			p[i] = new Point(pattern[i]);
			p_p[i] = p[i].num();
			p[i].renewnumber(i);
			x_order_p[i] = new Point(p[i]);
			y_order_p[i] = new Point(p[i]);
		}
		//ソート
		x_quickSort(x_order_p,0,n-1);
		y_quickSort(y_order_p,0,n-1);
	}
	public void makeList(){
		int count = 0;
		//pのリスト作成
		for(int j = 0; j < n; j++){//ｙソートの添え字
			count = 0;
			int i = 0;
			int second_top = 0;
			int second_right =0;
			int second_top_index = -1;
			int second_right_index = -1;
			//同じ座標になるまで,点を数えながら移動
			while(!x_order_p[i].equals(y_order_p[j])){
				if(x_order_p[i].y() <= y_order_p[j].y()){
					count++;
					//今までの点の中で一番大きい点の添え字を保存
					if(second_top <= x_order_p[i].y()){
						second_top = x_order_p[i].y();
						second_top_index = x_order_p[i].num();
					}
					
					if(second_right <= x_order_p[i].x()){
						second_right  = x_order_p[i].x();
						second_right_index = x_order_p[i].num();
					}
				}
				i++;
			}
			//リスト作成
			for(; i < n; i++){//ｘソートの添え字
				if(x_order_p[i].y() <= y_order_p[j].y()){
					if(x_order_p[i].y() == y_order_p[j].y() && x_order_p[i].x() > y_order_p[j].x()){
						//System.out.println(i +","+j);
					}else{
						count++;//ひとつ数を増やして
						if(! x_order_p[i].equals(y_order_p[j])){
							if(second_top <= x_order_p[i].y()){
								second_top = x_order_p[i].y();
								second_top_index = x_order_p[i].num();
							}
						}
						p_list[count-1].add(new PointList(x_order_p[i].num(), y_order_p[j].num(), second_top_index, second_right_index));
						second_right_index = x_order_p[i].num();

						patternList_count++;//計算実験用
					}
				}
			}
		}
		//tのリスト作成
		for(int j = 0; j < m; j++){//ｙソートの添え字
			count = 0;
			int i = 0;
			int second_top = 0;
			int second_right =0;
			int second_top_index = -1;
			int second_right_index = -1;
			//同じ座標になるまで,点を数えながら移動
			while(!x_order_t[i].equals(y_order_t[j])){
				if(x_order_t[i].y() <= y_order_t[j].y()){
					count++;
					//今までの点の中で一番大きい点の添え字を保存
					if(second_top <= x_order_t[i].y()){
						second_top = x_order_t[i].y();
						second_top_index = x_order_t[i].num();
					}
					if(second_right <= x_order_t[i].x()){
						second_right = x_order_t[i].x();
						second_right_index = x_order_t[i].num();
					}
				}
				i++;
			}
			//リスト作成
			for(; i < m; i++){//ｘソートの添え字
				if(x_order_t[i].y() <= y_order_t[j].y()){
					if(x_order_t[i].y() == y_order_t[j].y() && x_order_t[i].x() > y_order_t[j].x()){
						//System.out.println(i +","+j);
					}else{
						count++;//ひとつ数を増やして
						if(! x_order_t[i].equals(y_order_t[j])){
							if(second_top <= x_order_t[i].y()){
								second_top = x_order_t[i].y();
								second_top_index = x_order_t[i].num();
							}
						}
						t_list[count-1].add(new PointList(x_order_t[i].num(), y_order_t[j].num(), second_top_index, second_right_index));
						second_right_index = x_order_t[i].num();

						textList_count++;//計算実験用
					}
				}
			}
		}
	}

	public void makeDP(){
		//P[i,j] = 1 のとき
		for(int i = 0; i < p_list[0].size();i++){
			PointList pl = (PointList)p_list[0].get(i);
			for(int z = 0; z < m;z++){
				for(int lt = 0; lt < t_list[z].size();lt++){
					PointList kl = (PointList)t_list[z].get(lt);
					DP.add(pl.i(),pl.j(),kl.i(),kl.j(),0);
					List list = new ArrayList();
					HT.add(pl.i(),pl.j(),kl.i(),kl.j(),pl.i(),kl.i(),list);
				}
			}
		}
		
		for(int r = 1; r < n;r++){
			//System.out.print(".");
			for(int z = r; z < m;z++){
				//System.out.println("  make----text:"+ (z+1));			
				for(int lp = 0; lp < p_list[r].size();lp++){
					PointList ij = (PointList)p_list[r].get(lp);
					for(int lt = 0; lt < t_list[z].size();lt++){
						PointList kl = (PointList)t_list[z].get(lt);
						//System.out.println("\nmake----[" +ij.i()+","+ij.j()+";"+kl.i()+","+kl.j()+"]");
						
						if(t[kl.i()].x() > t[kl.j()].x() && t[kl.i()].y() < t[kl.j()].y()){
							//[k,l]R /= [k,l]
							DP_pattern2(ij,kl,r,z);
						}else{
							// [k,l]R = [k,l]T
							DP_pattern1(ij,kl,r,z);
						}
					}
				}
			}
			
			//ハッシュの内容を全部表示
//			System.out.println("hash---:"+ r);
//			Set set = DP.keySet();
//			Iterator iterator = set.iterator();
//			Object object;
//			while(iterator.hasNext()){
//				object = iterator.next();
//				System.out.println(object + " = " + DP.get(object));
//			}

			//System.out.println("hash size---:"+ DP.size());
			
			//ハッシュの削除
			//System.out.println("remove----:"+ r);
			for(int z = r-1; z < m;z++){
				for(int lp = 0; lp < p_list[r-1].size();lp++){
					PointList ij = (PointList)p_list[r-1].get(lp);
					for(int lt = 0; lt < t_list[z].size();lt++){
						PointList kl = (PointList)t_list[z].get(lt);
						DP.remove(ij.i(), ij.j(), kl.i(), kl.j());
						//HT.remove(ij.i(), ij.j(), kl.i(), kl.j());
					}
				}
			}
			//System.out.println("hash size---:"+ DP.size());
		}

	}
	
	//[k,l]R = [k,l]T	の場合のDPの計算
	private void DP_pattern1(PointList ij, PointList kl,int pattern_list_size, int text_list_size) {
		int rightD = 10000000;
		int topD = 10000000;
		int rightD_index = -1;
		int topD_index = -1;
		int top_ = -1;		
		int right_ = -1;
		int DP_value = -1;
		//まずはright
		for(int w = pattern_list_size - 1;w < text_list_size; w++){
			int most_y_value = -1;
			int point_count = 0;
			int iteration = 0;
			int top = -1;
			while(point_count <= w){//w個になるところを探す
				if(this.isRectangle(kl.i(),kl.j(),x_order_t[iteration].num())){
					point_count++;
					if(most_y_value <= x_order_t[iteration].y()){
						most_y_value = x_order_t[iteration].y();
						top = x_order_t[iteration].num();
					}
				}
				iteration++;
				//計算量のためのカウント
				loop_count++;
			}
			iteration--;
			//ここまでで[k,l]<wがわかる
			
			if(ij.i() == ij.j()){
				DP_value = DP.getint(ij.remove_right(),ij.remove_top(),x_order_t[iteration].num(),top);			
				if(DP_value < 9999999){//無限大でなかったらの代理
					int ppp = HT.get_last_p(ij.remove_right(),ij.remove_top(),x_order_t[iteration].num(),top);
					int ttt = HT.get_last_t(ij.remove_right(),ij.remove_top(),x_order_t[iteration].num(),top);
					if(L1flag){
						DP_value += Math.abs( L1norm(p[ij.i()],p[ppp],t[kl.i()],t[ttt]));
					}else{
						DP_value += Math.abs( L1norm(p[ij.i()],p[ij.remove_right()],t[kl.i()],t[x_order_t[iteration].num()]));
					}
				}
			}else{
				DP_value = DP.getint(ij.remove_right(),ij.j(),x_order_t[iteration].num(),top);
				if(DP_value < 9999999){
					int ppp = HT.get_last_p(ij.remove_right(),ij.j(),x_order_t[iteration].num(),top);
					int ttt = HT.get_last_t(ij.remove_right(),ij.j(),x_order_t[iteration].num(),top);
					if(L1flag){
						DP_value += Math.abs( L1norm(p[ij.i()],p[ppp],t[kl.i()],t[ttt]));
					}else{
						DP_value += Math.abs( L1norm(p[ij.i()],p[ij.remove_right()],t[kl.i()],t[x_order_t[iteration].num()]));
					}
				}
			}
			if(rightD > DP_value){
				rightD = DP_value;
				rightD_index = iteration;
				top_ = top;
			}
		}
		//次にtop

		for(int w=pattern_list_size-1;w < text_list_size;w++){
			int most_x_value = -1;
			int point_count = 0;
			int iteration = 0;
			int right = -1;
			while(point_count <= w){//w個になるところを探す

				if(this.isRectangle(kl.i(),kl.j(),y_order_t[iteration].num())){
					point_count++;
					if(most_x_value <= y_order_t[iteration].x()){
						most_x_value = y_order_t[iteration].x();
						right = y_order_t[iteration].num();
					}
				}
				iteration++;
				//計算量のためのカウント
				loop_count++;
			}iteration--;//ここまでで[k,l]<wがわかる
			if(ij.i() == ij.j()){
				DP_value = DP.getint(ij.remove_right(),ij.remove_top(),right,y_order_t[iteration].num());
				if(DP_value < 9999999){
					int ppp = HT.get_last_p(ij.remove_right(),ij.remove_top(),right,y_order_t[iteration].num());
					int ttt = HT.get_last_t(ij.remove_right(),ij.remove_top(),right,y_order_t[iteration].num());
					if(L1flag){
						DP_value += Math.abs( L1norm(p[ij.j()],p[ppp],t[kl.j()],t[ttt]));
					}else{
						DP_value += Math.abs( L1norm(p[ij.j()],p[ij.remove_top()],t[kl.j()],t[y_order_t[iteration].num()]));
					}
				}
			}else{
				DP_value = DP.getint(ij.i(),ij.remove_top(),right,y_order_t[iteration].num());
				if(DP_value < 9999999){
					int ppp = HT.get_last_p(ij.i(),ij.remove_top(),right,y_order_t[iteration].num());
					int ttt = HT.get_last_t(ij.i(),ij.remove_top(),right,y_order_t[iteration].num());
					if(L1flag){
						DP_value += Math.abs( L1norm(p[ij.j()],p[ppp],t[kl.j()],t[ttt]));
					}else{
						DP_value += Math.abs( L1norm(p[ij.j()],p[ij.remove_top()],t[kl.j()],t[y_order_t[iteration].num()]));
					}
				}
			}
			if(topD > DP_value){
				topD = DP_value;
				topD_index = iteration;
				right_ = right;
			}
		}
		DP.add(ij.i(),ij.j(),kl.i(),kl.j(),min(rightD,topD));
		if(rightD > topD){
			if(ij.i() == ij.j()){
				HT.add(ij.i(),ij.j(),kl.i(),kl.j(),ij.j(),kl.j(),HT.get(ij.remove_right(), ij.remove_top(), right_, y_order_t[topD_index].num()));
			}else{
				HT.add(ij.i(),ij.j(),kl.i(),kl.j(),ij.j(),kl.j(),HT.get(ij.i(),ij.remove_top(),right_,y_order_t[topD_index].num()));
			}
		}else{
			if(ij.i() == ij.j()){
				HT.add(ij.i(),ij.j(),kl.i(),kl.j(),ij.i(),kl.i(),HT.get(ij.remove_right(),ij.remove_top(),x_order_t[rightD_index].num(),top_));
			}else{
				HT.add(ij.i(),ij.j(),kl.i(),kl.j(),ij.i(),kl.i(),HT.get(ij.remove_right(),ij.j(),x_order_t[rightD_index].num(),top_));
			}
		}
	}

	//[k,l]R /= [k,l]T	の場合のDPの計算関数
	private void DP_pattern2(PointList ij, PointList kl,int pattern_list_size, int text_list_size) {
		
		if(pattern_list_size == 1){//実際は2個、0からだから。
			DP.add(ij.i(),ij.j(),kl.i(),kl.j(),L1norm(p[ij.i()],p[ij.remove_right()],t[kl.i()],t[kl.j()]));
			HT.add(ij.i(),ij.j(),kl.i(),kl.j(),ij.j(),kl.j(),HT.get(ij.remove_top(),ij.remove_top(),kl.i(),kl.i()));		
		}else{
			int rightD = 10000000;
			int topD = 10000000;
			
			int rightD_index = -1;
			int topD_index = -1;
			
			int DP_value = 0;
			//始点探し
			int ite = 0;
			int point_count = 0;
			while(x_order_t[ite].num() != kl.j()){
				if(this.isRectangle(kl.i(),kl.j(),x_order_t[ite].num())){
					point_count++;
				}
				ite++;
				//計算量のためのカウント
				loop_count++;
			}
			while(point_count < text_list_size){
				if(this.isRectangle(kl.i(),kl.j(),x_order_t[ite].num())){
					point_count++;
					if(ij.i() == ij.j()){
						DP_value = DP.getint(ij.remove_right(),ij.remove_top(),x_order_t[ite].num(),kl.j());
						//System.out.print(" R[" +ij.remove_right()+","+ij.remove_top()+";"+x_order_t[ite].num()+","+kl.j()+"] = "+DP_value);
						if(DP_value < 9999999){
							int ppp = HT.get_last_p(ij.remove_right(),ij.remove_top(),x_order_t[ite].num(),kl.j());
							int ttt = HT.get_last_t(ij.remove_right(),ij.remove_top(),x_order_t[ite].num(),kl.j());
							if(L1flag){
								DP_value += Math.abs( L1norm(p[ij.i()],p[ppp],t[kl.i()],t[ttt]));
							}else{
								DP_value += Math.abs( L1norm(p[ij.i()],p[ij.remove_right()],t[kl.i()],t[x_order_t[ite].num()]));
							}
						}
					}else{
						DP_value = DP.getint(ij.remove_right(),ij.j(),x_order_t[ite].num(),kl.j());
						//System.out.print(" R[" +ij.remove_right()+","+ij.j()+";"+x_order_t[ite].num()+","+kl.j()+"] = "+DP_value);
						if(DP_value < 9999999){
							int ppp = HT.get_last_p(ij.remove_right(),ij.j(),x_order_t[ite].num(),kl.j());
							int ttt = HT.get_last_t(ij.remove_right(),ij.j(),x_order_t[ite].num(),kl.j());
							if(L1flag){
								DP_value += Math.abs( L1norm(p[ij.i()],p[ppp],t[kl.i()],t[ttt]));
							}else{
								DP_value += Math.abs( L1norm(p[ij.i()],p[ij.remove_right()],t[kl.i()],t[x_order_t[ite].num()]));
							}						}
					}

					//System.out.println(" ->"+DP_value);
					if(rightD > DP_value){
						rightD = DP_value;
						rightD_index = ite;
					}
				}
				ite++;
				//計算量のためのカウント
				loop_count++;
			}
			ite = 0;
			point_count = 0;
			while(y_order_t[ite].num() != kl.i()){
				if(this.isRectangle(kl.i(),kl.j(),y_order_t[ite].num())){
					point_count++;
				}
				ite++;
				//計算量のためのカウント
				loop_count++;
			}
			while(point_count < text_list_size){
				if(this.isRectangle(kl.i(),kl.j(),y_order_t[ite].num())){
					if(ij.i() == ij.j()){
						DP_value = DP.getint(ij.remove_right(),ij.remove_top(),kl.i(),y_order_t[ite].num());
						//System.out.print(" T[" +ij.remove_right()+","+ij.remove_top()+";"+kl.i()+","+y_order_t[ite].num()+"] = "+DP_value);
						if(DP_value < 9999999){
							int ppp = HT.get_last_p(ij.remove_right(),ij.remove_top(),kl.i(),y_order_t[ite].num());
							int ttt = HT.get_last_t(ij.remove_right(),ij.remove_top(),kl.i(),y_order_t[ite].num());
							if(L1flag){
								DP_value += Math.abs( L1norm(p[ij.j()],p[ppp],t[kl.j()],t[ttt]));
							}else{
								DP_value += Math.abs( L1norm(p[ij.j()],p[ij.remove_top()],t[kl.j()],t[y_order_t[ite].num()]));
							}
						}
					}else{
						DP_value = DP.getint(ij.i(),ij.remove_top(),kl.i(),y_order_t[ite].num());
						//System.out.print(" T[" +ij.i()+","+ij.remove_top()+";"+kl.i()+","+y_order_t[ite].num()+"] = "+DP_value);
						if(DP_value < 9999999){
							int ppp = HT.get_last_p(ij.i(),ij.remove_top(),kl.i(),y_order_t[ite].num());
							int ttt = HT.get_last_t(ij.i(),ij.remove_top(),kl.i(),y_order_t[ite].num());
							if(L1flag){
								DP_value += Math.abs( L1norm(p[ij.j()],p[ppp],t[kl.j()],t[ttt]));
							}else{
								DP_value += Math.abs( L1norm(p[ij.j()],p[ij.remove_top()],t[kl.j()],t[y_order_t[ite].num()]));
							}
						}
					}							
					if(topD > DP_value){
						topD = DP_value;
						topD_index = ite;
					}
					point_count++;
				}
				ite++;
				//計算量のためのカウント
				loop_count++;
			}
			DP.add(ij.i(),ij.j(),kl.i(),kl.j(),min(rightD,topD));
			//System.out.println("/DP:[" +ij.i()+","+ij.j()+"],["+kl.i()+","+kl.j()+"] =" + DP[ij.i()][ij.j()][kl.i()][kl.j()]);
			if(rightD > topD){
/*				if(ij.i() == 3 && ij.j() == 3 && kl.i() == 5 && kl.j() == 2){
					System.out.println("program check ----");
					if(ij.i() == ij.j()){
						System.out.println("("+ij.remove_right()+","+ij.remove_top()+";"+kl.i()+","+y_order_t[topD_index].num()+")");
						System.out.println("check:"+HT.get_last_p(ij.remove_right(),ij.remove_top(),kl.i(),y_order_t[topD_index].num()));
						System.out.println("check:"+DP.getint(ij.remove_right(),ij.remove_top(),kl.i(),y_order_t[topD_index].num()));
					}else{
						System.out.println("("+ij.i()+","+ij.remove_top()+";"+kl.i()+","+y_order_t[topD_index].num()+")");
						System.out.println("check2:"+HT.get_last_p(ij.i(),ij.remove_top(),kl.i(),y_order_t[topD_index].num()));
						System.out.println("check2:"+DP.getint(ij.i(),ij.remove_top(),kl.i(),y_order_t[topD_index].num()));
					}
					System.out.println("program check end----");
				}*/
				if(ij.i() == ij.j()){
					HT.add(ij.i(),ij.j(),kl.i(),kl.j(),ij.j(),kl.j(),HT.get(ij.remove_right(),ij.remove_top(),kl.i(),y_order_t[topD_index].num()));
				}else{
					HT.add(ij.i(),ij.j(),kl.i(),kl.j(),ij.j(),kl.j(),HT.get(ij.i(),ij.remove_top(),kl.i(),y_order_t[topD_index].num()));
				}
			}else{
				if(ij.i() == ij.j()){
					HT.add(ij.i(),ij.j(),kl.i(),kl.j(),ij.i(),kl.i(),HT.get(ij.remove_right(),ij.remove_top(),x_order_t[rightD_index].num(),kl.j()));
				}else{
					HT.add(ij.i(),ij.j(),kl.i(),kl.j(),ij.i(),kl.i(),HT.get(ij.remove_right(),ij.j(),x_order_t[rightD_index].num(),kl.j()));
				}
			}
		}
	}

	private int L1norm(Point i, Point j, Point k, Point l){
		int p_x_distance = i.x()-j.x();
		int p_y_distance = i.y()-j.y();
		int t_x_distance = k.x()-l.x();
		int t_y_distance = k.y()-l.y();
		return Math.abs(p_x_distance - t_x_distance) + Math.abs(p_y_distance - t_y_distance);
		
	}
	
	private int min(int right, int top) {
		if(right > top){
			return top;
		}
		return right;
	}

	private void x_quickSort(Point[] arr, int left, int right){
		if (left <= right) {
			Point p = new Point(arr[(left+right) / 2]);
			int l = left;
			int r = right;
			
            while(l <= r) {
            	while(arr[l].x() <= p.x()){
            		if(arr[l].x() == p.x() && arr[l].y() >= p.y()){
            			break;
            		}
            		l++;
            	}
            	while(arr[r].x() >= p.x()){
            		if(arr[r].x() == p.x() && arr[r].y() <= p.y()){
            			break;
            		}
            		r--;
                }
            	if (l <= r) {
                    Point tmp = new Point(arr[l]);
                    arr[l].renew(arr[r]);
                    arr[r].renew(tmp);
                    l++; 
                    r--;
                }else{
                }
            }

 
            x_quickSort(arr, left, r);
            x_quickSort(arr, l, right);
        }
    }
    private void y_quickSort(Point[] arr, int left, int right){
        if (left <= right) {
            Point p = new Point(arr[(left+right) / 2]);
            int l = left;
            int r = right;
            
            while(l <= r) {
            	while(arr[l].y() <= p.y()){
            		if(arr[l].y() == p.y() && arr[l].x() >= p.x()){
            			break;
            		}
            		l++;
            	}
                while(arr[r].y() >= p.y()){
            		if(arr[r].y() == p.y() && arr[r].x() <= p.x()){
            			break;
            		}
                	r--; 
                }
                
                if (l <= r) {
                    Point tmp = new Point(arr[l]);
                    arr[l].renew(arr[r]);
                    arr[r].renew(tmp);
                    l++; 
                    r--;
                }
            }
            
            y_quickSort(arr, left, r);
            y_quickSort(arr, l, right);
        }
    }
    
    public boolean isRectangle(int i, int j, int k){
    	if(t[j].y() >= t[k].y()&& t[i].x() >= t[k].x()){
    		if(t[j].y() == t[k].y() && t[j].x() < t[k].x()){
    			return false;
    		}
    		if(t[i].x() == t[k].x() && t[i].y() < t[k].y()){
    			return false;
    		}
    		return true;
    	}
    	return false;
    }
    
    //ここからは表示用の関数
    public void printtext(){
       	int size = t.length;
       	
       	System.out.println("text info ::size "+ size);
    	System.out.print("text point set={");
    	for(int j = 0;j < size; j++){
    		System.out.print(t[j]);
    	}
    	System.out.println("}");
    	System.out.print("     (x order)={");
       	for(int j = 0;j < size; j++){
    		System.out.print(x_order_t[j]);
    	}
    	System.out.println("}");
    	System.out.print("     (y order)={");
       	for(int j = 0;j < size; j++){
    		System.out.print(y_order_t[j]);
    	}
    	System.out.println("}");
    }
    
    public void printpattern(){
    	int size = p.length;
    	
       	System.out.println("pattern info ::size "+ size);
    	System.out.print("pattern point set={");
    	for(int j = 0;j < size; j++){
    		System.out.print(p[j]);
    	}
    	System.out.println("}");
    	System.out.print("        (x order)={");
       	for(int j = 0;j < size; j++){
    		System.out.print(x_order_p[j]);
    	}
    	System.out.println("}");
    	System.out.print("        (y order)={");
       	for(int j = 0;j < size; j++){
    		System.out.print(y_order_t[j]);
    	}
    	System.out.println("}");
    }
    
    public void printList(){
    	for(int i=0;i < p_list.length;i++){
    		int list_size = i+1;
        	System.out.print("pattern List("+ list_size +")={");
    		int size = p_list[i].size();
    		for(int j = 0;j < size; j++){
    			System.out.print(p_list[i].get(j) + ",");
    		}
    		System.out.println("}");

    	}
    }
    
    public void printTextList(){
    	for(int i=0;i < t_list.length;i++){
    		int list_size = i+1;
    		System.out.print("text List("+ list_size +")={");
    		int size = t_list[i].size();
    		for(int j = 0;j < size; j++){
    			System.out.print(t_list[i].get(j)+",");
    		}
 
        	System.out.println("}");

    	}
    }
    
    public void printDP(){
		for(int i = 0; i < n;i++){
			for(int j = 0; j < n;j++){
				for(int k = 0; k < m;k++){
					for(int a = 0; a < m;a++){
						
						System.out.print("DP["+i+"]["+j+"]["+k+"]["+a+"] = ");
						System.out.println(DP.getint(i,j,k,a));
					}
				}
			}
		}
    }
    
    public void printTrace(){
    	
    	int max = 100000;
    	int x_t = -1,y_t = -1;
       	int x_p = x_order_p[n-1].num();
       	int y_p = y_order_p[n-1].num();
    	System.out.println("trace------");   	
    	for(int i=0;i < m;i++){
        	for(int j=0;j < m;j++){
        		if(max > DP.getint(x_p,y_p,x_order_t[i].num(),y_order_t[j].num())){
        			max = DP.getint(x_p,y_p,x_order_t[i].num(),y_order_t[j].num());
        			x_t = x_order_t[i].num();
        			y_t = y_order_t[j].num();
        		}
        	}
    	}
		System.out.println(DP.getint(x_p, y_p, x_t, y_t));
    	List list1 = HT.get(x_p, y_p, x_t, y_t);
    	
    	try {
			BufferedWriter bw = new BufferedWriter(new FileWriter("result.txt", true));
	    	for(int i = 0; i < list1.size();i++){
	    		System.out.println(list1.get(i));
	    		String line = (String)list1.get(i);
	    		StringTokenizer stk = new StringTokenizer(line,",");
	    		String out = "";
	    		Integer a = new Integer(stk.nextToken());
	    		out += p[a.intValue()].x();
	    		out += ",";
	    		out += p[a.intValue()].y();
	    		Integer s = new Integer(stk.nextToken());
	    		out += ",";
	    		out += t[s.intValue()].x();
	    		out += ",";
	    		out += t[s.intValue()].y();
	    		out += "\n";
	    		bw.write(out);
	    	}
	    	bw.close();
		} catch (IOException e) {
			System.err.println(e);
		}
    }
   
    public Point[] outtextpoint(){
	   	Point out[];
    	int max = 100000;
    	int x_t = -1,y_t = -1;
       	int x_p = x_order_p[n-1].num();
       	int y_p = y_order_p[n-1].num();
 //   	System.out.println("trace------");
    	for(int i=0;i < m;i++){
        	for(int j=0;j < m;j++){
        		if(DP.getint(x_p,y_p,x_order_t[i].num(),y_order_t[j].num()) < 1000){
        			System.out.println(" !"+DP.getint(x_p,y_p,x_order_t[i].num(),y_order_t[j].num())+" (" + x_order_t[i].num() +","+y_order_t[j].num()+")");
        		}
        		if(max > DP.getint(x_p,y_p,x_order_t[i].num(),y_order_t[j].num())){
        			max = DP.getint(x_p,y_p,x_order_t[i].num(),y_order_t[j].num());
        			x_t = x_order_t[i].num();
        			y_t = y_order_t[j].num();
        		}
        	}
    	}
//		x_p = 2;
//		y_p = 2;
		x_t = 9;
		y_t = 9;
//		System.out.print("("+x_p+ ","+ y_p+";"+ x_t +","+ y_t+") DP:");
		System.out.println("  "+DP.getint(x_p, y_p, x_t, y_t)+" -Hashapm.outTrace");
    	List list1 = HT.get(x_p, y_p, x_t, y_t);
    	out = new Point[list1.size()];
    	for(int i = 0; i < list1.size();i++){
	    		String line = (String)list1.get(i);
	    		StringTokenizer stk = new StringTokenizer(line,",");
	    		Integer a = new Integer(stk.nextToken());
	    		Integer s = new Integer(stk.nextToken());
	    		out[i] = new Point(t[s.intValue()]);
	    		out[i].renewnumber(p_p[p[a.intValue()].num()]);
	   	}
    	
    	return out;
    }
    
   public Matching[] outTrace(){
	   	Matching out[];
    	int max = 100000;
    	int x_t = -1,y_t = -1;
       	int x_p = x_order_p[n-1].num();
       	int y_p = y_order_p[n-1].num();
 //   	System.out.println("trace------");
    	for(int i=0;i < m;i++){
        	for(int j=0;j < m;j++){
        		if(max > DP.getint(x_p,y_p,x_order_t[i].num(),y_order_t[j].num())){
        			max = DP.getint(x_p,y_p,x_order_t[i].num(),y_order_t[j].num());
        			x_t = x_order_t[i].num();
        			y_t = y_order_t[j].num();
        		}
        	}
    	}

//		System.out.print("("+x_p+ ","+ y_p+";"+ x_t +","+ y_t+") DP:");
		System.out.println("  "+DP.getint(x_p, y_p, x_t, y_t)+" -Hashapm.outTrace");
    	List list1 = HT.get(x_p, y_p, x_t, y_t);
    	out = new Matching[list1.size()];
    	for(int i = 0; i < list1.size();i++){
	    		String line = (String)list1.get(i);
	    		StringTokenizer stk = new StringTokenizer(line,",");
	    		Integer a = new Integer(stk.nextToken());
	    		Integer s = new Integer(stk.nextToken());
	    		out[i] = new Matching(p[a.intValue()],t[s.intValue()]);
	   	}
    	
    	return out;
    }
    
   public void printcell(int p1, int p2, int t1, int t2){
	   System.out.println(DP.getint(p1, p2, t1, t2));
   }
    public void jikken(){
    	System.out.print("テキストサイズ:" + m);
    	System.out.print("  矩形:" + textList_count);
    	System.out.print("  パターンサイズ:" + n);
    	System.out.println("  矩形:" + patternList_count+"");
    }
    
    public void jikken2(){
    	System.out.println("ループの回数:" + loop_count);   	
    }
}
