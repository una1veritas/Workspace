import java.io.*;
import java.util.Vector;
import java.util.Iterator;
import java.util.regex.*;
import java.lang.*;

import HideYoshi.Point;

/*
 * args[0] = coordinate : 座標ファイル 
 * args[1] = 1～4 :分割法 1: kd木 2:     3:面積等分  4:quad tree
 *
 *
 */
public class GridNodeLayout {
	public static int maxsize = 1000;
	Node p[];
	Node t[];
	int use_size;
	int txmin, txmax, tymin, tymax;
	//
	// }
	// public class ApmMain {
	public int size() {
		return use_size;
	}
	
	public GridNodeLayout(int maxsize){
		p = new Node[maxsize];
		t = new Node[maxsize];
		use_size = 0;
		txmin = 100000;
		tymin = 100000;
		txmax = 0;
		tymax = 0;
	}

	public static void main(String[] args) {
		String line, tokens[];
		Vector<Node> pvec = new Vector<Node>();
		// パターンデータを読み込み
		long instart, inend;
		instart = System.currentTimeMillis();
		try {
			BufferedReader reader = new BufferedReader(new FileReader(args[0]));
			int number = 0;
			int scale = 3;
			Pattern p = Pattern.compile(",");
			while ((line = reader.readLine()) != null) {
				// StringTokenizer stk = new StringTokenizer(line,",");
				tokens = p.split(line);
				int dx = Integer.parseInt(tokens[0]); // stk.nextToken());
				int dy = Integer.parseInt(tokens[1]); // stk.nextToken());
				// pvec.addElement(new
				// Node(dx.intValue(),dy.intValue(),number));
				pvec.add(new Node(number, dx * scale, dy * scale));
				number++;
			}
			reader.close();
		} catch (Exception e) {
			System.out.println("Error while reading points. ");
			System.out.println("      " + e);
		}

		// 配列への格納
		Node p[] = new Node[pvec.size()];
		java.util.Iterator<Node> e = pvec.iterator();
		for (int i = 0; i < p.length; i++) {
			p[i] = e.next();
		}

		// テキストデータの作成

		inend = System.currentTimeMillis();
		// System.out.println("インプット:"+(InputEnd-InputStart)+"[msec]" +
		// "- ApmMain.Main");
		// 近似照合
		long MatchingStart, MatchingEnd;
		int type = Integer.parseInt(args[1]);// 分割法を決める
		int grid = Integer.parseInt(args[2]);// グリッド幅
		int pCutSize = Integer.parseInt(args[3]);// カット幅

		MatchingStart = System.currentTimeMillis();

		// グリットの幅、カット時の最大サイズの設定はここ
		// Matching(p,100,10,type,args[0]);
		matching(p, grid, pCutSize, type, args[0]);
		MatchingEnd = System.currentTimeMillis();
		System.out.println("分割マッチング:" + (MatchingEnd - MatchingStart)
				+ "[msec]" + "- ApmMain.Main");
	}

	static void matching(Node[] p, int grid, int pCutSize, int type, String args) {
		// int n=0;
		GridNodeLayout m;

		m = approximatematch(p, grid, pCutSize, 0);
		OutputNodeCoord(m, args);
		System.out.println("Finished a pure approx. matching.");
	}

	// non-partitioning
	static GridNodeLayout approximatematch(Node[] p, int grid, int cut_size,
			int depth) {
		// System.out.println("depth:"+depth + "  size:" +p.length);
		long MatchingStart, MatchingEnd;
		MatchingStart = System.currentTimeMillis();
		// identifying the bounding rect.
		int p_xmax = p[0].x;
		int p_ymax = p[0].y;
		int p_xmin = p[0].x;
		int p_ymin = p[0].y;
		System.out.println("px: " + p[0].x + "  py: " + p[0].y);
		for (int i = 1; i < p.length; i++) {
			System.out.println("px: " + p[i].x + "  py: " + p[i].y);
			p_xmin = Math.min(p_xmin, p[i].x);
			p_ymin = Math.min(p_ymin, p[i].y);
			p_xmax = Math.max(p_xmax, p[i].x);
			p_ymax = Math.max(p_ymax, p[i].y);
		}
		System.out.println();
		//
		int p_xWidth = p_xmax - p_xmin;
		int p_yWidth = p_ymax - p_ymin;

		if (p_xWidth / grid < p.length - 1) {
			p_xWidth = (p.length - 1) * grid;
		}
		if (p_yWidth / grid < p.length - 1) {
			p_yWidth = (p.length - 1) * grid;
		}
		p_xmin = (p_xmin / grid) * grid;
		p_ymin = (p_ymin / grid) * grid;
		Node t[] = new Node[(p_xWidth / grid + 1) * (p_yWidth / grid + 1)];// Grid作成
		int t_count = 0;
		for (int i = 0; i <= p_xWidth; i += grid) {
			for (int j = 0; j <= p_yWidth; j += grid) {
				t[t_count] = new Node(i + p_xmin, j + p_ymin, t_count);
				System.out.println("tx: " + t[t_count].x + "  ty: "
						+ t[t_count].y);
				t_count++;
			}
		}
		System.out.println();

		GridNodeLayout out = new GridNodeLayout(p.length);
		// System.out.println("分割マッチング("+p.length+","+t.length+")  - ApmMain.Partition");
//		Hashapm apm = new Hashapm(t, p);
//		apm.makeList();
		// apm.jikken();
//		apm.makeDP();
		MatchingEnd = System.currentTimeMillis();
		System.out.println("  time: " + (MatchingEnd - MatchingStart)
				+ "[msec]  - ApmMain.Partition");
//		out.add(apm.outTrace());
		return out;

	}

	/*
	 * static void Marge(Matching[] m, Matching[] m1, Matching[] m2) { int j =
	 * 0;
	 * 
	 * for (; j < m1.length; j++) { m[j] = m1[j]; } for (; j < m.length; j++) {
	 * m[j] = m2[j - m1.length]; } }
	 */

	static void X_quickSort(Node[] arr, int left, int right) {
		if (left <= right) {
			Node p = new Node(arr[(left + right) / 2]);
			int l = left;
			int r = right;

			while (l <= r) {
				while (arr[l].x <= p.x) {
					if (arr[l].x == p.x && arr[l].y >= p.y) {
						break;
					}
					l++;
				}
				while (arr[r].x >= p.x) {
					if (arr[r].x == p.x && arr[r].y <= p.y) {
						break;
					}
					r--;
				}
				if (l <= r) {
					Node tmp = new Node(arr[l]);
					arr[l].replaceWith(arr[r]);
					arr[r].replaceWith(tmp);
					l++;
					r--;
				} else {
				}
			}
			X_quickSort(arr, left, r);
			X_quickSort(arr, l, right);
		}
	}

	static void Y_quickSort(Node[] arr, int left, int right) {
		if (left <= right) {
			Node p = new Node(arr[(left + right) / 2]);
			int l = left;
			int r = right;

			while (l <= r) {
				while (arr[l].y <= p.y) {
					if (arr[l].y == p.y && arr[l].x >= p.x) {
						break;
					}
					l++;
				}
				while (arr[r].y >= p.y) {
					if (arr[r].y == p.y && arr[r].x <= p.x) {
						break;
					}
					r--;
				}

				if (l <= r) {
					Node tmp = new Node(arr[l]);
					arr[l].replaceWith(arr[r]);
					arr[r].replaceWith(tmp);
					l++;
					r--;
				}
			}
			Y_quickSort(arr, left, r);
			Y_quickSort(arr, l, right);
		}
	}

	static Node[] makeGrid(int p_xWidth, int p_yWidth, int p_xmin, int p_ymin,
			int grid) {
		Node t[] = new Node[(p_xWidth / grid + 1) * (p_yWidth / grid + 1)];
		int t_count = 0;
		for (int i = 0; i <= p_xWidth; i += grid) {
			for (int j = 0; j <= p_yWidth; j += grid) {
				t[t_count] = new Node(i + p_xmin, j + p_ymin, t_count);
				t_count++;
			}
		}
		return t;
	}

	/*
	static Matching2 Apm(Node[] p, int grid) {
		long MatchingStart, MatchingEnd;
		MatchingStart = System.currentTimeMillis();
		int p_xmax = p[0].x();
		int p_ymax = p[0].y();
		int p_xmin = p[0].x();
		int p_ymin = p[0].y();
		for (int i = 1; i < p.length; i++) {
			if (p_xmin > p[i].x()) {
				p_xmin = p[i].x();
			}
			if (p_ymin > p[i].y()) {
				p_ymin = p[i].y();
			}
			if (p_xmax < p[i].x()) {
				p_xmax = p[i].x();
			}
			if (p_ymax < p[i].y()) {
				p_ymax = p[i].y();
			}
		}
		int p_xWidth = p_xmax - p_xmin;
		int p_yWidth = p_ymax - p_ymin;

		if (p_xWidth / grid < p.length - 1) {
			p_xWidth = (p.length - 1) * grid;
		}
		if (p_yWidth / grid < p.length - 1) {
			p_yWidth = (p.length - 1) * grid;
		}
		p_xmin = (p_xmin / grid) * grid;
		p_ymin = (p_ymin / grid) * grid;
		Node t[] = new Node[(p_xWidth / grid + 1) * (p_yWidth / grid + 1)];
		int t_count = 0;
		for (int i = 0; i <= p_xWidth; i += grid) {
			for (int j = 0; j <= p_yWidth; j += grid) {
				t[t_count] = new Node(i + p_xmin, j + p_ymin, t_count);
				t_count++;
			}
		}
		Matching2 out = new Matching2(p.length);
		// System.out.println("分割マッチング("+p.length+","+t.length+")  - ApmMain.Apm");
		Hashapm apm = new Hashapm(t, p);
		apm.makeList();
		// apm.jikken();
		apm.makeDP();
		MatchingEnd = System.currentTimeMillis();
		// System.out.println("  time: "+(MatchingEnd-MatchingStart)+"[msec]  - ApmMain.Apm");
		// System.out.println(apm.outTrace().length);
		out.add(apm.outTrace(), p.length);

		return out;
	}
*/
	
	static void OutputNodeCoord(GridNodeLayout m, String args) {
		/*
		 * //Draw用出力ファイル try{ BufferedWriter bw = new BufferedWriter(new
		 * FileWriter("result.txt")); for(int i=0;i < m.size();i++){
		 * bw.write(m.toString(i)+"\n"); } bw.close(); } catch(Exception e){
		 * System.out.println("Error1:パターンデータの読み込み"+e); }
		 */
		// MATLAB用出力ファイル
		try {
			String line, tokens[];
			Node[] old = new Node[m.size()];
			int number = 0;
			int x, y;
			BufferedReader reader = new BufferedReader(new FileReader(args));
			Pattern p = Pattern.compile(",");
			while ((line = reader.readLine()) != null) {
				// StringTokenizer stk = new StringTokenizer(line,",");
				tokens = p.split(line);
				x = Integer.parseInt(tokens[0]); // stk.nextToken());
				y = Integer.parseInt(tokens[1]); // stk.nextToken());
				old[number] = new Node(x * 3, y * 3, number);
				number++;
			}
			reader.close();

			BufferedWriter bw = new BufferedWriter(new FileWriter(
					"MATLAB_PMcoordinate.txt"));
			for (int i = 0; i < m.size(); i++) {
				for (int j = 0; j < m.size(); j++) {
					if ( (old[i].x == m.p[j].x) & old[i].y == m.p[j].y ) {
						bw.write(m.t[j].x + "");
						bw.write(",");
						bw.write(m.t[j].y + "\n");
						break;
					}
				}
			}
			bw.close();
		} catch (Exception e) {
			e.printStackTrace();
			System.out.println("Error2:パターンデータの読み込み" + e);
		}
	}
}
