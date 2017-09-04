package data;

import java.util.Enumeration;
import java.util.BitSet;

/**
 * 組み合わせを生成する Enumeration
 * N 個のうちから K 個を取り出す組み合わせ
 * 個数は Binomial[N, K]
 * （参考）Ｃ言語によるアルゴリズム入門
 */
public class CombEnum implements Enumeration {

	private int N;
	private BitSet X;
	private Object[] items;
	private Object[] array;

	public CombEnum(Object[] items, int k) {
		this.items = items;
		N = items.length;
		array = new Object[k];
		X  = new BitSet(N + 1);
		for(int i=0; i<k; i++) X.set(i);
	}

	public boolean hasMoreElements() {
		return ! X.get(N);
	}

	// 右から見て最初に１となる位置を返す
	// オール０なら -1 を返す
	// bs = 10011100
	// return 2
	private int findOne(BitSet bs) {
		int len = bs.size();
		for(int i=0; i<=N; i++) {
			if(bs.get(i)) return i;
		}
		return -1;
	}

	// 指定された位置でインクリメントする
	// 桁上がりした分の桁数を返す
	// bs = 10011100, n = 2
	// bs = 10100000
	// return 3
	private int incr(BitSet bs, int n) {
		int a = 0;
		for(;;) {
			if(bs.get(n)) {
				bs.clear(n); n++; a++;
			} else {
				bs.set(n); break;
			}
		}
		return a;
	}

	public Object nextElement() {
		int k = 0;
		for(int i=0; i<=N; i++) {
			if(X.get(i)) array[k++] = items[i];
		}
		int u = incr(X, findOne(X)) - 1;
		for(int i=0; i<u; i++) X.set(i);
		return array;
	}

	// テスト
	public static void main(String[] args) {
		String[] strs = {"1", "2", "3", "4", "5", "6"};
		int k = 4;
		System.out.println("N="+strs.length);
		System.out.println("K="+k);
		int count = 0;
		Enumeration e = new CombEnum(strs, k);
		while(e.hasMoreElements()) {
			Object[] a = (Object[])e.nextElement();
			System.out.print("{" + a[0]);
			for(int i=1; i<a.length; i++)
				System.out.print(", "+a[i]);
			System.out.println("}");
			count++;
		}
		System.out.println("count="+count);
	}

}
/*
N=6
K=3
{1, 2, 3}
{1, 2, 4}
{1, 3, 4}
{2, 3, 4}
{1, 2, 5}
{1, 3, 5}
{2, 3, 5}
{1, 4, 5}
{2, 4, 5}
{3, 4, 5}
{1, 2, 6}
{1, 3, 6}
{2, 3, 6}
{1, 4, 6}
{2, 4, 6}
{3, 4, 6}
{1, 5, 6}
{2, 5, 6}
{3, 5, 6}
{4, 5, 6}
count=20
*/
