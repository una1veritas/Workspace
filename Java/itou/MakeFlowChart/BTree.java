import java.util.Comparator;
import java.sql.*;
import java.io.*;
import java.util.ArrayList;
import java.lang.String;
import java.lang.System;
import java.util.Arrays;
import java.lang.Exception;

public class BTree extends BT {

	// 検索後の挿入・削除に用いる挿入・削除対象親節点の記憶用変数．
	protected BTNode parent;
	// 検索後の挿入・削除に用いる挿入・削除対象親節点の左右の記憶用変数．
	protected boolean lesser;

	// 空の二分探索木を構築します．
	public BTree() {  this(null);  }

	// 根を指定して，二分探索木を構築します．
	// @param root 二分探索木の根	
	public BTree(BTNode root) {  super(root);  }

	// 改めて利得計算用のColumnKindを設定する(属性値の数を再計算)
	public static ArrayList[] SetC_Each(BCon Data, ArrayList nodeList){
		int eachNUM, ColumnLeng=0;
		ArrayList[] C_E;
		String[] list;
		String[] kind; 
		int[] each;

		C_E = new ArrayList[Data.GetColumn()];
		for(int k=0; k<Data.GetColumn(); k++){	// 初期化
			C_E[k] = new ArrayList();
		}
		for(int n=0; n<Data.GetColumn(); n++){
			eachNUM = Data.GetC_EachNUM(n);  // 属性が取りうる値の種類の数
			kind = new String[Data.GetC_EachNUM(n)];
			for(int i=0; i<kind.length; i++){
				kind[i] = Data.getColumnKind(n,i);
			}
			each = new int[eachNUM];
			for(int i=0; i<nodeList.size(); i++){
				String scan = String.valueOf(Data.getArr(Integer.valueOf((String)nodeList.get(i)).intValue(),n));
				for(int j=0; j<kind.length; j++){
					if(kind[j].equals(scan)){
						each[j]++; }
				}
			}
			for(int a=0; a<eachNUM; a++){
				int flag=0;
				for(int b=0; b<kind.length; b++){
// >>>>>>>>
					if(Data.getColumnKind(n,a).equals(kind[b])){
						C_E[n].add(String.valueOf(each[b]));
						flag = 1;
						break;
					}
				}
				if(flag == 0){ C_E[n].add("0"); }
			}
		}
		return C_E;
	}
	public static boolean OneNode(int count[]){// もし子供が１つならばtrue、2つ以上ならばfalseを返す
		int flag=0;
		for(int i=0; i<count.length; i++){
			if(count[i] != 0 && flag==1){ return false; }
			if(count[i] != 0 && flag==0){ flag=1; }
		}
		return true;
	}
	public static void WriteNodeList(ArrayList<String> NodeList){
		System.out.print("<NodeList:");
		for(int i=0; i<NodeList.size(); i++){
			System.out.print(NodeList.get(i)+", ");
		}
		System.out.println(">");
	}

	//////// 1回目の木作成 ////////////
	public static BTNode ConnectTopNode(BCon Data, ArrayList<String> NodeList){
		int count[], GaNo=0, ColumnLeng=0;
		double gain=0.000, GetInfo=0.000;
		String Name = "Root";
		String Classes[];
		ArrayList[] Orig;			// 要素数確定版の子ノードリスト(可変×可変)
		String Cope[][];			// レコード番号のリスト(a,b:子ノード、List:Rootノード)
		String[][] C_each;
		BTNode Node, child[];

	// ①空ノードの作成
		Node = new BTNode(Name, NodeList);
	// ②利得比基準の計算(テストの決定)
		System.out.print(" [ "+Name+" ] について計算");
		WriteNodeList(NodeList);
		for(int i=0; i<Data.GetColumn()-1; i++){
			GetInfo = Data.GetGainRatio(i, Data.GetC_kind(), Data.GetC_each(), NodeList);
			System.out.println("test"+i+"の利得計算結果:Gain("+i+")="+GetInfo);
			if(gain < GetInfo){ gain = GetInfo; GaNo = i;}
		}
		ColumnLeng = Data.GetC_EachNUM(GaNo);
		System.out.println("Gain = "+gain+ " >> 属性"+GaNo+"で分別します。");
		System.out.println("");

	// ③選んだテストで分別する
		child = new BTNode[ColumnLeng];
		Cope = new String[ColumnLeng][NodeList.size()];
		count = new int[ColumnLeng];

		for(int j=0; j<NodeList.size(); j++){
			for(int k=0; k<ColumnLeng; k++){
				if(Data.getArr((Integer.valueOf((String)NodeList.get(j)).intValue()),GaNo).equals(Data.getColumnKind(GaNo,k))){
					Cope[k][count[k]] = String.valueOf(NodeList.get(j));
					count[k]++;
				}
			}
		}
		if(OneNode(count)){
			Node = new BTNode(Name, NodeList, Data.getArr(Data.GetColumn()-1, Integer.valueOf(Cope[0][0]).intValue()));
			return Node;
		}
		Orig = new ArrayList[ColumnLeng];
		for(int k=0; k<ColumnLeng; k++){	// 初期化
			Orig[k] = new ArrayList<String>();
		}
		for(int i=0; i<count.length; i++){	// N個のノードに分別
			for(int j=0; j<count[i]; j++){
				Orig[i].add(Cope[i][j]);
			}
		}
		Classes = new String[ColumnLeng];	// 各ノードのクラスを調べる
		for(int j=0; j<ColumnLeng; j++){
			Classes[j] = Data.getArr(Integer.valueOf((String)Orig[j].get(0)).intValue(), Data.GetColumn()-1);
		}
		for(int k=0; k<ColumnLeng; k++){	// ノードとして作成する
			child[k] = new BTNode(Data.getColumnKind(GaNo,k), Orig[k], Classes[k]);
		}

	// ④左右それぞれの子ノードの葉orノードを確認（再帰）
		for(int i=0; i<ColumnLeng; i++){
			if(!Data.CLASSsearch(Orig[i])){		// ノードの場合
				child[i] = ConnectNode(child[i], Data, count[i], Orig[i]);
			}
		}

	// ⑤親ノードに子、質問などをセットして返す
		return Node.Connect(Name, child, NodeList, GaNo);
	}

	////////////// 2回目以降の木作成部分 ///////////////
	public static BTNode ConnectNode(BTNode ParNode, BCon Data, int ListLength, ArrayList NodeList){
		int count[], GaNo=0, ColumnLeng=0;
		double gain=0.000, GetInfo=0.000;
		String Name = ParNode.getNodeName();
		String Classes[];
		ArrayList[] Orig;			// 要素数確定版の子ノードリスト
		String Cope[][];			// レコード番号のリスト(a,b:子ノード、List:Rootノード)
		ArrayList[] C_each;
		BTNode Node, child[];

	// ①親ノードはParNode（BTNode ParNode）で作成
		Node = ParNode;
	// ②利得比基準の計算(テストの決定)
		System.out.print("> [ "+Name+" ] について計算");
		WriteNodeList(NodeList);
// >>>>>
		C_each = SetC_Each(Data, NodeList);
		Data.WriteRecord(C_each);
		for(int i=0; i<Data.GetColumn()-1; i++){
			GetInfo = Data.GetGainRatio(i, Data.GetC_kind(), C_each, NodeList);
			System.out.println("test"+i+"の利得計算結果:Gain("+i+")="+GetInfo);
			if(gain < GetInfo){ gain = GetInfo; GaNo = i;}
		}
		ColumnLeng = Data.GetC_EachNUM(GaNo);
		System.out.println("Gain = "+gain+ " >> 属性"+GaNo+"で分別します。");
		System.out.println("");

	// ③分別する
		child = new BTNode[ColumnLeng];
		Cope = new String[ColumnLeng][NodeList.size()];
		count = new int[ColumnLeng];
		for(int j=0; j<NodeList.size(); j++){
			for(int k=0; k<ColumnLeng; k++){
				if(Data.getArr((Integer.valueOf((String)NodeList.get(j)).intValue()),GaNo).equals(Data.getColumnKind(GaNo,k))){
					Cope[k][count[k]] = String.valueOf(NodeList.get(j));
					count[k]++;
				}
			}
		}
		if(OneNode(count)){
			Node = new BTNode(Name, NodeList, Data.getArr(Data.GetColumn()-1, Integer.valueOf(Cope[0][0]).intValue()));
			return Node;
		}
		Orig = new ArrayList[ColumnLeng];
		for(int k=0; k<ColumnLeng; k++){	// 初期化
			Orig[k] = new ArrayList<String>();
		}
		for(int i=0; i<count.length; i++){	// N個のノードに分別
			for(int j=0; j<count[i]; j++){
				Orig[i].add(Cope[i][j]);
			}
		}
		Classes = new String[ColumnLeng];	// 各ノードのクラスを調べる
		for(int j=0; j<ColumnLeng; j++){
			if(!(C_each[GaNo].get(j)).equals("0"))	// 子ノードに分別されたレコード数が０でないとき
			Classes[j] = Data.getArr(Integer.valueOf((String)Orig[j].get(0)).intValue(), Data.GetColumn()-1);
	}
		for(int k=0; k<ColumnLeng; k++){	// ノードとして作成する
			if(!(C_each[GaNo].get(k)).equals("0"))
			child[k] = new BTNode(Data.getColumnKind(GaNo,k), Orig[k], Classes[k]);
		}

	// ④それぞれの子ノードの葉orノードを確認（再帰）
		for(int i=0; i<ColumnLeng; i++){
			if(!(C_each[GaNo].get(i)).equals("0"))
			if(!Data.CLASSsearch(Orig[i])){		// ノードの場合
				child[i] = ConnectNode(child[i], Data, count[i], Orig[i]);
			}
		}
	// ⑤親ノードに子、質問などをセットして返す
		return Node.Connect(Node.getNodeName(), child, NodeList, GaNo); 	// N1(right), N2(left)	
	}
///////////////////////////////////////

	public static void main(String args[]){

		int Case, Column, GainNUM=0, CaseNUM=0;
		BTNode TreeData, TreeCope;
		BCon AtrData;
		double gain=0.000, GetInfo=0.000;

//		try{

       	// データベースから読み込む
		System.out.println(">>> ①DBより事例集合を読み込みます。");
		AtrData = new BCon(args[0], args[1]);
        	Case = AtrData.GetCase();
        	Column = AtrData.GetColumn();
		// 事例集、属性値と属性値の数を表示する
		AtrData.PrintArr();
		AtrData.PrintC_KindAndEach(AtrData.GetC_kind(), AtrData.GetC_each());

		ArrayList<String> List = new ArrayList<String>();
		for(int i=0; i<Case; i++){
			List.add(String.valueOf(i));
		}
	// 決定木作成(使うデータ、事例番号リスト)
		System.out.println(">>> ②事例集合より木を作成します。");
		TreeData = ConnectTopNode(AtrData, List);
	// 決定木を表示する
		BTree tree = new BTree(TreeData);
		System.out.println(">>> ③決定木を表示します。");
		System.out.println(" << Decition Tree! >>"); System.out.println();
		tree.show();
/*		}
		catch(Exception e){
			System.out.println("例外："+e);
		}
*/	}
}

