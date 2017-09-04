import java.util.Arrays;
import java.util.ArrayList;

/**
 * 二分木の節点．親と左右の子節点と，要素を持ちます．要素は比較可能でなくてはなりません．
 */
public class BTNode {

	private BTNode parent;
	private BTNode left, right;
	private ArrayList<BTNode> Child;
	private String Question;
	private String classname;	// どのクラスが属しているか(node, yes, no)
	private ArrayList<String> Elm;
	private String nodeName;

	// 要素も子節点もない空の節点を構築します．
	// （ノード名、右、左、リスト）
	public BTNode() {
		this.classname = null;
		this.Child = new ArrayList<BTNode>();
		this.Question = "NO TEST";
	}

	// (親ノード)要素も子節点もない空の節点を構築します.
	public BTNode(String TOPname, ArrayList List) {
		this.nodeName = TOPname;
		this.classname = "node";
		this.Child = new ArrayList<BTNode>();	
		this.Elm = List;
		this.Question = "NO TEST";
	}

	/**(葉)
	 * 要素だけを持ち，子節点は持たない節点を構築します．
	 */
	public BTNode(String name, ArrayList List, String clas) {
		this.nodeName = name;
		this.classname = clas;
		this.Child = new ArrayList<BTNode>();
		this.Child.add(null);
		this.Child.add(null);
		this.Elm = List;
		this.Question = "NO TEST";
	}

	/*
	 * 要素と，左右の子節点を持つ節点を構築します．
	 * @param element 要素 @param right 右子節点 @param left 左子節点
	*/
	public BTNode Connect(String name, BTNode[] child, ArrayList List, int QusNo) {
		this.nodeName = name;
		this.Elm = List;
		this.classname = "node";
		this.Question = "test"+QusNo;
		for(int i=0; i<child.length; i++){
			this.Child.add(child[i]);
		}
		return this;
	}

	public void setNodename(String name) { this.nodeName = name; }
	public String getNodeName() { return nodeName; }
	public void setClassname(String name) { this.classname = name; }
	public String getClassname() { return classname; }
	public void setName(String nodeName) { this.nodeName = nodeName; }
	public int getChildNUM() { return Child.size();}

	public void setZero(BTNode node) {
		this.Child.add(node);
		if (node != null)
			node.parent = this; // 親節点の設定
	}
	public void setRight(BTNode node) {
		this.Child.add(node);
		if (node != null)
			node.parent = this; // 親節点の設定
	}
	public BTNode getZero() { return Child.get(0); }
	public BTNode getChild(int x) { return Child.get(x); }
	/**
	 * 節点が葉かどうかを調べます．子節点のない節点は葉と呼ばれます．
	 */
	public boolean isLeaf() {
		return Child == null;
	}

	/**
	 * 節点の要素を文字列に変換します．
	 */
	public String toString() {
		if(Question.equals("NO TEST")){
			return nodeName.toString()+"_(<"+classname.toString()+":"+Elm.size()+">)";
		}
		return nodeName.toString()+"_(<"+Question.toString()+"><"+classname.toString()+":"+Elm.size()+">)";
	}
}
