public class BT {
	// 二分木の頭です．頭の右の子節点が二分木の根になります．
	private BTNode head;

	// 空の二分木を構築します．
	public BT() {  this(null);  }

	// 根を指定して，二分木を構築します．
	// @param root 二分木の根
	public BT(BTNode root) {
		head = new BTNode(); // 木の頭を生成します．
		head.setZero(root); // 頭の左をルートにします．
	}

	// 木の根を設定します
	// @param node 根ノード
	public void setRoot(BTNode node) { head.setZero(node); }

	// 木の根を取得します
	// @return 根ノード
	public BTNode getRoot() { return head.getZero(); }

	/**
	 * ノードが木の根かどうかを判断します．
	 * @param node ノード
	 * @return 根なら真
	 */
	public boolean isRoot(BTNode node) { return getRoot() == node; }
	
	/**
	 * 木の頭を取得します．取得操作のみ．
	 * @return 木の頭
	 */
	protected BTNode getHead() { return head; }

	/**
	 * 二分木を表示します．インオーダで出力し，インデントもつけます．
	 * 実際の処理は補助操作を再帰的に使用します．
	 */
	public void show() { show(getRoot(), 0); }

	/**
	 * 二分木を表示するための再帰的処理です．
	 * @param node 表示する節点
	 * @param level インデントの深さ
	 */
	private void show(BTNode node, int level) {
		System.out.println(node);
		if(node.getZero() != null){
			for (int i = 0; i <= level; i++)
				System.out.print("       ");
			System.out.print("Ｌ");
			show(node.getZero(), level + 1);
		}
		int childNUM=0;
		childNUM = node.getChildNUM();
		for(int j=1; j<childNUM; j++){
			if(node.getChild(j) != null){
				for (int i = 0; i <= level; i++)
					System.out.print("       ");
				System.out.print("Ｌ");
				show(node.getChild(j), level + 1);
			}
		}
	}
}
