import java.util.Arrays;
import java.util.ArrayList;

/**
 * �񕪖؂̐ߓ_�D�e�ƍ��E�̎q�ߓ_�ƁC�v�f�������܂��D�v�f�͔�r�\�łȂ��Ă͂Ȃ�܂���D
 */
public class BTNode {

	private BTNode parent;
	private BTNode left, right;
	private ArrayList<BTNode> Child;
	private String Question;
	private String classname;	// �ǂ̃N���X�������Ă��邩(node, yes, no)
	private ArrayList<String> Elm;
	private String nodeName;

	// �v�f���q�ߓ_���Ȃ���̐ߓ_���\�z���܂��D
	// �i�m�[�h���A�E�A���A���X�g�j
	public BTNode() {
		this.classname = null;
		this.Child = new ArrayList<BTNode>();
		this.Question = "NO TEST";
	}

	// (�e�m�[�h)�v�f���q�ߓ_���Ȃ���̐ߓ_���\�z���܂�.
	public BTNode(String TOPname, ArrayList List) {
		this.nodeName = TOPname;
		this.classname = "node";
		this.Child = new ArrayList<BTNode>();	
		this.Elm = List;
		this.Question = "NO TEST";
	}

	/**(�t)
	 * �v�f�����������C�q�ߓ_�͎����Ȃ��ߓ_���\�z���܂��D
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
	 * �v�f�ƁC���E�̎q�ߓ_�����ߓ_���\�z���܂��D
	 * @param element �v�f @param right �E�q�ߓ_ @param left ���q�ߓ_
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
			node.parent = this; // �e�ߓ_�̐ݒ�
	}
	public void setRight(BTNode node) {
		this.Child.add(node);
		if (node != null)
			node.parent = this; // �e�ߓ_�̐ݒ�
	}
	public BTNode getZero() { return Child.get(0); }
	public BTNode getChild(int x) { return Child.get(x); }
	/**
	 * �ߓ_���t���ǂ����𒲂ׂ܂��D�q�ߓ_�̂Ȃ��ߓ_�͗t�ƌĂ΂�܂��D
	 */
	public boolean isLeaf() {
		return Child == null;
	}

	/**
	 * �ߓ_�̗v�f�𕶎���ɕϊ����܂��D
	 */
	public String toString() {
		if(Question.equals("NO TEST")){
			return nodeName.toString()+"_(<"+classname.toString()+":"+Elm.size()+">)";
		}
		return nodeName.toString()+"_(<"+Question.toString()+"><"+classname.toString()+":"+Elm.size()+">)";
	}
}
