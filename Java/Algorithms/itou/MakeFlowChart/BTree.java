import java.util.Comparator;
import java.sql.*;
import java.io.*;
import java.util.ArrayList;
import java.lang.String;
import java.lang.System;
import java.util.Arrays;
import java.lang.Exception;

public class BTree extends BT {

	// ������̑}���E�폜�ɗp����}���E�폜�Ώېe�ߓ_�̋L���p�ϐ��D
	protected BTNode parent;
	// ������̑}���E�폜�ɗp����}���E�폜�Ώېe�ߓ_�̍��E�̋L���p�ϐ��D
	protected boolean lesser;

	// ��̓񕪒T���؂��\�z���܂��D
	public BTree() {  this(null);  }

	// �����w�肵�āC�񕪒T���؂��\�z���܂��D
	// @param root �񕪒T���؂̍�	
	public BTree(BTNode root) {  super(root);  }

	// ���߂ė����v�Z�p��ColumnKind��ݒ肷��(�����l�̐����Čv�Z)
	public static ArrayList[] SetC_Each(BCon Data, ArrayList nodeList){
		int eachNUM, ColumnLeng=0;
		ArrayList[] C_E;
		String[] list;
		String[] kind; 
		int[] each;

		C_E = new ArrayList[Data.GetColumn()];
		for(int k=0; k<Data.GetColumn(); k++){	// ������
			C_E[k] = new ArrayList();
		}
		for(int n=0; n<Data.GetColumn(); n++){
			eachNUM = Data.GetC_EachNUM(n);  // ��������肤��l�̎�ނ̐�
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
	public static boolean OneNode(int count[]){// �����q�����P�Ȃ��true�A2�ȏ�Ȃ��false��Ԃ�
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

	//////// 1��ڂ̖؍쐬 ////////////
	public static BTNode ConnectTopNode(BCon Data, ArrayList<String> NodeList){
		int count[], GaNo=0, ColumnLeng=0;
		double gain=0.000, GetInfo=0.000;
		String Name = "Root";
		String Classes[];
		ArrayList[] Orig;			// �v�f���m��ł̎q�m�[�h���X�g(�ρ~��)
		String Cope[][];			// ���R�[�h�ԍ��̃��X�g(a,b:�q�m�[�h�AList:Root�m�[�h)
		String[][] C_each;
		BTNode Node, child[];

	// �@��m�[�h�̍쐬
		Node = new BTNode(Name, NodeList);
	// �A�������̌v�Z(�e�X�g�̌���)
		System.out.print(" [ "+Name+" ] �ɂ��Čv�Z");
		WriteNodeList(NodeList);
		for(int i=0; i<Data.GetColumn()-1; i++){
			GetInfo = Data.GetGainRatio(i, Data.GetC_kind(), Data.GetC_each(), NodeList);
			System.out.println("test"+i+"�̗����v�Z����:Gain("+i+")="+GetInfo);
			if(gain < GetInfo){ gain = GetInfo; GaNo = i;}
		}
		ColumnLeng = Data.GetC_EachNUM(GaNo);
		System.out.println("Gain = "+gain+ " >> ����"+GaNo+"�ŕ��ʂ��܂��B");
		System.out.println("");

	// �B�I�񂾃e�X�g�ŕ��ʂ���
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
		for(int k=0; k<ColumnLeng; k++){	// ������
			Orig[k] = new ArrayList<String>();
		}
		for(int i=0; i<count.length; i++){	// N�̃m�[�h�ɕ���
			for(int j=0; j<count[i]; j++){
				Orig[i].add(Cope[i][j]);
			}
		}
		Classes = new String[ColumnLeng];	// �e�m�[�h�̃N���X�𒲂ׂ�
		for(int j=0; j<ColumnLeng; j++){
			Classes[j] = Data.getArr(Integer.valueOf((String)Orig[j].get(0)).intValue(), Data.GetColumn()-1);
		}
		for(int k=0; k<ColumnLeng; k++){	// �m�[�h�Ƃ��č쐬����
			child[k] = new BTNode(Data.getColumnKind(GaNo,k), Orig[k], Classes[k]);
		}

	// �C���E���ꂼ��̎q�m�[�h�̗tor�m�[�h���m�F�i�ċA�j
		for(int i=0; i<ColumnLeng; i++){
			if(!Data.CLASSsearch(Orig[i])){		// �m�[�h�̏ꍇ
				child[i] = ConnectNode(child[i], Data, count[i], Orig[i]);
			}
		}

	// �D�e�m�[�h�Ɏq�A����Ȃǂ��Z�b�g���ĕԂ�
		return Node.Connect(Name, child, NodeList, GaNo);
	}

	////////////// 2��ڈȍ~�̖؍쐬���� ///////////////
	public static BTNode ConnectNode(BTNode ParNode, BCon Data, int ListLength, ArrayList NodeList){
		int count[], GaNo=0, ColumnLeng=0;
		double gain=0.000, GetInfo=0.000;
		String Name = ParNode.getNodeName();
		String Classes[];
		ArrayList[] Orig;			// �v�f���m��ł̎q�m�[�h���X�g
		String Cope[][];			// ���R�[�h�ԍ��̃��X�g(a,b:�q�m�[�h�AList:Root�m�[�h)
		ArrayList[] C_each;
		BTNode Node, child[];

	// �@�e�m�[�h��ParNode�iBTNode ParNode�j�ō쐬
		Node = ParNode;
	// �A�������̌v�Z(�e�X�g�̌���)
		System.out.print("> [ "+Name+" ] �ɂ��Čv�Z");
		WriteNodeList(NodeList);
// >>>>>
		C_each = SetC_Each(Data, NodeList);
		Data.WriteRecord(C_each);
		for(int i=0; i<Data.GetColumn()-1; i++){
			GetInfo = Data.GetGainRatio(i, Data.GetC_kind(), C_each, NodeList);
			System.out.println("test"+i+"�̗����v�Z����:Gain("+i+")="+GetInfo);
			if(gain < GetInfo){ gain = GetInfo; GaNo = i;}
		}
		ColumnLeng = Data.GetC_EachNUM(GaNo);
		System.out.println("Gain = "+gain+ " >> ����"+GaNo+"�ŕ��ʂ��܂��B");
		System.out.println("");

	// �B���ʂ���
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
		for(int k=0; k<ColumnLeng; k++){	// ������
			Orig[k] = new ArrayList<String>();
		}
		for(int i=0; i<count.length; i++){	// N�̃m�[�h�ɕ���
			for(int j=0; j<count[i]; j++){
				Orig[i].add(Cope[i][j]);
			}
		}
		Classes = new String[ColumnLeng];	// �e�m�[�h�̃N���X�𒲂ׂ�
		for(int j=0; j<ColumnLeng; j++){
			if(!(C_each[GaNo].get(j)).equals("0"))	// �q�m�[�h�ɕ��ʂ��ꂽ���R�[�h�����O�łȂ��Ƃ�
			Classes[j] = Data.getArr(Integer.valueOf((String)Orig[j].get(0)).intValue(), Data.GetColumn()-1);
	}
		for(int k=0; k<ColumnLeng; k++){	// �m�[�h�Ƃ��č쐬����
			if(!(C_each[GaNo].get(k)).equals("0"))
			child[k] = new BTNode(Data.getColumnKind(GaNo,k), Orig[k], Classes[k]);
		}

	// �C���ꂼ��̎q�m�[�h�̗tor�m�[�h���m�F�i�ċA�j
		for(int i=0; i<ColumnLeng; i++){
			if(!(C_each[GaNo].get(i)).equals("0"))
			if(!Data.CLASSsearch(Orig[i])){		// �m�[�h�̏ꍇ
				child[i] = ConnectNode(child[i], Data, count[i], Orig[i]);
			}
		}
	// �D�e�m�[�h�Ɏq�A����Ȃǂ��Z�b�g���ĕԂ�
		return Node.Connect(Node.getNodeName(), child, NodeList, GaNo); 	// N1(right), N2(left)	
	}
///////////////////////////////////////

	public static void main(String args[]){

		int Case, Column, GainNUM=0, CaseNUM=0;
		BTNode TreeData, TreeCope;
		BCon AtrData;
		double gain=0.000, GetInfo=0.000;

//		try{

       	// �f�[�^�x�[�X����ǂݍ���
		System.out.println(">>> �@DB��莖��W����ǂݍ��݂܂��B");
		AtrData = new BCon(args[0], args[1]);
        	Case = AtrData.GetCase();
        	Column = AtrData.GetColumn();
		// ����W�A�����l�Ƒ����l�̐���\������
		AtrData.PrintArr();
		AtrData.PrintC_KindAndEach(AtrData.GetC_kind(), AtrData.GetC_each());

		ArrayList<String> List = new ArrayList<String>();
		for(int i=0; i<Case; i++){
			List.add(String.valueOf(i));
		}
	// ����؍쐬(�g���f�[�^�A����ԍ����X�g)
		System.out.println(">>> �A����W�����؂��쐬���܂��B");
		TreeData = ConnectTopNode(AtrData, List);
	// ����؂�\������
		BTree tree = new BTree(TreeData);
		System.out.println(">>> �B����؂�\�����܂��B");
		System.out.println(" << Decition Tree! >>"); System.out.println();
		tree.show();
/*		}
		catch(Exception e){
			System.out.println("��O�F"+e);
		}
*/	}
}

