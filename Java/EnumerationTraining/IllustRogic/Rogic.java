import java.util.Vector;

/*
 * �쐬��: 2005/07/25
 *
 * ���̐������ꂽ�R�����g�̑}�������e���v���[�g��ύX���邽��
 * �E�B���h�E > �ݒ� > Java > �R�[�h���� > �R�[�h�ƃR�����g
 */

/**
 * @author mokomoko
 *
 * ���̐������ꂽ�R�����g�̑}�������e���v���[�g��ύX���邽��
 * �E�B���h�E > �ݒ� > Java > �R�[�h���� > �R�[�h�ƃR�����g
 */
public class Rogic {
	public Vector r[], c[];
	public int group[][];
	
	public Rogic(){
		//Vector�̏�����
		r = new Vector[3];
		c = new Vector[3];
		
		r[0] = new Vector();
		r[0].addElement("1");
/*		
		r1.addElement("1");
		r1.addElement("1");
		r2.addElement("0");
		r2.addElement("0");
		r3.addElement("0");
		r3.addElement("0");
		c1.addElement("0");
		c1.addElement("0");
		c2.addElement("0");
		c2.addElement("0");
		c3.addElement("0");
		c3.addElement("0");
*/		 
		//group�̏�����
		group = new int[3][3];
		
		group[0][0] = 0;
		group[0][1] = 0;
		group[0][2] = 0;
		group[1][0] = 0;
		group[1][1] = 0;
		group[1][2] = 0;
		group[2][0] = 0;
		group[2][1] = 0;
		group[2][2] = 0;
		
		
	}
	
		
	public void print(){
		Object value;
		System.out.println("�s�ɂ����Ă̓h����̐��̗�");
		for(int j = 0; j < 3; j++){
			System.out.print(j+1 + "�s��:");
			for(int i = 0; i < r1.size(); i++){
				System.out.print(r1.get(i) + ", ");
			}
			System.out.println();
		}
		System.out.println("��ɂ����Ă̓h����̐��̗�");
		for(int j = 0; j < 3; j++){
			System.out.print(j+1 + "���:");
			for(int i = 0; i < r1.size(); i++){
				System.out.print(r1.get(i) + ", ");
			}
			System.out.println();
		}
		System.out.println();
		System.out.println("�\");
		for(int i = 0; i < 3; i++){
			for(int j = 0; j < 3; j++){
				if(group[i][j] == 0){
					System.out.print("��");
				}
				else	
				System.out.print("��");
			}
			System.out.println();
		}
	}
		
	public static void main(String[] args) {
		Rogic rogic = new Rogic();
		
		rogic.print();
	
	}
}
