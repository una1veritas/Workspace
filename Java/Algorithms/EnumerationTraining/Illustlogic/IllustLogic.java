import java.util.*;

public class IllustLogic{
	int rSize=5;
	int cSize=5;
	private Vector r[] ;		/* �c�̐��� */
	private Vector c[] ;		/* ���̐��� */
	private int graph[][];	/* �O���t�̔z�� */

	public IllustLogic(){
		rSize=5;
		cSize=5;
		r = new Vector[rSize];		/* �c�̐��� */
		c = new Vector[cSize];		/* ���̐��� */
		graph = new int[rSize][cSize];	/* �O���t�̔z�� */
		
		Random rand = new Random();	/* �����쐬�p�ϐ� */	
		int randomNumber;
		
		for (int i=0;i<rSize;i++){	/* �c�̔z��̗̈���m�� */
			this.r[i] = new Vector();
		}
		
		for (int j=0;j<cSize;j++){	/* ���̔z��̗̈���m�� */
			this.c[j] = new Vector();
		}

/* �K���Ȓl���c�Ɖ��̔z��ɑ�� */
		
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
/* ����I��� */

/* �\�̏����� */		
//		int this.graph[][] = new int[rSize][cSize];
		
		for (int rCount=0;rCount<rSize;rCount++){
			for (int cCount=0;cCount<cSize;cCount++){
				this.graph[rCount][cCount] = 0;
			}
		}
/* �������I��� */

	}
	
	public String toString(){
		String row;			/*�c�̃L�[�̏o�͂�����ϐ�*/
		String colum;		/*���̃L�[�̏o�͂�����ϐ�*/
		String graphString;	/*�\�̏o�͂�����ϐ�*/
		
/* �c�̃L�[���o�� */
		row = "�c�̃L�[ : \n";
		for(int rCount=0;rCount<rSize;rCount++){
			row += rCount + "��� : ";
			row += this.r[rCount].toString();
			row += "\n";
		}
		
/* ���̃L�[���o�� */
		colum = "���̃L�[ : \n";
		for(int cCount=0;cCount<cSize;cCount++){
			colum += cCount + "�s�� : ";
			colum += this.c[cCount].toString();
			colum += "\n";
		}
		
/* �\���o�� */
		graphString = "�\ : \n";
		for (int rCount=0;rCount<rSize;rCount++){
			for (int cCount=0;cCount<cSize;cCount++){
				if(this.graph[rCount][cCount] == 0){
					graphString += "��";
				} else {
					graphString += "��";
				}
			}
			graphString += "\n";
		}
		
/* �c�A���A�\�̏o�͂��Ȃ��Ė߂�l�Ƃ��� */
		return row+colum+graphString;
	}
}
		