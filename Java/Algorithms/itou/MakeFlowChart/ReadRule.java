import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;

public class ReadRule {
    public static String[][] Rule;	// �e�L�X�g�t�@�C������ǂݍ��񂾊��m�̋K��
    public static ArrayList[] Att;	// �����Ƒ����l�̊֌W
    public static ArrayList<ArrayList<String>> Examples = new ArrayList<ArrayList<String>>();	// ������������W
    public static int lineCount=0;	// �s��

    public static String[][] getRule(){		// Rule��Ԃ�
	return Rule;
    }
    public static ArrayList[] getAtt(){		// Att��Ԃ�
	return Att;
    }
    public static ArrayList getExample(){		// Att��Ԃ�
	return Examples;
    }
    public static String getRuleValue(int x, int y){	// Rule[x][y]�̒l��Ԃ�
	return Rule[x][y];
    }
    public static String getAttValue(int x, int y){	// Att[x][y]�̒l��Ԃ�
	return (String)Att[x].get(y);
    }

    // setAtt(): Attribute[][]����g���₷��Att[][]�ɕϊ�����
    public static void setAtt(String[][] Rule, Attr data){
	int ValueNUM = Rule[0].length;

	Att = new ArrayList[ValueNUM];
	for(int i=0; i<ValueNUM; i++){
	     Att[i] = new ArrayList<String>();
	     for(int j=0; j<data.getCase(); j++){
	          if(data.getAttribute(j,0).equals("test"+(i))){
		       Att[i].add(data.getAttribute(j,1));
		  }
	     }
	}
    }
    public static void PrintRule(String[][] Rule){
	System.out.println("���^����ꂽ�K��: Rule[][]");
	System.out.print("       ");
	for(int i=1; i<=Rule[0].length; i++){
	     System.out.print("|test"+i+" |");
	}
	System.out.println();	System.out.print("       ");
	for(int i=0; i<Rule[0].length; i++){
	     System.out.print("--------");
	}
	System.out.println();
	for(int k=0; k<lineCount; k++){
	     System.out.print("Rule<"+(k+1)+">:");
    	     for(int j=0; j<Rule[k].length; j++){
	         System.out.print(Rule[k][j]+" ");
	     }
	     System.out.println();
	}
    }
    // PrintAtt(): �e�����l��\��Att[][]��\������
    public static void PrintAtt(){
	for(int i=0; i<Rule[0].length; i++){
	     System.out.print("Att["+(i)+"]: ");
	     for(int j=0; j<Att[i].size(); j++){
	          System.out.print(Att[i].get(j)+",");
             }
             System.out.println();
	}
    }

    public static ArrayList RuleAndAttribute(String fileName, String DBname) {

	ArrayList<String> RuleLine = new ArrayList<String>();

	// �����l��DB����ǂݍ���
	Attr AttData = new Attr(DBname);
	AttData.SetData();	// �����A�����l�Z�b�g
//	AttData.PrintAttribute();
        try {
	// �s���ƋK���̍s���Ƃ̓ǂݍ���: RuleLine[], lineCount
            BufferedReader reader = 
                      new BufferedReader( new FileReader(fileName) );
            while(true) {
		String line = reader.readLine();
                if(line==null) break;
                RuleLine.add(line);
                lineCount++;
            }
            reader.close();
	// �K���̕����P�ʂł̊i�[: Rule[][]
	    Rule = new String[lineCount][];
	    for(int j=0; j<RuleLine.size(); j++){
		String str = new String(RuleLine.get(j));
		String[] RuleReco = str.split(" ");
		Rule[j] = RuleReco;
	    }
//	    PrintRule(Rule);
	    setAtt(Rule, AttData);
//	    PrintAtt();

	    // ���ᐶ���̏���
	    Exam ExamData = new Exam(Rule);
	    Exam.SetExam(ExamData, Rule, Att);
	    System.out.println();
	    Examples = Exam.getExample();

        } catch(FileNotFoundException e) {
            System.out.println("�t�@�C��������܂���");
        } catch(IOException e) {
            System.out.println("���o�̓G���[�ł�");
        }
	return Examples;
    }
}

