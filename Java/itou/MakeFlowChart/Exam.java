import java.sql.*;
import java.util.ArrayList;
import java.util.Arrays;

public class Exam{
    // Example: ����W���ρ~�ρ�
    private static ArrayList<ArrayList<String>> Example = new ArrayList<ArrayList<String>>();	// ������������W

    public static ArrayList getExample(){	// ����W��Ԃ�
	return Example;
    }

    // Exam(): Rule[][]�����炢�A����WExample�ɃR�s�[����
    public Exam(String Rule[][]){
	int AttrNUM = Rule.length;
	int ValueNUM = Rule[0].length;

	for(int i=0; i<Rule.length; i++){
	    ArrayList<String> exam = new ArrayList<String>();
	    for(int j=0; j<ValueNUM; j++){
		exam.add(Rule[i][j]);
	    }
	    Example.add(exam);
	}
    }
    // SetExam(): �e���Ⴉ��"*"��T���A���̕������玖�ᐔ�𑝂₷
    public static void SetExam(Exam ExamData, String[][] Rule, ArrayList[] Att){
	LOOP1:
	for(int i=0; i<Example.size(); i++){
	    for(int j=0; j<(Example.get(i)).size(); j++){
		if(((Example.get(i)).get(j)).equals("*")){
		    MakeRule(i,j,ExamData, Att);
		    i=-1; j=-1;
		    continue LOOP1;
		}
	    }
	}
    }
    // PrintExam(): ����WExample��\������
    public static void PrintExam(){
	for(int k=0; k<Example.size(); k++){
	     System.out.print("Exam<"+(k)+">:");
    	     for(int j=0; j<(Example.get(k)).size(); j++){
	         System.out.print((Example.get(k)).get(j)+" ");
	     }
	     System.out.println();
	}
    }
    // MakeRule(): SetExam()�Œ��ׂ��ʒu�̒l��Example���C������
    public static void MakeRule(int AstRuleNUM, int AstValueNUM, Exam ExamData, ArrayList[] att){

	for(int count=0; count<att[AstValueNUM].size(); ){
	    ArrayList<String> elm = new ArrayList<String>();
	    for(int i=0; i<((ExamData.Example).get(AstRuleNUM)).size(); i++){
		if(i == AstValueNUM){
		    elm.add( (String)(att[AstValueNUM].get(count)) );
		    count++;
		} else {
		    elm.add(((ExamData.Example).get(AstRuleNUM)).get(i));
		}
	    }
	    (ExamData.Example).add(elm);
	}
	(ExamData.Example).remove(AstRuleNUM);
    }

}
