import java.sql.*;
import java.util.ArrayList;
import java.util.Arrays;

public class Exam{
    // Example: 事例集＜可変×可変＞
    private static ArrayList<ArrayList<String>> Example = new ArrayList<ArrayList<String>>();	// 生成した事例集

    public static ArrayList getExample(){	// 事例集を返す
	return Example;
    }

    // Exam(): Rule[][]をもらい、事例集Exampleにコピーする
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
    // SetExam(): 各事例から"*"を探し、その部分から事例数を増やす
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
    // PrintExam(): 事例集Exampleを表示する
    public static void PrintExam(){
	for(int k=0; k<Example.size(); k++){
	     System.out.print("Exam<"+(k)+">:");
    	     for(int j=0; j<(Example.get(k)).size(); j++){
	         System.out.print((Example.get(k)).get(j)+" ");
	     }
	     System.out.println();
	}
    }
    // MakeRule(): SetExam()で調べた位置の値でExampleを修正する
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
