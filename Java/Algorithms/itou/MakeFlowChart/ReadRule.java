import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;

public class ReadRule {
    public static String[][] Rule;	// テキストファイルから読み込んだ既知の規則
    public static ArrayList[] Att;	// 属性と属性値の関係
    public static ArrayList<ArrayList<String>> Examples = new ArrayList<ArrayList<String>>();	// 生成した事例集
    public static int lineCount=0;	// 行数

    public static String[][] getRule(){		// Ruleを返す
	return Rule;
    }
    public static ArrayList[] getAtt(){		// Attを返す
	return Att;
    }
    public static ArrayList getExample(){		// Attを返す
	return Examples;
    }
    public static String getRuleValue(int x, int y){	// Rule[x][y]の値を返す
	return Rule[x][y];
    }
    public static String getAttValue(int x, int y){	// Att[x][y]の値を返す
	return (String)Att[x].get(y);
    }

    // setAtt(): Attribute[][]から使いやすいAtt[][]に変換する
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
	System.out.println("＊与えられた規則: Rule[][]");
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
    // PrintAtt(): 各属性値を表すAtt[][]を表示する
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

	// 属性値をDBから読み込む
	Attr AttData = new Attr(DBname);
	AttData.SetData();	// 属性、属性値セット
//	AttData.PrintAttribute();
        try {
	// 行数と規則の行ごとの読み込み: RuleLine[], lineCount
            BufferedReader reader = 
                      new BufferedReader( new FileReader(fileName) );
            while(true) {
		String line = reader.readLine();
                if(line==null) break;
                RuleLine.add(line);
                lineCount++;
            }
            reader.close();
	// 規則の文字単位での格納: Rule[][]
	    Rule = new String[lineCount][];
	    for(int j=0; j<RuleLine.size(); j++){
		String str = new String(RuleLine.get(j));
		String[] RuleReco = str.split(" ");
		Rule[j] = RuleReco;
	    }
//	    PrintRule(Rule);
	    setAtt(Rule, AttData);
//	    PrintAtt();

	    // 事例生成の準備
	    Exam ExamData = new Exam(Rule);
	    Exam.SetExam(ExamData, Rule, Att);
	    System.out.println();
	    Examples = Exam.getExample();

        } catch(FileNotFoundException e) {
            System.out.println("ファイルがありません");
        } catch(IOException e) {
            System.out.println("入出力エラーです");
        }
	return Examples;
    }
}

