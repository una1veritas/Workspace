import java.sql.*;
import java.lang.Math;
import java.util.ArrayList;
import java.util.Arrays;


public class BCon extends ReadRule{

  private static ArrayList<ArrayList<String>> Arr = new ArrayList<ArrayList<String>>();	// 事例集
  private ArrayList[] ColumnKind;		// 各属性の種類
  private ArrayList[] ColumnEach;	// 各属性値の種類の数
  private int C_EachNUM[];		// 各属性の取りうる値の種類の数
  String DBname, RuleName;		// Database名, 規則のテキストファイル
  private int CaseNUM=0, ColumnNUM=0;	// 事例数、属性数(質問数)


////// 値を返す関数 //////
  public int GetCase(){ return CaseNUM; } 	// 事例数を返す
  public int GetColumn(){ return ColumnNUM; } 	// 属性数を返す
  public ArrayList[] GetC_kind() { return ColumnKind; }
  public ArrayList[] GetC_each() { return ColumnEach; }
  public int[] GetCENUM() { return C_EachNUM; }


  public int GetC_EachNUM(int i){ return C_EachNUM[i]; }

  public String getArr(int x, int y){		// arr[x][y]の値を返す
	if(GetCase()>x && GetColumn()>y) return (String)(Arr.get(x)).get(y);
 	else return "Empty";
  }
  // PrintArr(): 事例集Arrを表示する
  public static void PrintArr(){
	for(int k=0; k<Arr.size(); k++){
	    System.out.print("Arr<"+(k)+">:");
    	    for(int j=0; j<(Arr.get(k)).size(); j++){
		System.out.print((Arr.get(k)).get(j)+" ");
	    }
	    System.out.println();
	}
	System.out.println();
  }

  // 属性値を表示する
  public void PrintC_Kind(ArrayList[] Kind){
	for(int i=0; i<Kind.length; i++){
	    System.out.print("Kind["+i+"]: ");
	    for(int j=0; j<Kind[i].size(); j++){
		System.out.print(Kind[i].get(j)+" ");
	    }
    	    System.out.println("");
	}
  }
  // 属性値の数を表示する
  public void PrintC_Each(ArrayList[] Each){
	for(int i=0; i<Each.length; i++){
	    System.out.print("Each["+i+"]: ");
	    for(int j=0; j<Each[i].size(); j++){
		System.out.print(Each[i].get(j)+" ");
	    }
    	    System.out.println("");
	}
  }
  // 属性値と属性値の数を表示する
  public void PrintC_KindAndEach(ArrayList[] Kind, ArrayList[] Each){
	for(int i=0; i<GetColumn(); i++){
	    System.out.print("("+i+")の種類と数: ");
	    for(int j=0; j<Kind[i].size(); j++){
		System.out.print(Kind[i].get(j)+"("+Each[i].get(j)+")  ");
	    }
    	    System.out.println("");
	}
	System.out.println("");
  }
  public String getColumnKind(int x, int y){		// ColumnKind[x][y]の値を返す
 	if((ColumnKind[x].get(y)).equals("NULL")) return "Enpty";
	else if(GetColumn()>x && GetCase()>y) return (String)ColumnKind[x].get(y);
        else return "Empty";
  }
  public String getColumnEach(int x, int y){		// ColumnEach[x][y]の値を返す
 	if((ColumnEach[x].get(y)).equals("NULL")) return "Enpty";
	else if(GetColumn()>x && GetCase()>y) return (String)ColumnEach[x].get(y);
        else return "Empty";
  }

///////// データベースの初期情報取得 /////////

  public BCon(String value1, String value2){		
	try{
	    DBname = value1;
	    RuleName = value2;

	    // 事例集(Arr)、事例数(CaseNUM)、属性数(ColumnNUM)をセット
	    Arr = RuleAndAttribute(RuleName, DBname);
	    CaseNUM = Arr.size();
	    ColumnNUM = (Arr.get(0)).size();

	    // 各属性の種類を数を取得する
      	    C_EachNUM = new int[GetColumn()];
	    ColumnKind = getAtt();
	    ColumnEach = new ArrayList[GetColumn()];
	    // 属性値の種類(ColumnEach)を初期化 ＆ C_EachNUM[]のセット
	    for(int k=0; k<GetColumn(); k++){
		ColumnEach[k] = new ArrayList<Integer>();
		C_EachNUM[k] = ColumnKind[k].size();
	    }
	    // ColumnEach[][]のセット
	    int count=0;
	    for(int i=0; i<ColumnKind.length; i++){		// 各属性ごとに
		for(int j=0; j<ColumnKind[i].size(); j++){	// 各属性値で
		    for(int k=0; k<GetCase(); k++){		// その属性値をとる事例の数を調べる
			if((ColumnKind[i].get(j)).equals((Arr.get(k)).get(i))){
			    count++;
			}
		    }
		ColumnEach[i].add(String.valueOf(count));
		count = 0;
		}
	    }

	} catch (Exception e) {
	    e.printStackTrace();
	}
  }


// リスト内のクラスがすべて同じなら真、違えば偽。
  public boolean CLASSsearch(ArrayList list){
   	String element = (String)(getArr(Integer.valueOf((String)list.get(0)).intValue(), GetColumn()-1));

	for(int i=1; i<list.size(); i++){
      		if(((String)list.get(i)).equals(null)){ break; }
      		else{
        		String check = (String)(getArr(Integer.valueOf((String)list.get(i)).intValue(), GetColumn()-1));
        		if(check.equals(element)){	}
        		else { return false; }
      		}
    	}
    	return true;
  }

// i番目のレコードがNodeListに含まれているか？
  public boolean IncludeCheck(int TestNo, ArrayList NodeList){
	String Test = String.valueOf(TestNo);

	for(int j=0; j<NodeList.size(); j++){
		String NodeElm = (String)NodeList.get(j);
		if(Test.equals(NodeElm)) { return true; }
	}
	return false;
  }

// クラスyesの数を計算
// テストTestNo番目でvalueをとる事例の中で、クラスがYESをとるものの数は？
  public int GetYesClass(int TestNo, String value, ArrayList NodeList){
  	int YesCount=0, x=0;
    	for(int i=0; i<CaseNUM; i++){
		if(IncludeCheck(i, NodeList)){
			if(((Arr.get(i)).get(TestNo)).equals(value)){
        			if(((Arr.get(i).get(GetColumn()-1))).equals("yes")){
          				YesCount++;
        			}
			}
		}
    	}
    	return YesCount;
  }

// 利得比基準計算
  // Info(T) = Σ-a/b * log[2](a/b)
  // InfoX(T) = Σc/b * (-d/c*log[2](d/c) -e/c*log[2](e/c))
  // SplitInfo(T) = Σ-f/b * log[2](f/b)
  // GainRatio = Info(T) - InfoX(t)/ SplitInfo(T)
  public double GetGainRatio(int TestNo, ArrayList[] C_Kind, ArrayList[] C_Each, ArrayList NodeList){
    	double a = 0.0, c = 0.0, d = 0.0, e = 0.0, f = 0.0;
   	double Info=0.0, InfoX=0.0, SplitInfo=0.0, Gain =0.0, GainRatio=0.0;

   	double b = NodeList.size();
    	int C_NUM = GetColumn() -1;
//	PrintC_KindAndEach(C_Kind, C_Each);
	try{
		
	     	 // 全体のメッセージの平均情報量：Info(T)
     		for(int j=0; j<C_Kind[C_NUM].size(); j++){
     			a = Double.parseDouble((String)(C_Each[C_NUM].get(j)));
 	    		Info += -a/b * ((Math.log(a)/Math.log(2.0)) - (Math.log(b)/Math.log(2.0)));
//	System.out.print("a="+a+", b="+b+"//");
      		}

    	  	// 部分集合に分割したときの情報量：InfoX(T)
      		for(int k=0; k<GetC_EachNUM(TestNo); k++){		// test(i)について
        		c = Double.parseDouble((String)(C_Each[TestNo].get(k)));
        		d = (double)GetYesClass(TestNo, (String)C_Kind[TestNo].get(k), NodeList);
        		e = c - d;
//	System.out.println("TestNo:"+TestNo+"<KIND:"+k+"> c="+c+", YES="+d+", NO="+e);
			if(c != 0.0){
        			if(d == 0.0){ InfoX += c/b * (-e/c*(Math.log(e)/Math.log(2.0)-(Math.log(c)/Math.log(2.0)) )); }
        			else if(e == 0.0){ InfoX += c/b * (-d/c*(Math.log(d)/Math.log(2.0)-(Math.log(c)/Math.log(2.0)) )); }
        			else{ InfoX += c/b * (-d/c*(Math.log(d)/Math.log(2.0)-(Math.log(c)/Math.log(2.0))) 
               	                -e/c*(Math.log(e)/Math.log(2.0)-(Math.log(c)/Math.log(2.0)))); }
			}
     		}
		// 分割によって得られる全情報量：SplitInfo(T)
		for(int n=0; n<GetC_EachNUM(TestNo); n++){
			f = Double.parseDouble((String)(C_Each[TestNo].get(n)));
			if(f != 0.0){
				SplitInfo += -f/b * ((Math.log(f)/Math.log(2.0)) - (Math.log(b)/Math.log(2.0)));
			}
		}
//		System.out.println("Info:"+Info);
//		System.out.println("InfoX:"+InfoX);
//		System.out.println("SplitInfo:"+SplitInfo);
		Gain = Info - InfoX;
		if(SplitInfo == 0.0 || Gain < 0.0) { return 0.0; }
		GainRatio = (Info-InfoX)/SplitInfo;
	} catch (ArithmeticException err) {
		System.out.println("0で除算はできません。");
	} catch (NumberFormatException err) {
		System.out.println("例外：" + err);
		System.out.println("引数を整数で入力してください。");
	}

	return GainRatio;
  }

  public void WriteRecord(ArrayList[] Each){
	for(int i=0; i<ColumnNUM; i++){
        	System.out.print("("+i+")の種類と数: ");
		for(int j=0; j<Each[i].size(); j++){
			String elm = (String)Each[i].get(j);
          		System.out.print(ColumnKind[i].get(j)+"("+elm+")   ");
        	}
		System.out.println("");
	}
  }


}
//ファイルの終了
