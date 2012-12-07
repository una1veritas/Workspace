import java.sql.*;

public class Attr{
  private int Case=0;	// ATTRのレコード数
  private String[][] Attribute;	// 属性と属性値のデータ
  public String DBname;
	
  public int getCase(){ return Case; } 	// レコード数を返す
  public String getAttribute(int x, int y){		// Attribute[x][y]の値を返す
	return Attribute[x][y];
  }
///////// データベースの初期情報取得 /////////
  public Attr(String value){		
	try{
		DBname = value;
		Class.forName("com.mysql.jdbc.Driver").newInstance();

		Connection con3 = DriverManager.getConnection("jdbc:mysql://localhost/"+DBname, "root", "shin");
		Statement stmt3 = con3.createStatement();
		String sql3 = "SELECT COUNT(*) FROM ATTR";
		ResultSet rs3 = stmt3.executeQuery(sql3);

		// 事例数取得
		while(rs3.next()){ Case = Integer.valueOf(rs3.getString("count(*)")); }

		// データベースを閉じる
		stmt3.close();
		con3.close();
	} catch (Exception e) {
		e.printStackTrace();
	}
  }


////////// データベースの値をセット /////////
  public void SetData() {			
	try {
		Attribute = new String[getCase()][2];
		Class.forName("com.mysql.jdbc.Driver").newInstance();

		Connection con2 = DriverManager.getConnection("jdbc:mysql://localhost/"+DBname, "root", "shin");
		Statement stmt2 = con2.createStatement();
		String sql2 = "SELECT * FROM ATTR";
		ResultSet rs2 = stmt2.executeQuery(sql2);

		// 検索された行数分ループ
		for(int i=0; rs2.next(); i++){
			// 属性１～ColumnNUMまで
          		Attribute[i][0] = rs2.getString(1);
			Attribute[i][1] = rs2.getString(2);
      		}
      		// データベースから切断
      		stmt2.close();
      		con2.close();

	} catch (Exception e) {
      	e.printStackTrace();
    	}
  }
  public void PrintAttribute(){
	System.out.println("＊属性、属性値の取る値: Attribute[Case][2]");
	for(int i=0; i<getCase(); i++){
		for(int j=0; j<2; j++){
			System.out.print(getAttribute(i,j)+" ");
		}
		System.out.println();
	}
  }
}
