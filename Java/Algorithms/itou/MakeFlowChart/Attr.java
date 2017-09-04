import java.sql.*;

public class Attr{
  private int Case=0;	// ATTR�̃��R�[�h��
  private String[][] Attribute;	// �����Ƒ����l�̃f�[�^
  public String DBname;
	
  public int getCase(){ return Case; } 	// ���R�[�h����Ԃ�
  public String getAttribute(int x, int y){		// Attribute[x][y]�̒l��Ԃ�
	return Attribute[x][y];
  }
///////// �f�[�^�x�[�X�̏������擾 /////////
  public Attr(String value){		
	try{
		DBname = value;
		Class.forName("com.mysql.jdbc.Driver").newInstance();

		Connection con3 = DriverManager.getConnection("jdbc:mysql://localhost/"+DBname, "root", "shin");
		Statement stmt3 = con3.createStatement();
		String sql3 = "SELECT COUNT(*) FROM ATTR";
		ResultSet rs3 = stmt3.executeQuery(sql3);

		// ���ᐔ�擾
		while(rs3.next()){ Case = Integer.valueOf(rs3.getString("count(*)")); }

		// �f�[�^�x�[�X�����
		stmt3.close();
		con3.close();
	} catch (Exception e) {
		e.printStackTrace();
	}
  }


////////// �f�[�^�x�[�X�̒l���Z�b�g /////////
  public void SetData() {			
	try {
		Attribute = new String[getCase()][2];
		Class.forName("com.mysql.jdbc.Driver").newInstance();

		Connection con2 = DriverManager.getConnection("jdbc:mysql://localhost/"+DBname, "root", "shin");
		Statement stmt2 = con2.createStatement();
		String sql2 = "SELECT * FROM ATTR";
		ResultSet rs2 = stmt2.executeQuery(sql2);

		// �������ꂽ�s�������[�v
		for(int i=0; rs2.next(); i++){
			// �����P�`ColumnNUM�܂�
          		Attribute[i][0] = rs2.getString(1);
			Attribute[i][1] = rs2.getString(2);
      		}
      		// �f�[�^�x�[�X����ؒf
      		stmt2.close();
      		con2.close();

	} catch (Exception e) {
      	e.printStackTrace();
    	}
  }
  public void PrintAttribute(){
	System.out.println("�������A�����l�̎��l: Attribute[Case][2]");
	for(int i=0; i<getCase(); i++){
		for(int j=0; j<2; j++){
			System.out.print(getAttribute(i,j)+" ");
		}
		System.out.println();
	}
  }
}
