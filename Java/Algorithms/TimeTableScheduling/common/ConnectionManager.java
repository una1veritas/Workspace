package common;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;


/**
 * �R�l�N�V�����Ǘ��N���X
 */
public class ConnectionManager {
	
	/**
	 * JDBC�h���C�o�[�̃N���X��
	 */
	private String driver;
	/**
	 * �f�[�^�x�[�X��URL
	 */
	private String url;
	/**
	 * �f�[�^�x�[�X�̃��[�U
	 */
	private String user;
	/**
	 * �f�[�^�x�[�X�̃p�X���[�h
	 */
	private String pass;
	
	public ConnectionManager(String url,String user,String pass){
		driver = "com.mysql.jdbc.Driver";
		this.url = url;
		this.user = user;
		this.pass = pass;
	}
	/**
	 * Conncection���擾���܂��B
	 */	
	public Connection getConnection() throws SQLException{
		try {
			Class.forName(driver).newInstance();
		} catch (Exception e) {
			e.printStackTrace();
		}
		Connection con = DriverManager.getConnection(url,user,pass);
		return con;
	}
}
