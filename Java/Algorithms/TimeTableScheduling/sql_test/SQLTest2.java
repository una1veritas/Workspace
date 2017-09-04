package sql_test;
import java.sql.CallableStatement;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

import org.apache.commons.lang.time.StopWatch;

import common.ConnectionManager;

/*
 * �쐬��: 2007/05/24
 *
 * TODO ���̐������ꂽ�t�@�C���̃e���v���[�g��ύX����ɂ͎��փW�����v:
 * �E�B���h�E - �ݒ� - Java - �R�[�h�E�X�^�C�� - �R�[�h�E�e���v���[�g
 */

/**
 * @author �Y�����Y
 *
 * TODO ���̐������ꂽ�^�R�����g�̃e���v���[�g��ύX����ɂ͎��փW�����v:
 * �E�B���h�E - �ݒ� - Java - �R�[�h�E�X�^�C�� - �R�[�h�E�e���v���[�g
 */
public class SQLTest2 {
	public static void main(String args[]) throws Exception{
		ConnectionManager cmanager = new ConnectionManager("jdbc:mysql://localhost/schedule?useUnicode=true&characterEncoding=sjis","yukiko","nishimura");
		StopWatch sp = new StopWatch();
		Connection con = null;
		CallableStatement smt = null;
		int penalties = 0;
		int mode = 1;
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.prepareCall("call getPenalties4()");
			sp.start();
			for(int n = 0; n < 100; n++){
				if(mode==1){
					smt.execute();
				}else if(mode==2){
				}else{
				}
			}
			sp.stop();
			System.out.println(sp);
		}catch(SQLException e) {
			System.out.println(e);
		}catch(Exception e){
			e.printStackTrace();
		}finally{
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				con.close();
			}
		}
	}
}
