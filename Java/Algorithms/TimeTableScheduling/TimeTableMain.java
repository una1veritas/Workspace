import java.net.BindException;
import java.net.ServerSocket;
import java.util.Date;
import java.sql.Timestamp;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;

import org.apache.commons.lang.time.StopWatch;

import data.SaveTimeTable;

/*
 * �쐬��: 2007/10/22
 *
 * TODO ���̐������ꂽ�t�@�C���̃e���v���[�g��ύX����ɂ͎��փW�����v:
 * �E�B���h�E - �ݒ� - Java - �R�[�h�E�X�^�C�� - �R�[�h�E�e���v���[�g
 */

/**
 * @author masayoshi
 *
 * TODO �u�`���Ԋ��쐬�̃��C��
 *
 */
public class TimeTableMain {
	private static int PORT = 3776;
	public static void main(String args[]) throws Exception{
        if (!isOnlyMe()) {
            System.out.println("���ɋN������Ă܂�");
            System.exit(-1);
        }
        System.out.println("�P�ƋN������܂���");
        //�쐬�����̎擾
        Timestamp create_date = new Timestamp(System.currentTimeMillis());
        SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMddkkmmss");
        String s_create_date = sdf.format(new Date(create_date.getTime()));
		TimeTableSchedulingTaskSeries tts = null;
		//serverhost:daisygw.daisy.ai.kyutech.ac.jp
		String host = "localhost",db = "timetable",user = "satonaka",
		pass = "asdfg",name = "valuatesql";
		//readType 1:�ԍ��t���r�p�k��name���̃t�@�C������ 2:�ԍ������r�p�k��name���̃t�@�C������
		//         3:name���̃f�[�^�x�[�X��̃e�[�u������ǂݏo��
		//mode 3:�Ȃ� 4:�v���W�F�N�g���l�� 5:�v���W�F�N�g���l��&&�A������u�`���l��
		int count = 1,readType = 3,mode = 4,list_size = 5,project_id = 1;
		//readType:3&&(mode:5or4)�ŋN��
		switch (args.length) {
		case 7:
			host = args[0];db = args[1];user = args[2];pass = args[3];project_id= Integer.parseInt(args[4]);
			count = Integer.parseInt(args[5]);list_size = Integer.parseInt(args[6]);
			break;
		case 6:
			host = args[0];db = args[1];user = args[2];pass = args[3];project_id= Integer.parseInt(args[4]);
			count = Integer.parseInt(args[5]);
			break;
		case 1:
			count = Integer.parseInt(args[0]);
			break;
		default: break;
		}
		//�]���ǂ���list_size�Ɏ��Ԋ��Ă����郊�X�g
		ArrayList list = new ArrayList(list_size);
		for(int i = 0; i< list_size;i++){
			list.add(new SaveTimeTable(null,999999,999999));
		}
		SaveTimeTable t_stt,best_sst;
		int count_offence = 999999,min_count_offence=99999,t_count_offence = 999999;
		int valuate = 999999,min_valuate = 99999,t_valuate=99999;
		tts = new TimeTableSchedulingTaskSeries(host,db,user,pass,name,readType,project_id,mode);
		//BEST���Ԋ����̓ǂݍ���(���ݖ��g�p)
		//�e�[�u��timetablesql�Ƀf�[�^��ǂݍ���(�v�]�̕]��)
		//tts.loadTimeTableAddProject("besttimetablesql",project_id);
		//�v���O������Ɏ��Ԋ�����ǂݍ���(����̕]��)
		//tts.getTimetable().loadTaskSeriesTimeTable();
		//min_valuate = tts.valuateTimeTableSQLDB();
		//min_count_offence = tts.count_offence();
		//BEST�̎��Ԋ����̕]���l��\��
		//System.out.println("besttimetablesql valueate:"+min_valuate);
		//System.out.println("besttimetablesql offence:"+min_count_offence);
		//best_sst = new SaveTimeTable(tts.getTimetable().getTaskSeriesTimeTableRowsCopy(),min_count_offence,min_valuate);
		
		//count�� ���Ԋ��Ă̐���
		for(int i = 1; i<=count; i++){
			//�������𐶐�
			if(mode==4){
				tts.initsolAddProject(project_id);
			}else if(mode==5){
				tts.initsolTaskSeriesAddProject(project_id);
				//tts.initsolNotRandomTaskSeriesAddProject(i,project_id);
			}
			//�������̕]���l(�ڍ�)��\��
			if(mode==5) System.out.println(tts.task_series_count_offenceDetail());
			else System.out.println(tts.count_offenceDetail());
			System.out.println(tts.valuateTimeTableSQLDBDetail());
			
			StopWatch sp = new StopWatch();
			sp.start();
			//�Ǐ��T���̎��s
			System.out.println("local_serch start");
			if(mode==5){
				tts.task_series_local_search();
			}else tts.local_search();
			System.out.println("local_serch end");
			sp.stop();
			System.out.println("����:"+sp);
			
			//���Ԋ��Ă̕]���l�v�Z
			if(mode==5) count_offence = tts.task_series_count_offenceDetail();
			else count_offence = tts.count_offenceDetail();
			valuate = tts.valuateTimeTableSQLDBDetail();
			//�]���l��\��
			System.out.println("valuate:"+valuate);
			System.out.println("count_offence:"+count_offence);
			
			//list���̎��Ԋ��Ă̓���ԕ]���̈������̂��擾
			t_stt=((SaveTimeTable)Collections.max(list));
			t_valuate = t_stt.getValuate();
			t_count_offence = t_stt.getCount_offence();
			//list�̍X�V
			if(count_offence < t_count_offence){
				int index = list.indexOf(new SaveTimeTable(null,t_count_offence,t_valuate));
				//mode:4�̎��@�o�O�����邩��
				list.set(index,new SaveTimeTable(tts.getTimetable().getTaskSeriesTimeTableRowsCopy(),count_offence,valuate));
			}else if(count_offence == t_count_offence && valuate < t_valuate){
				int index = list.indexOf(new SaveTimeTable(null,t_count_offence,t_valuate));
				//mode:4�̎��@�o�O�����邩��
				list.set(index,new SaveTimeTable(tts.getTimetable().getTaskSeriesTimeTableRowsCopy(),count_offence,valuate));
			}
			/*BEST���Ԋ����̍X�V
 			if(count_offence < min_count_offence){
				min_count_offence = count_offence;
				min_valuate = valuate;
				best_sst=new SaveTimeTable(tts.getTimetable().getTaskSeriesTimeTableRowsCopy(),min_count_offence,min_valuate);
			}else if(count_offence == min_count_offence && valuate < min_valuate){
				min_valuate = valuate;
				best_sst=new SaveTimeTable(tts.getTimetable().getTaskSeriesTimeTableRowsCopy(),min_count_offence,min_valuate);
			}*/
		}
		Collections.sort(list);
		Iterator i = list.iterator();
		int index = 1;
		while(i.hasNext()){
			SaveTimeTable stt = (SaveTimeTable)i.next();
			if(stt.getTimetablerows()!=null){
				//DBLog�̂���
				//sst�̃f�[�^���c�a���timetablesql�ɓǂݍ���
				stt.saveTimetablesql(tts.getCmanager());
				//�]���t�@�C���̕ۑ��F"����,project_id,���ԍ�.txt"
				tts.fileTimeTableSQLDBLog("./result/"+s_create_date+""+project_id+""+index+".txt");
				tts.saveTimeTableAddProject("timetable",project_id,index,create_date);
			}
			index++;
		}
		//BEST���Ԋ����̕ۑ�
		//best_sst.saveTimetablesql(tts.getCmanager());
		//tts.fileTimeTableSQLDBLog("./result/besttimetablesql"+project_id+".txt");
		//tts.saveTimeTableAddProject("besttimetablesql",project_id);
		tts.close();
		return;
	}
	
	public static boolean isOnlyMe() {
        try {
            ServerSocket sock = new ServerSocket(PORT);
        } catch (BindException e) {
            return false;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
        return true;
    }
}
