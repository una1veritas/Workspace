/*
 * �쐬��: 2007/04/02
 *
 * TODO ���̐������ꂽ�t�@�C���̃e���v���[�g��ύX����ɂ͎��փW�����v:
 * �E�B���h�E - �ݒ� - Java - �R�[�h�E�X�^�C�� - �R�[�h�E�e���v���[�g
 */
package data;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Set;
import java.util.TreeSet;

import common.DBConnectionPool;

/**
 * @author masayoshi
 *
 * TODO �u�`���Ԋ��̏����Ǘ�
 *
 */
public class TimeTable {
	//(task_id,processor_id,period_id)�̑g�̏W��
	private LinkedList timetablerows;
	//�R�l�N�V�����v�[��
	private DBConnectionPool cmanager;
	//�S�u�t�̏��(���X�g�̒��gObject:Teacher)
	private ArrayList teachers;
	//�S�u�`�̏��(���X�g�̒��gObject:Lecture)
	private ArrayList lectures;
	//�S���Ԃ̏��(���X�g�̒��gObject:DayPeriod)
	private OrderDayPeriods day_periods;
	
	/**
	 * @return day_periods ��߂��܂��B
	 */
	public OrderDayPeriods getDay_periods() {
		return day_periods;
	}
	
	//timetablerows�̃R�s�[��Ԃ�
	public ArrayList getTimeTableRowsCopy(){
		ArrayList result = new ArrayList();
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			result.add(new TimeTableRow(t.getTask_id(),t.getPeriod_id(),t.getProcessor_id()));
		}
		return result;
	}
	
	//timetablerows�̃R�s�[��Ԃ�
	//�A������u�`��task_series_num��
	public ArrayList getTaskSeriesTimeTableRowsCopy(){
		ArrayList result = new ArrayList();
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			result.add(new TimeTableRow(t.getTask_id(),t.getPeriod_id(),t.getProcessor_id()));
			Lecture l = searchLecture(t.getTask_id());
			if(l.getTask_series_num()>1){
				int next_period_id = t.getPeriod_id();
				for(int num=1;num<l.getTask_series_num();num++){
					next_period_id = day_periods.getNextPeriods(next_period_id);
					result.add(new TimeTableRow(t.getTask_id(),next_period_id,t.getProcessor_id()));
				}
			}
		}
		return result;
	}
	
	//period_id��task_id��S�����Ă���processor_id�̃��X�g��Ԃ�
	public ArrayList getProcessors(int task_id, int period_id){
		ArrayList result = new ArrayList();
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			if((task_id==t.getTask_id())&&(period_id==t.getPeriod_id())){
				result.add(new Integer(t.getProcessor_id()));
			}
		}
		return result;
	}
	
	//���݂̎��Ԋ��̍u�`�Ƃ��̒S���l���ƊJ�u���Ԃ̃��X�g���擾
	public ArrayList getTimeTableTasksNum(){
		ArrayList result = new ArrayList();
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			int index = result.indexOf(new TaskPeriodNum(t.getTask_id(),t.getPeriod_id()));
			if(index==-1){
				result.add(new TaskPeriodNum(t.getTask_id(),t.getPeriod_id()));
			}else{
				 ((TaskPeriodNum)result.get(index)).updateNum();
			}
		}
		return result;
	}
	//���݂̎��Ԋ��̍u�`task_id�̒S���l�����擾
	public int getTimeTableTasks(int task_id){
		int count = 0;
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			if(t.getTask_id()==task_id){
				count++;
			}
		}
		return count;
	}
	//���݂̎��Ԋ����u�`�Ƃ��̊J�u���Ԃ̃��X�g�Ƃ��ĕԂ�
	public ArrayList getTaskPeriods(){
		ArrayList result = new ArrayList();
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			if(!result.contains(new TaskPeriod(t.getTask_id(),t.getPeriod_id()))){
				result.add(new TaskPeriod(t.getTask_id(),t.getPeriod_id()));
			}
		}
		return result;
	}
	
	public TimeTable(DBConnectionPool cmanager, ArrayList teachers, ArrayList lectures) throws Exception{
		this.cmanager = cmanager;
		this.teachers = teachers;
		this.lectures = lectures;
		this.timetablerows = new LinkedList();
	}
	
	public TimeTable(DBConnectionPool cmanager, ArrayList teachers, ArrayList lectures,OrderDayPeriods day_periods) throws Exception{
		this.cmanager = cmanager;
		this.teachers = teachers;
		this.lectures = lectures;
		this.day_periods = day_periods;
		this.timetablerows = new LinkedList();
	}
	
	//timetablerows�̓��e��\��
	public void printTimeTable(){
		System.out.println("(task_id,period_id,processor_id)");
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			System.out.println(t.toString());
		}
	}
	//timetablerows�̓��e���f�[�^�x�[�X��insert�`���ŕ\��
	public void printInsertTimeTable(){
		System.out.println("(task_id,period_id,processor_id)");
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			System.out.println("insert into timetablesql(task_id,period_id,processor_id) values"+t.toString()+";");
		}
	}
	//timetablerows���N���A����
	public void clearTimeTable() throws Exception{
		timetablerows.clear();
	}
	
	//DB���timetablesql����f�[�^���쐬(�ύX�\��)
	public void loadTaskSeriesTimeTable() throws Exception{
		timetablerows.clear();
		ArrayList list = new ArrayList();
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		int task_id,period_id,processor_id;
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			//���݂̎��Ԋ��̎擾
			String sql = "select * from timetablesql";
			rs = smt.executeQuery(sql);
			while(rs.next()) {
				task_id = rs.getInt("task_id");
				period_id = rs.getInt("period_id");
				processor_id = rs.getInt("processor_id");
				Lecture l = searchLecture(task_id);
				int task_series_num=l.getTask_series_num();
				if(task_series_num>1){
					list.add(new TaskPeriod(task_id,period_id));
				}
				timetablerows.add(new TimeTableRow(task_id,period_id,processor_id));
			}
		}catch(SQLException e) {
			System.out.println(e);				
			throw e;
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		//DB��̃e�[�u���ɂ͘A������u�`�������Ă���
		//�A������u�`�̍폜
		Iterator iterator = list.iterator();
		while(iterator.hasNext()){
			TaskPeriod tp = (TaskPeriod)iterator.next();
			Lecture l = searchLecture(tp.getTask_id());
			int task_series_num = l.getTask_series_num();
			int next_period_id = tp.getPeriod_id();
			for(int num=1;num<task_series_num;num++){
				next_period_id = day_periods.getNextPeriods(next_period_id);
				if(!list.contains(new TaskPeriod(tp.getTask_id(),next_period_id))){
					deleteTaskTimeTable(tp.getTask_id(),tp.getPeriod_id());
				}
			}
		}
	}

	//DB���timetablesql����f�[�^���쐬
	public void loadTimeTable() throws Exception{
		timetablerows.clear();
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		int task_id,period_id,processor_id;
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			//���݂̎��Ԋ��̎擾
			String sql = "select * from timetablesql";
			rs = smt.executeQuery(sql);
			while(rs.next()) {
				task_id = rs.getInt("task_id");
				period_id = rs.getInt("period_id");
				processor_id = rs.getInt("processor_id");
				timetablerows.add(new TimeTableRow(task_id,period_id,processor_id));
			}
		}catch(SQLException e) {
			System.out.println(e);				
			throw e;
		}finally{
			//ResultSet�C���^�[�t�F�[�X�̔j��
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}
	
	public void insertTimeTableRow(int task_id,int period_id,int processor_id){
		timetablerows.add(new TimeTableRow(task_id,period_id,processor_id));
	}
	
	public void deleteTimeTableRow(int task_id,int period_id,int processor_id){
		timetablerows.remove(new TimeTableRow(task_id,period_id,processor_id));
	}
	
	public void updateProcessorTimeTable(int new_processor_id,int old_processor_id, int old_task_id, int old_period_id){
		timetablerows.remove(new TimeTableRow(old_task_id,old_period_id,old_processor_id));
		timetablerows.add(new TimeTableRow(old_task_id,old_period_id,new_processor_id));
	}
	
	public void deleteTaskTimeTable(int task_id){
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			if(t.getTask_id()==task_id) i.remove();
		}
	}
	
	public void deleteTaskTimeTable(int task_id, int period_id){
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			if((t.getTask_id()==task_id)&&(t.getPeriod_id()==period_id)) i.remove();
		}
	}
	
	//����ᔽ���Ă鐔��Ԃ�
	public int countOffence(){
		Set temp1 = new TreeSet();
		ArrayList temp2 = new ArrayList();
	    ArrayList temp3 = new ArrayList();
		//��1.�e�u�t�͒S���\���Ԃ�(�S���\�ȍu�`)��S��
		//2.�e�u�`�́A�e�u�`�ŗL�̊J�u�\���Ԃ̂�����ɊJ�u�����
		//��2-1.�e�u�`�͊J�u�\���ԂɊJ�u�����
		//��2-2.�e�u�`�͊J�u�\�ł���ΕK���J�u�����
		//��2-3.�e�u�`�͂���������J�u�����
		//��3.�e�u�t�́A�������Ԃɕ����̍u�`��S�����Ȃ�
		Iterator i = timetablerows.iterator();
		int count = 0;
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			Teacher teacher = searchTeacher(t.getProcessor_id());
			Lecture lecture = searchLecture(t.getTask_id());
			if(!teacher.isPeriod(t.getPeriod_id()))count++;
			if(!teacher.isQualification(lecture.getQualification_id()))count++;
			if(!lecture.isPeriod(t.getPeriod_id()))count++;
			//�J�u���Ă���u�`���𒲂ׂ�
			temp1.add(new Integer(t.getTask_id()));
			//�u�`�E���Ԃ̑g�ɂ܂Ƃ߂�i�d�����Ȃ��j
			if(!temp2.contains(new TaskPeriod(t.getTask_id(),t.getPeriod_id()))){
				temp2.add(new TaskPeriod(t.getTask_id(),t.getPeriod_id()));
			}
			//�u�t�E���Ԃ̑g�ɂ܂Ƃ߂�i�d�����Ȃ��j
			if(!temp3.contains(new ProcessorPeriod(t.getProcessor_id(),t.getPeriod_id()))){
				temp3.add(new ProcessorPeriod(t.getProcessor_id(),t.getPeriod_id()));
			}
		}
		if(temp1.size()<lectures.size()){
			count = count + (lectures.size()-temp1.size());
		}
		i = temp2.iterator();
		while(i.hasNext()){
			TaskPeriod t = (TaskPeriod)i.next();
			if(!isNotTasks3(t.getTask_id(),t.getPeriod_id()))count++;
		}
		
		i = temp3.iterator();
		while(i.hasNext()){
			ProcessorPeriod t = (ProcessorPeriod)i.next();
			if(!isNotTasks2(t.getProcessor_id(),t.getPeriod_id()))count++;
		}
		return count;
	}

	//����ᔽ���Ă鐔��Ԃ��i�ڍ׏��\���j
	public int countOffenceDetail(){
		Set temp1 = new TreeSet();
		ArrayList temp2 = new ArrayList();
	    ArrayList temp3 = new ArrayList();
		//��1.�e�u�t�͒S���\���Ԃ�(�S���\�ȍu�`)��S��
		//2.�e�u�`�́A�e�u�`�ŗL�̊J�u�\���Ԃ̂�����ɊJ�u�����
		//��2-1.�e�u�`�͊J�u�\���ԂɊJ�u�����
		//��2-2.�e�u�`�͊J�u�\�ł���ΕK���J�u�����
		//��2-3.�e�u�`�͂���������J�u�����
		//��3.�e�u�t�́A�������Ԃɕ����̍u�`��S�����Ȃ�
		Iterator i = timetablerows.iterator();
		int count = 0;
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			Teacher teacher = searchTeacher(t.getProcessor_id());
			Lecture lecture = searchLecture(t.getTask_id());
			if(!teacher.isPeriod(t.getPeriod_id())){
				System.out.println("�u�t:"+t.getProcessor_id()+"�͍u�`����:"+t.getPeriod_id()+"�ɒS���ł��܂���");
				count++;
			}
			//if(!teacher.isQualification(lecture.getQualification_id()))count++;
			if(!lecture.isPeriod(t.getPeriod_id())){
				System.out.println("�u�`:"+t.getTask_id()+"�͍u�`����:"+t.getPeriod_id()+"�ɊJ�u�ł��܂���");
				count++;
			}
			//�J�u���Ă���u�`���𒲂ׂ�
			temp1.add(new Integer(t.getTask_id()));
			//�u�`�E���Ԃ̑g�ɂ܂Ƃ߂�i�d�����Ȃ��j
			if(!temp2.contains(new TaskPeriod(t.getTask_id(),t.getPeriod_id()))){
				temp2.add(new TaskPeriod(t.getTask_id(),t.getPeriod_id()));
			}
			//�u�t�E���Ԃ̑g�ɂ܂Ƃ߂�i�d�����Ȃ��j
			if(!temp3.contains(new ProcessorPeriod(t.getProcessor_id(),t.getPeriod_id()))){
				temp3.add(new ProcessorPeriod(t.getProcessor_id(),t.getPeriod_id()));
			}
		}
		if(temp1.size()<lectures.size()){
			System.out.println("�J�u����ĂȂ��u�`��:"+(lectures.size()-temp1.size()));
			i = lectures.iterator();
			while(i.hasNext()){
				Lecture t = (Lecture)i.next();
				if(!temp1.contains(new Integer(t.getTask_id()))){
					System.out.println("�u�`:"+t.getTask_id()+"�͊J�u����Ă��܂���");
				}
			}
			count = count + (lectures.size()-temp1.size());
		}
		//�e�u�`�͂���������J�u�����
		i = temp2.iterator();
		while(i.hasNext()){
			TaskPeriod t = (TaskPeriod)i.next();
			if(!isNotTasks3(t.getTask_id(),t.getPeriod_id())){
				System.out.println("�u�`:"+t.getTask_id()+"�͍u�`����:"+t.getPeriod_id()+"�ȊO�̍u�`���Ԃɂ��J�u����Ă���");
				count++;
			}
		}
		//�e�u�t�́A�������Ԃɕ����̍u�`��S�����Ȃ�
		i = temp3.iterator();
		while(i.hasNext()){
			ProcessorPeriod t = (ProcessorPeriod)i.next();
			if(!isNotTasks2(t.getProcessor_id(),t.getPeriod_id())){
				System.out.println("�u�t:"+t.getProcessor_id()+"�͍u�`����:"+t.getPeriod_id()+"�ɕ����̍u�`��S�����Ă���");
				count++;
			}
		}
		return count;
	}
	//�A������u�`�ɑΉ�
	//����ᔽ���Ă鐔��Ԃ��i�ڍ׏��\���j
	public int taskSeriesCountOffenceDetail(){
		Set temp1 = new TreeSet();
		ArrayList temp2 = new ArrayList();
	    ArrayList temp3 = new ArrayList();
		//��1.�e�u�t�͒S���\���Ԃ�(�S���\�ȍu�`)��S��
		//2.�e�u�`�́A�e�u�`�ŗL�̊J�u�\���Ԃ̂�����ɊJ�u�����
		//��2-1.�e�u�`�͊J�u�\���ԂɊJ�u�����
		//��2-2.�e�u�`�͊J�u�\�ł���ΕK���J�u�����
		//��2-3.�e�u�`�͂���������J�u�����
		//��3.�e�u�t�́A�������Ԃɕ����̍u�`��S�����Ȃ�
		Iterator i = timetablerows.iterator();
		int count = 0;
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			Teacher teacher = searchTeacher(t.getProcessor_id());
			Lecture lecture = searchLecture(t.getTask_id());
			//if(!teacher.isQualification(lecture.getQualification_id()))count++;
			//�A������u�`�̈ᔽ�͘A�����鐔�{(�ᔽ�̕ύX)
			if(lecture.getTask_series_num()==1){
				if(!lecture.isPeriod(t.getPeriod_id())){
					System.out.println("�u�`:"+t.getTask_id()+"�͍u�`����:"+t.getPeriod_id()+"�ɊJ�u�ł��܂���");
					count++;
				}
				if(!teacher.isPeriod(t.getPeriod_id())){
					System.out.println("�u�t:"+t.getProcessor_id()+"�͍u�`����:"+t.getPeriod_id()+"�ɒS���ł��܂���");
					count++;
				}
				//�u�t�E���Ԃ̑g�ɂ܂Ƃ߂�i�d�����Ȃ��j
				if(!temp3.contains(new ProcessorPeriod(t.getProcessor_id(),t.getPeriod_id()))){
					temp3.add(new ProcessorPeriod(t.getProcessor_id(),t.getPeriod_id()));
				}
			}else{
				System.out.println("�u�`:"+t.getTask_id()+"�͍u�`����:"+t.getPeriod_id()+"����"+lecture.getTask_series_num()+"���ԘA�����ĊJ�u");
				if(!lecture.isPeriod(t.getPeriod_id())){
					System.out.println("�u�`:"+t.getTask_id()+"�͍u�`����:"+t.getPeriod_id()+"�ɊJ�u�ł��܂���");
					count++;
				}
				if(!teacher.isPeriod(t.getPeriod_id())){
					System.out.println("�u�t:"+t.getProcessor_id()+"�͍u�`����:"+t.getPeriod_id()+"�ɒS���ł��܂���");
					count++;
				}
				//�u�t�E���Ԃ̑g�ɂ܂Ƃ߂�i�d�����Ȃ��j
				if(!temp3.contains(new ProcessorPeriod(t.getProcessor_id(),t.getPeriod_id()))){
					temp3.add(new ProcessorPeriod(t.getProcessor_id(),t.getPeriod_id()));
				}
				int next_period_id = t.getPeriod_id();
				for(int num=1;num<lecture.getTask_series_num();num++){
					if(next_period_id<0)next_period_id = next_period_id-1;
					else next_period_id = day_periods.getNextPeriods(next_period_id);
					if(next_period_id==-1)next_period_id = -1*t.getPeriod_id()-1;
					if(!lecture.isPeriod(next_period_id)){
						System.out.println("�u�`:"+t.getTask_id()+"�͍u�`����:"+next_period_id+"�ɊJ�u�ł��܂���");
						count++;
					}
					if(!teacher.isPeriod(next_period_id)){
						System.out.println("�u�t:"+t.getProcessor_id()+"�͍u�`����:"+next_period_id+"�ɒS���ł��܂���");
						count++;
					}
					//�u�t�E���Ԃ̑g�ɂ܂Ƃ߂�i�d�����Ȃ��j
					if(!temp3.contains(new ProcessorPeriod(t.getProcessor_id(),next_period_id))){
						temp3.add(new ProcessorPeriod(t.getProcessor_id(),next_period_id));
					}
				}
			}
			//�J�u���Ă���u�`���𒲂ׂ�
			temp1.add(new Integer(t.getTask_id()));
			//�u�`�E���Ԃ̑g�ɂ܂Ƃ߂�i�d�����Ȃ��j
			if(!temp2.contains(new TaskPeriod(t.getTask_id(),t.getPeriod_id()))){
				temp2.add(new TaskPeriod(t.getTask_id(),t.getPeriod_id()));
			}
		}
		if(temp1.size()<lectures.size()){
			System.out.println("�J�u����ĂȂ��u�`��:"+(lectures.size()-temp1.size()));
			i = lectures.iterator();
			while(i.hasNext()){
				Lecture t = (Lecture)i.next();
				if(!temp1.contains(new Integer(t.getTask_id()))){
					System.out.println("�u�`:"+t.getTask_id()+"�͊J�u����Ă��܂���");
				}
			}
			count = count + (lectures.size()-temp1.size());
		}
		//�e�u�`�͂���������J�u�����
		//System.out.println("temp2");
		i = temp2.iterator();
		while(i.hasNext()){
			TaskPeriod t = (TaskPeriod)i.next();
			//System.out.println("task_id:"+t.getTask_id()+" period_id:"+t.getPeriod_id());
			if(!isNotTasks3(t.getTask_id(),t.getPeriod_id())){
				System.out.println("�u�`:"+t.getTask_id()+"�͍u�`����:"+t.getPeriod_id()+"�ȊO�̍u�`���Ԃɂ��J�u����Ă���");
				count++;
			}
		}
		//�e�u�t�́A�������Ԃɕ����̍u�`��S�����Ȃ�
		//System.out.println("temp3");
		i = temp3.iterator();
		while(i.hasNext()){
			ProcessorPeriod t = (ProcessorPeriod)i.next();
			//System.out.println("processor_id:"+t.getProcessor_id()+" period_id:"+t.getPeriod_id());
			if(!isNotTasks2(t.getProcessor_id(),t.getPeriod_id())){
				System.out.println("�u�t:"+t.getProcessor_id()+"�͍u�`����:"+t.getPeriod_id()+"�ɕ����̍u�`��S�����Ă���");
				count++;
			}
		}
		return count;
	}
	//�A������u�`�ɑΉ�
	//����ᔽ���Ă鐔��Ԃ�
	public int taskSeriesCountOffence(){
		Set temp1 = new TreeSet();
		ArrayList temp2 = new ArrayList();
	    ArrayList temp3 = new ArrayList();
		//��1.�e�u�t�͒S���\���Ԃ�(�S���\�ȍu�`)��S��
		//2.�e�u�`�́A�e�u�`�ŗL�̊J�u�\���Ԃ̂�����ɊJ�u�����
		//��2-1.�e�u�`�͊J�u�\���ԂɊJ�u�����
		//��2-2.�e�u�`�͊J�u�\�ł���ΕK���J�u�����
		//��2-3.�e�u�`�͂���������J�u�����
		//��3.�e�u�t�́A�������Ԃɕ����̍u�`��S�����Ȃ�
		Iterator i = timetablerows.iterator();
		int count = 0;
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			Teacher teacher = searchTeacher(t.getProcessor_id());
			Lecture lecture = searchLecture(t.getTask_id());
			//if(!teacher.isQualification(lecture.getQualification_id()))count++;
			//�A������u�`�̈ᔽ�͘A�����鐔�{(�ᔽ�̕ύX)
			if(lecture.getTask_series_num()==1){
				if(!teacher.isPeriod(t.getPeriod_id()))count++;
				if(!lecture.isPeriod(t.getPeriod_id()))count++;
				//�u�t�E���Ԃ̑g�ɂ܂Ƃ߂�i�d�����Ȃ��j
				if(!temp3.contains(new ProcessorPeriod(t.getProcessor_id(),t.getPeriod_id()))){
					temp3.add(new ProcessorPeriod(t.getProcessor_id(),t.getPeriod_id()));
				}
			}else{
				if(!teacher.isPeriod(t.getPeriod_id()))count++;
				if(!lecture.isPeriod(t.getPeriod_id()))count++;
				//�u�t�E���Ԃ̑g�ɂ܂Ƃ߂�i�d�����Ȃ��j
				if(!temp3.contains(new ProcessorPeriod(t.getProcessor_id(),t.getPeriod_id()))){
					temp3.add(new ProcessorPeriod(t.getProcessor_id(),t.getPeriod_id()));
				}
				int next_period_id = t.getPeriod_id();
				for(int num=1;num<lecture.getTask_series_num();num++){
					if(next_period_id<0)next_period_id = next_period_id-1;
					else next_period_id = day_periods.getNextPeriods(next_period_id);
					if(next_period_id==-1)next_period_id = -1*t.getPeriod_id()-1;
					if(!teacher.isPeriod(next_period_id))count++;
					if(!lecture.isPeriod(next_period_id))count++;
					//�u�t�E���Ԃ̑g�ɂ܂Ƃ߂�i�d�����Ȃ��j
					if(!temp3.contains(new ProcessorPeriod(t.getProcessor_id(),next_period_id))){
						temp3.add(new ProcessorPeriod(t.getProcessor_id(),next_period_id));
					}
				}
			}
			//�J�u���Ă���u�`���𒲂ׂ�
			temp1.add(new Integer(t.getTask_id()));
			if(!temp2.contains(new TaskPeriod(t.getTask_id(),t.getPeriod_id()))){
				temp2.add(new TaskPeriod(t.getTask_id(),t.getPeriod_id()));
			}
		}
		if(temp1.size()<lectures.size()){
			count = count + (lectures.size()-temp1.size());
		}
		//���݂̎��Ԋ����Ɋe�u�`��������J�u���Ȃ����ǂ���
		i = temp2.iterator();
		while(i.hasNext()){
			TaskPeriod t = (TaskPeriod)i.next();
			if(!isNotTasks3(t.getTask_id(),t.getPeriod_id()))count++;
		}
		//���݂̎��Ԋ����ɍu�t���������Ԃɕ����̍u�`��S�����Ȃ����ǂ���
		i = temp3.iterator();
		while(i.hasNext()){
			ProcessorPeriod t = (ProcessorPeriod)i.next();
			if(!isNotTasks2(t.getProcessor_id(),t.getPeriod_id()))count++;
		}
		return count;
	}
	
	//����ᔽ���Ă��Ȃ����ǂ���(���ݖ��g�p)
	//(�ᔽ���Ă��Ȃ��ꍇ:true ���Ă���:false)
	public boolean isNotOffence(){
		Set temp = new TreeSet();
		//��1.�e�u�t�͒S���\���ԂɒS���\�ȍu�`��S��
		//2.�e�u�`�́A�e�u�`�ŗL�̊J�u�\���Ԃ̂�����ɊJ�u�����
		//��2-1.�e�u�`�͊J�u�\���ԂɊJ�u�����
		//��2-2.�e�u�`�͊J�u�\�ł���ΕK���J�u�����
		//��2-3.�e�u�`�͂���������J�u�����
		//��3.�e�u�t�́A�������Ԃɕ����̍u�`��S�����Ȃ�
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			Teacher teacher = searchTeacher(t.getProcessor_id());
			Lecture lecture = searchLecture(t.getTask_id());
			if(!teacher.isPeriod(t.getPeriod_id()))return false;
			if(!teacher.isQualification(lecture.getQualification_id()))return false;
			if(!lecture.isPeriod(t.getPeriod_id()))return false;
			if(!isNotTasks2(t.getProcessor_id(),t.getPeriod_id()))return false;
			if(!isNotTasks3(t.getTask_id(),t.getPeriod_id()))return false;
			temp.add(new Integer(t.getTask_id()));
		}
		if(temp.size()!=lectures.size())return false;
		return true;
	}
	
	//�u�tprocessor_id��period_id�ɕ����̍u�`��S�����Ȃ����ǂ���
	//(�S�����Ȃ�:true �S������:false)
	//processor_id��period_id���P�����݂��Ă��Ȃ���
	public boolean isNotTasks(int processor_id, int period_id){
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			if((t.getProcessor_id()==processor_id)&&(t.getPeriod_id()==period_id)) return false;
		}
		return true;
	}
	
	//�u�tprocessor_id��period_id�ɕ����̍u�`��S�����Ȃ����ǂ���
	//(�S�����Ȃ�:true �S������:false)
	//processor_id��period_id�ɒS�����Ă���u�`�����X�P�������݂��Ă��Ȃ���
	public boolean isNotTasks2(int processor_id, int period_id){
		int count = 0;
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			if((t.getProcessor_id()==processor_id)&&(t.getPeriod_id()==period_id)){
				if(count==1)return false;
				count++;
			}
		}
		return true;
	}
	
	//���݂̎��Ԋ����ɍu�`��������J�u����Ă��Ȃ�
	//(�J�u����Ă��Ȃ�:true �J�u����Ă���:false)
	public boolean isNotTasks3(int task_id, int period_id){
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			if((t.getTask_id()==task_id)&&(t.getPeriod_id()!=period_id)){
				return false;
			}
		}
		return true;
	}
	
	public LinkedList getTimetablerows() {
		return timetablerows;
	}

	public void setTimetablerows(LinkedList timetablerows) {
		this.timetablerows = timetablerows;
	}

	public DBConnectionPool getCmanager() {
		return cmanager;
	}

	public void setCmanager(DBConnectionPool cmanager) {
		this.cmanager = cmanager;
	}
	
	//timetablerows�̃f�[�^��DB��̃e�[�u��timetablesql�ɕۑ�
	public void saveTimeTable() throws Exception{
		Connection con = null;
		PreparedStatement psmt = null;
		Statement smt = null;
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			smt.executeUpdate("delete from timetablesql");
			psmt = con.prepareStatement("insert into timetablesql(task_id,period_id,processor_id) values( ? , ? , ? )");
			Iterator i = timetablerows.iterator();
			while(i.hasNext()){
				TimeTableRow t = (TimeTableRow)i.next();
				psmt.setInt(1,t.getTask_id());
				psmt.setInt(2,t.getPeriod_id());
				psmt.setInt(3,t.getProcessor_id());
				psmt.executeUpdate();
				
			}
		}catch(SQLException e) {
			System.out.println(e);				
			throw e;
		}finally{
			if(psmt!=null){
				smt.close();
				smt = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}

	//timetablerows�̃f�[�^��DB��̃e�[�u��timetablesql�ɕۑ�(�A������u�`���ۑ�)
	public void saveTaskSeriesTimeTable() throws Exception{
		Connection con = null;
		PreparedStatement psmt = null;
		Statement smt = null;
		try{
			//MySQL�T�[�o�ڑ�
			con = cmanager.getConnection();
			//Statement�C���^�[�t�F�[�X�̐���
			smt = con.createStatement();
			smt.executeUpdate("delete from timetablesql");
			psmt = con.prepareStatement("insert into timetablesql(task_id,period_id,processor_id) values( ? , ? , ? )");
			Iterator i = timetablerows.iterator();
			while(i.hasNext()){
				TimeTableRow t = (TimeTableRow)i.next();
				psmt.setInt(1,t.getTask_id());
				psmt.setInt(2,t.getPeriod_id());
				psmt.setInt(3,t.getProcessor_id());
				psmt.executeUpdate();
				
			}
		}catch(SQLException e) {
			System.out.println(e);				
			throw e;
		}finally{
			if(psmt!=null){
				smt.close();
				smt = null;
			}
			// Statement�C���^�[�t�F�[�X�̔j��
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQL�T�[�o�ؒf
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}

	//lectures����task_id�ł���u�`���擾����
	private Lecture searchLecture(int task_id){
		Iterator i = lectures.iterator();
		while(i.hasNext()){
			Lecture l = (Lecture)i.next();
			//�u�`l��task_id�ł���
			if(l.getTask_id()==task_id){
				return l;
			}
		}
		return null;
	}

	//teachers����processor_id�ł���u�t���擾����
	private Teacher searchTeacher(int processor_id){
		Iterator i = teachers.iterator();
		while(i.hasNext()){
			Teacher t = (Teacher)i.next();
			//�u�tt��processor_id�ł���
			if(t.getProcessor_id()==processor_id){
				return t;
			}
		}
		return null;
	}
	
}
