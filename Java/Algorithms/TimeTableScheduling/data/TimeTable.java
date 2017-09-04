/*
 * 作成日: 2007/04/02
 *
 * TODO この生成されたファイルのテンプレートを変更するには次へジャンプ:
 * ウィンドウ - 設定 - Java - コード・スタイル - コード・テンプレート
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
 * TODO 講義時間割の情報を管理
 *
 */
public class TimeTable {
	//(task_id,processor_id,period_id)の組の集合
	private LinkedList timetablerows;
	//コネクションプール
	private DBConnectionPool cmanager;
	//全講師の情報(リストの中身Object:Teacher)
	private ArrayList teachers;
	//全講義の情報(リストの中身Object:Lecture)
	private ArrayList lectures;
	//全時間の情報(リストの中身Object:DayPeriod)
	private OrderDayPeriods day_periods;
	
	/**
	 * @return day_periods を戻します。
	 */
	public OrderDayPeriods getDay_periods() {
		return day_periods;
	}
	
	//timetablerowsのコピーを返す
	public ArrayList getTimeTableRowsCopy(){
		ArrayList result = new ArrayList();
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			result.add(new TimeTableRow(t.getTask_id(),t.getPeriod_id(),t.getProcessor_id()));
		}
		return result;
	}
	
	//timetablerowsのコピーを返す
	//連続する講義はtask_series_num個
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
	
	//period_idにtask_idを担当しているprocessor_idのリストを返す
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
	
	//現在の時間割の講義とその担当人数と開講時間のリストを取得
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
	//現在の時間割の講義task_idの担当人数を取得
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
	//現在の時間割を講義とその開講時間のリストとして返す
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
	
	//timetablerowsの内容を表示
	public void printTimeTable(){
		System.out.println("(task_id,period_id,processor_id)");
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			System.out.println(t.toString());
		}
	}
	//timetablerowsの内容をデータベースのinsert形式で表示
	public void printInsertTimeTable(){
		System.out.println("(task_id,period_id,processor_id)");
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			System.out.println("insert into timetablesql(task_id,period_id,processor_id) values"+t.toString()+";");
		}
	}
	//timetablerowsをクリアする
	public void clearTimeTable() throws Exception{
		timetablerows.clear();
	}
	
	//DB上のtimetablesqlからデータを作成(変更予定)
	public void loadTaskSeriesTimeTable() throws Exception{
		timetablerows.clear();
		ArrayList list = new ArrayList();
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		int task_id,period_id,processor_id;
		try{
			//MySQLサーバ接続
			con = cmanager.getConnection();
			//Statementインターフェースの生成
			smt = con.createStatement();
			//現在の時間割の取得
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
			//ResultSetインターフェースの破棄
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statementインターフェースの破棄
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQLサーバ切断
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
		//DB上のテーブルには連続する講義も入っている
		//連続する講義の削除
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

	//DB上のtimetablesqlからデータを作成
	public void loadTimeTable() throws Exception{
		timetablerows.clear();
		Connection con = null;
		ResultSet rs = null;
		Statement smt = null;
		int task_id,period_id,processor_id;
		try{
			//MySQLサーバ接続
			con = cmanager.getConnection();
			//Statementインターフェースの生成
			smt = con.createStatement();
			//現在の時間割の取得
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
			//ResultSetインターフェースの破棄
			if(rs!=null){
				rs.close();
				rs = null;
			}
			// Statementインターフェースの破棄
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQLサーバ切断
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
	
	//制約違反してる数を返す
	public int countOffence(){
		Set temp1 = new TreeSet();
		ArrayList temp2 = new ArrayList();
	    ArrayList temp3 = new ArrayList();
		//●1.各講師は担当可能時間に(担当可能な講義)を担当
		//2.各講義は、各講義固有の開講可能時間のうち一つに開講される
		//●2-1.各講義は開講可能時間に開講される
		//●2-2.各講義は開講可能であれば必ず開講される
		//▲2-3.各講義はただ一つだけ開講される
		//▲3.各講師は、同じ時間に複数の講義を担当しない
		Iterator i = timetablerows.iterator();
		int count = 0;
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			Teacher teacher = searchTeacher(t.getProcessor_id());
			Lecture lecture = searchLecture(t.getTask_id());
			if(!teacher.isPeriod(t.getPeriod_id()))count++;
			if(!teacher.isQualification(lecture.getQualification_id()))count++;
			if(!lecture.isPeriod(t.getPeriod_id()))count++;
			//開講している講義数を調べる
			temp1.add(new Integer(t.getTask_id()));
			//講義・時間の組にまとめる（重複しない）
			if(!temp2.contains(new TaskPeriod(t.getTask_id(),t.getPeriod_id()))){
				temp2.add(new TaskPeriod(t.getTask_id(),t.getPeriod_id()));
			}
			//講師・時間の組にまとめる（重複しない）
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

	//制約違反してる数を返す（詳細情報表示）
	public int countOffenceDetail(){
		Set temp1 = new TreeSet();
		ArrayList temp2 = new ArrayList();
	    ArrayList temp3 = new ArrayList();
		//●1.各講師は担当可能時間に(担当可能な講義)を担当
		//2.各講義は、各講義固有の開講可能時間のうち一つに開講される
		//●2-1.各講義は開講可能時間に開講される
		//●2-2.各講義は開講可能であれば必ず開講される
		//▲2-3.各講義はただ一つだけ開講される
		//▲3.各講師は、同じ時間に複数の講義を担当しない
		Iterator i = timetablerows.iterator();
		int count = 0;
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			Teacher teacher = searchTeacher(t.getProcessor_id());
			Lecture lecture = searchLecture(t.getTask_id());
			if(!teacher.isPeriod(t.getPeriod_id())){
				System.out.println("講師:"+t.getProcessor_id()+"は講義時間:"+t.getPeriod_id()+"に担当できません");
				count++;
			}
			//if(!teacher.isQualification(lecture.getQualification_id()))count++;
			if(!lecture.isPeriod(t.getPeriod_id())){
				System.out.println("講義:"+t.getTask_id()+"は講義時間:"+t.getPeriod_id()+"に開講できません");
				count++;
			}
			//開講している講義数を調べる
			temp1.add(new Integer(t.getTask_id()));
			//講義・時間の組にまとめる（重複しない）
			if(!temp2.contains(new TaskPeriod(t.getTask_id(),t.getPeriod_id()))){
				temp2.add(new TaskPeriod(t.getTask_id(),t.getPeriod_id()));
			}
			//講師・時間の組にまとめる（重複しない）
			if(!temp3.contains(new ProcessorPeriod(t.getProcessor_id(),t.getPeriod_id()))){
				temp3.add(new ProcessorPeriod(t.getProcessor_id(),t.getPeriod_id()));
			}
		}
		if(temp1.size()<lectures.size()){
			System.out.println("開講されてない講義数:"+(lectures.size()-temp1.size()));
			i = lectures.iterator();
			while(i.hasNext()){
				Lecture t = (Lecture)i.next();
				if(!temp1.contains(new Integer(t.getTask_id()))){
					System.out.println("講義:"+t.getTask_id()+"は開講されていません");
				}
			}
			count = count + (lectures.size()-temp1.size());
		}
		//各講義はただ一つだけ開講される
		i = temp2.iterator();
		while(i.hasNext()){
			TaskPeriod t = (TaskPeriod)i.next();
			if(!isNotTasks3(t.getTask_id(),t.getPeriod_id())){
				System.out.println("講義:"+t.getTask_id()+"は講義時間:"+t.getPeriod_id()+"以外の講義時間にも開講されている");
				count++;
			}
		}
		//各講師は、同じ時間に複数の講義を担当しない
		i = temp3.iterator();
		while(i.hasNext()){
			ProcessorPeriod t = (ProcessorPeriod)i.next();
			if(!isNotTasks2(t.getProcessor_id(),t.getPeriod_id())){
				System.out.println("講師:"+t.getProcessor_id()+"は講義時間:"+t.getPeriod_id()+"に複数の講義を担当している");
				count++;
			}
		}
		return count;
	}
	//連続する講義に対応
	//制約違反してる数を返す（詳細情報表示）
	public int taskSeriesCountOffenceDetail(){
		Set temp1 = new TreeSet();
		ArrayList temp2 = new ArrayList();
	    ArrayList temp3 = new ArrayList();
		//●1.各講師は担当可能時間に(担当可能な講義)を担当
		//2.各講義は、各講義固有の開講可能時間のうち一つに開講される
		//●2-1.各講義は開講可能時間に開講される
		//●2-2.各講義は開講可能であれば必ず開講される
		//▲2-3.各講義はただ一つだけ開講される
		//▲3.各講師は、同じ時間に複数の講義を担当しない
		Iterator i = timetablerows.iterator();
		int count = 0;
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			Teacher teacher = searchTeacher(t.getProcessor_id());
			Lecture lecture = searchLecture(t.getTask_id());
			//if(!teacher.isQualification(lecture.getQualification_id()))count++;
			//連続する講義の違反は連続する数倍(違反の変更)
			if(lecture.getTask_series_num()==1){
				if(!lecture.isPeriod(t.getPeriod_id())){
					System.out.println("講義:"+t.getTask_id()+"は講義時間:"+t.getPeriod_id()+"に開講できません");
					count++;
				}
				if(!teacher.isPeriod(t.getPeriod_id())){
					System.out.println("講師:"+t.getProcessor_id()+"は講義時間:"+t.getPeriod_id()+"に担当できません");
					count++;
				}
				//講師・時間の組にまとめる（重複しない）
				if(!temp3.contains(new ProcessorPeriod(t.getProcessor_id(),t.getPeriod_id()))){
					temp3.add(new ProcessorPeriod(t.getProcessor_id(),t.getPeriod_id()));
				}
			}else{
				System.out.println("講義:"+t.getTask_id()+"は講義時間:"+t.getPeriod_id()+"から"+lecture.getTask_series_num()+"時間連続して開講");
				if(!lecture.isPeriod(t.getPeriod_id())){
					System.out.println("講義:"+t.getTask_id()+"は講義時間:"+t.getPeriod_id()+"に開講できません");
					count++;
				}
				if(!teacher.isPeriod(t.getPeriod_id())){
					System.out.println("講師:"+t.getProcessor_id()+"は講義時間:"+t.getPeriod_id()+"に担当できません");
					count++;
				}
				//講師・時間の組にまとめる（重複しない）
				if(!temp3.contains(new ProcessorPeriod(t.getProcessor_id(),t.getPeriod_id()))){
					temp3.add(new ProcessorPeriod(t.getProcessor_id(),t.getPeriod_id()));
				}
				int next_period_id = t.getPeriod_id();
				for(int num=1;num<lecture.getTask_series_num();num++){
					if(next_period_id<0)next_period_id = next_period_id-1;
					else next_period_id = day_periods.getNextPeriods(next_period_id);
					if(next_period_id==-1)next_period_id = -1*t.getPeriod_id()-1;
					if(!lecture.isPeriod(next_period_id)){
						System.out.println("講義:"+t.getTask_id()+"は講義時間:"+next_period_id+"に開講できません");
						count++;
					}
					if(!teacher.isPeriod(next_period_id)){
						System.out.println("講師:"+t.getProcessor_id()+"は講義時間:"+next_period_id+"に担当できません");
						count++;
					}
					//講師・時間の組にまとめる（重複しない）
					if(!temp3.contains(new ProcessorPeriod(t.getProcessor_id(),next_period_id))){
						temp3.add(new ProcessorPeriod(t.getProcessor_id(),next_period_id));
					}
				}
			}
			//開講している講義数を調べる
			temp1.add(new Integer(t.getTask_id()));
			//講義・時間の組にまとめる（重複しない）
			if(!temp2.contains(new TaskPeriod(t.getTask_id(),t.getPeriod_id()))){
				temp2.add(new TaskPeriod(t.getTask_id(),t.getPeriod_id()));
			}
		}
		if(temp1.size()<lectures.size()){
			System.out.println("開講されてない講義数:"+(lectures.size()-temp1.size()));
			i = lectures.iterator();
			while(i.hasNext()){
				Lecture t = (Lecture)i.next();
				if(!temp1.contains(new Integer(t.getTask_id()))){
					System.out.println("講義:"+t.getTask_id()+"は開講されていません");
				}
			}
			count = count + (lectures.size()-temp1.size());
		}
		//各講義はただ一つだけ開講される
		//System.out.println("temp2");
		i = temp2.iterator();
		while(i.hasNext()){
			TaskPeriod t = (TaskPeriod)i.next();
			//System.out.println("task_id:"+t.getTask_id()+" period_id:"+t.getPeriod_id());
			if(!isNotTasks3(t.getTask_id(),t.getPeriod_id())){
				System.out.println("講義:"+t.getTask_id()+"は講義時間:"+t.getPeriod_id()+"以外の講義時間にも開講されている");
				count++;
			}
		}
		//各講師は、同じ時間に複数の講義を担当しない
		//System.out.println("temp3");
		i = temp3.iterator();
		while(i.hasNext()){
			ProcessorPeriod t = (ProcessorPeriod)i.next();
			//System.out.println("processor_id:"+t.getProcessor_id()+" period_id:"+t.getPeriod_id());
			if(!isNotTasks2(t.getProcessor_id(),t.getPeriod_id())){
				System.out.println("講師:"+t.getProcessor_id()+"は講義時間:"+t.getPeriod_id()+"に複数の講義を担当している");
				count++;
			}
		}
		return count;
	}
	//連続する講義に対応
	//制約違反してる数を返す
	public int taskSeriesCountOffence(){
		Set temp1 = new TreeSet();
		ArrayList temp2 = new ArrayList();
	    ArrayList temp3 = new ArrayList();
		//●1.各講師は担当可能時間に(担当可能な講義)を担当
		//2.各講義は、各講義固有の開講可能時間のうち一つに開講される
		//●2-1.各講義は開講可能時間に開講される
		//●2-2.各講義は開講可能であれば必ず開講される
		//▲2-3.各講義はただ一つだけ開講される
		//▲3.各講師は、同じ時間に複数の講義を担当しない
		Iterator i = timetablerows.iterator();
		int count = 0;
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			Teacher teacher = searchTeacher(t.getProcessor_id());
			Lecture lecture = searchLecture(t.getTask_id());
			//if(!teacher.isQualification(lecture.getQualification_id()))count++;
			//連続する講義の違反は連続する数倍(違反の変更)
			if(lecture.getTask_series_num()==1){
				if(!teacher.isPeriod(t.getPeriod_id()))count++;
				if(!lecture.isPeriod(t.getPeriod_id()))count++;
				//講師・時間の組にまとめる（重複しない）
				if(!temp3.contains(new ProcessorPeriod(t.getProcessor_id(),t.getPeriod_id()))){
					temp3.add(new ProcessorPeriod(t.getProcessor_id(),t.getPeriod_id()));
				}
			}else{
				if(!teacher.isPeriod(t.getPeriod_id()))count++;
				if(!lecture.isPeriod(t.getPeriod_id()))count++;
				//講師・時間の組にまとめる（重複しない）
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
					//講師・時間の組にまとめる（重複しない）
					if(!temp3.contains(new ProcessorPeriod(t.getProcessor_id(),next_period_id))){
						temp3.add(new ProcessorPeriod(t.getProcessor_id(),next_period_id));
					}
				}
			}
			//開講している講義数を調べる
			temp1.add(new Integer(t.getTask_id()));
			if(!temp2.contains(new TaskPeriod(t.getTask_id(),t.getPeriod_id()))){
				temp2.add(new TaskPeriod(t.getTask_id(),t.getPeriod_id()));
			}
		}
		if(temp1.size()<lectures.size()){
			count = count + (lectures.size()-temp1.size());
		}
		//現在の時間割中に各講義が一つしか開講しないかどうか
		i = temp2.iterator();
		while(i.hasNext()){
			TaskPeriod t = (TaskPeriod)i.next();
			if(!isNotTasks3(t.getTask_id(),t.getPeriod_id()))count++;
		}
		//現在の時間割中に講師が同じ時間に複数の講義を担当しないかどうか
		i = temp3.iterator();
		while(i.hasNext()){
			ProcessorPeriod t = (ProcessorPeriod)i.next();
			if(!isNotTasks2(t.getProcessor_id(),t.getPeriod_id()))count++;
		}
		return count;
	}
	
	//制約違反していないかどうか(現在未使用)
	//(違反していない場合:true している:false)
	public boolean isNotOffence(){
		Set temp = new TreeSet();
		//●1.各講師は担当可能時間に担当可能な講義を担当
		//2.各講義は、各講義固有の開講可能時間のうち一つに開講される
		//●2-1.各講義は開講可能時間に開講される
		//●2-2.各講義は開講可能であれば必ず開講される
		//▲2-3.各講義はただ一つだけ開講される
		//▲3.各講師は、同じ時間に複数の講義を担当しない
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
	
	//講師processor_idがperiod_idに複数の講義を担当しないかどうか
	//(担当しない:true 担当する:false)
	//processor_idがperiod_idが１つも存在していない時
	public boolean isNotTasks(int processor_id, int period_id){
		Iterator i = timetablerows.iterator();
		while(i.hasNext()){
			TimeTableRow t = (TimeTableRow)i.next();
			if((t.getProcessor_id()==processor_id)&&(t.getPeriod_id()==period_id)) return false;
		}
		return true;
	}
	
	//講師processor_idがperiod_idに複数の講義を担当しないかどうか
	//(担当しない:true 担当する:false)
	//processor_idがperiod_idに担当している講義が高々１つしか存在していない時
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
	
	//現在の時間割中に講義が一つしか開講されていない
	//(開講されていない:true 開講されている:false)
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
	
	//timetablerowsのデータをDB上のテーブルtimetablesqlに保存
	public void saveTimeTable() throws Exception{
		Connection con = null;
		PreparedStatement psmt = null;
		Statement smt = null;
		try{
			//MySQLサーバ接続
			con = cmanager.getConnection();
			//Statementインターフェースの生成
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
			// Statementインターフェースの破棄
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQLサーバ切断
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}

	//timetablerowsのデータをDB上のテーブルtimetablesqlに保存(連続する講義も保存)
	public void saveTaskSeriesTimeTable() throws Exception{
		Connection con = null;
		PreparedStatement psmt = null;
		Statement smt = null;
		try{
			//MySQLサーバ接続
			con = cmanager.getConnection();
			//Statementインターフェースの生成
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
			// Statementインターフェースの破棄
			if(smt!=null){
				smt.close();
				smt = null;
			}
			//MySQLサーバ切断
			if(con!=null){
				cmanager.freeConnection(con);
			}
		}
	}

	//lecturesからtask_idである講義を取得する
	private Lecture searchLecture(int task_id){
		Iterator i = lectures.iterator();
		while(i.hasNext()){
			Lecture l = (Lecture)i.next();
			//講義lがtask_idである
			if(l.getTask_id()==task_id){
				return l;
			}
		}
		return null;
	}

	//teachersからprocessor_idである講師を取得する
	private Teacher searchTeacher(int processor_id){
		Iterator i = teachers.iterator();
		while(i.hasNext()){
			Teacher t = (Teacher)i.next();
			//講師tがprocessor_idである
			if(t.getProcessor_id()==processor_id){
				return t;
			}
		}
		return null;
	}
	
}
