import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;

import org.apache.commons.lang.time.StopWatch;

import data.SaveTimeTable;

/*
 * 作成日: 2006/11/10
 *
 * TODO この生成されたファイルのテンプレートを変更するには次へジャンプ:
 * ウィンドウ - 設定 - Java - コード・スタイル - コード・テンプレート
 */

/**
 * @author masayoshi
 *
 * TODO この生成された型コメントのテンプレートを変更するには次へジャンプ:
 * ウィンドウ - 設定 - Java - コード・スタイル - コード・テンプレート
 */
public class TimeTableSchedulingTaskSeriesMain {
	public static void main(String args[]) throws Exception{
		TimeTableSchedulingTaskSeries tts = null;
		String host = "localhost",db = "timetable",user = "satonaka",
		pass = "asdfg",name = "valuatesql";
		//readType 1:番号付きＳＱＬのname名のファイルから
		//readType 2:番号無しＳＱＬのname名のファイルから
		//readType 3:name名のデータベースのテーブルから読み出す
		int count = 1,readType = 3,mode = 5,list_size = 5,project_id = 3;
		//daisygw.daisy.ai.kyutech.ac.jp schedule yukiko nishimura t_valuate3.txt
		switch (args.length) {
		case 8:
			host = args[0];db = args[1];user = args[2];pass = args[3];
			name = args[4];readType = Integer.parseInt(args[5]);
			count = Integer.parseInt(args[6]);mode = Integer.parseInt(args[7]);
			break;
		case 7:
			host = args[0];db = args[1];user = args[2];pass = args[3];
			project_id= Integer.parseInt(args[4]);count = Integer.parseInt(args[5]);
			list_size = Integer.parseInt(args[6]);
			break;
		case 6:
			host = args[0];db = args[1];user = args[2];pass = args[3];
			project_id= Integer.parseInt(args[4]);count = Integer.parseInt(args[5]);
			break;
		case 5:
			host = args[0];db = args[1];user = args[2];pass = args[3];name = args[4];
			break;
		case 1:
			count = Integer.parseInt(args[0]);
			break;
		default:
			break;
		}
		ArrayList list = new ArrayList(list_size);
		for(int i = 0; i< list_size;i++){
			list.add(new SaveTimeTable(null,999999,999999));
		}
		SaveTimeTable t_stt,best_sst;
		int valuate = 999999,min_valuate = 99999,t_valuate=99999;
		int count_offence = 999999,min_count_offence=99999,t_count_offence = 999999;
		if(mode==4||mode==5){
			tts = new TimeTableSchedulingTaskSeries(host,db,user,pass,name,readType,project_id,mode);
		}else{
			tts = new TimeTableSchedulingTaskSeries(host,db,user,pass,name,readType);
		}
		tts.loadTimeTableAddProject("besttimetablesql",project_id);
		tts.getTimetable().loadTaskSeriesTimeTable();
		min_valuate = tts.valuateTimeTableSQLDB();
		min_count_offence = tts.count_offence();
		System.out.println("besttimetablesql valueate:"+min_valuate);
		System.out.println("besttimetablesql offence:"+min_count_offence);
		best_sst = new SaveTimeTable(tts.getTimetable().getTaskSeriesTimeTableRowsCopy(),min_count_offence,min_valuate);
		//tts.printTimeTable();
		for(int i = 1; i<=count; i++){
			if(mode==1){
	            tts.initsolNotRandom(i);
			}else if(mode==2){
				tts.initsol();
				//tts.SaveTimetable("initTimeTable");
			}else if(mode==3){
				tts.loadTimeTable("timetablesql");
			}else if(mode==4){
				tts.initsolAddProject(project_id);
			}else if(mode==5){
				System.out.println("initSol start");
				tts.initsolTaskSeriesAddProject(project_id);
				//tts.initsolNotRandomTaskSeriesAddProject(i,project_id);
			}
			if(mode==5){
				System.out.println(tts.task_series_count_offenceDetail());
				System.out.println(tts.valuateTimeTableSQLDBDetail());
			}else{
				System.out.println(tts.count_offenceDetail());
				System.out.println(tts.valuateTimeTableSQL());
			}
			StopWatch sp = new StopWatch();
			sp.start();
			//局所探索の実行
			if(mode==5){
				System.out.println("local_serch start");
				tts.task_series_local_search();
				System.out.println("local_serch end");
			}else{
				tts.local_search();
			}
			sp.stop();
			System.out.println("StopWatchの時間:"+sp);
			if(mode==5){
				valuate = tts.valuateTimeTableSQL();
				count_offence = tts.task_series_count_offenceDetail();
			}else{
				valuate = tts.valuateTimeTableSQL();
				count_offence = tts.count_offenceDetail();
			}
			System.out.println("valuate:"+valuate);
			System.out.println("count_offence:"+count_offence);
			t_stt=((SaveTimeTable)Collections.max(list));
			t_valuate = t_stt.getValuate();
			t_count_offence = t_stt.getCount_offence();
			//System.out.println("t_valuate:"+t_valuate);
			//System.out.println("t_count_offence:"+t_count_offence);
			if(count_offence < t_count_offence){
				int index = list.indexOf(new SaveTimeTable(null,t_count_offence,t_valuate));
				list.set(index,new SaveTimeTable(tts.getTimetable().getTaskSeriesTimeTableRowsCopy(),count_offence,valuate));
			}else if(count_offence == t_count_offence && valuate < t_valuate){
				int index = list.indexOf(new SaveTimeTable(null,t_count_offence,t_valuate));
				list.set(index,new SaveTimeTable(tts.getTimetable().getTaskSeriesTimeTableRowsCopy(),count_offence,valuate));
				//tts.TimeTableSQLDBLog("timetablesql"+(index+1));
				//tts.SaveTimetable("timetablesql"+(index+1));
			}
			if(count_offence < min_count_offence){
				min_count_offence = count_offence;
				min_valuate = valuate;
				best_sst=new SaveTimeTable(tts.getTimetable().getTaskSeriesTimeTableRowsCopy(),min_count_offence,min_valuate);
			}else if(count_offence == min_count_offence && valuate < min_valuate){
				min_valuate = valuate;
				best_sst=new SaveTimeTable(tts.getTimetable().getTaskSeriesTimeTableRowsCopy(),min_count_offence,min_valuate);
			}
		}
		Collections.sort(list);
		Iterator i = list.iterator();
		int index = 1;
		while(i.hasNext()){
			SaveTimeTable stt = (SaveTimeTable)i.next();
			if(stt.getTimetablerows()!=null){
				//sstのデータをＤＢ上のテーブルtimetablesql上に
				//DBLogのため
				stt.saveTimetable(tts.getCmanager());
				if(mode==4||mode==5){
					tts.fileTimeTableSQLDBLog("./result/timetablesql"+index+"-"+project_id+".txt");
					tts.saveTimeTableAddProject("timetablesql"+index,project_id);
				}else{
					tts.fileTimeTableSQLDBLog("timetablesql"+index+".txt");
					tts.saveTimeTable("timetablesql"+index);
				}
			}else{
				//if(mode==4)tts.deleteTimetable("timetablesql"+index,project_id);
			}
			index++;
		}
		best_sst.saveTimetable(tts.getCmanager());
		tts.fileTimeTableSQLDBLog("./result/besttimetablesql"+project_id+".txt");
		tts.saveTimeTableAddProject("besttimetablesql",project_id);
		tts.close();
		return;
	}
}

