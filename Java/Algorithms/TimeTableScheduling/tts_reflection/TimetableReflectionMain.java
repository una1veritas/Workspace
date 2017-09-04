package tts_reflection;
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
public class TimetableReflectionMain {
	public static void main(String args[]) throws Exception{
		TimeTableSchedulingReflection tts = null;
		String host = "localhost",db = "timetable",user = "satonaka",
		pass = "asdfg",file = "valuatesql",ClassName = "data.TimeTableValuateTemp";
		//readType 1:番号付きＳＱＬのfile名のファイルから
		//readType 2:番号無しＳＱＬのfile名のファイルから
		//readType 3:file名のデータベースのテーブルから読み出す
		int count = 1,readType = 3,mode = 4,list_size = 5,project_id = 3;
		//daisygw.daisy.ai.kyutech.ac.jp schedule yukiko nishimura t_valuate3.txt
		//引数 localhost schedule yukiko nishimura t_valuate3.txt 1 1 7 1
		switch (args.length) {
		case 8:
			host = args[0];db = args[1];user = args[2];pass = args[3];
			file = args[4];readType = Integer.parseInt(args[5]);
			count = Integer.parseInt(args[6]);
			mode = Integer.parseInt(args[7]);
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
			host = args[0];db = args[1];user = args[2];
			pass = args[3];file = args[4];
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
		SaveTimeTable t_stt;
		int valuate = 999999,min_valuate = 99999,t_valuate=99999;
		int count_offence = 999999,t_count_offence = 999999;
		if(mode==4){
			tts = new TimeTableSchedulingReflection(host,db,user,pass,ClassName,project_id);
		}else{
			tts = new TimeTableSchedulingReflection(host,db,user,pass,ClassName);
		}
		//tts.LoadTimetable("BesttimetableSQL");
		//min_valuate = tts.valuateTimeTableSQLDB();
		//System.out.println("BesttimetableSQL:"+min_valuate);
		for(int i = 1; i<=count; i++){
			if(mode==1){
	            tts.initsolNotRandom(i);
			}else if(mode==2){
				tts.initsol();
				//tts.SaveTimetable("initTimeTable");
			}else if(mode==3){
				tts.LoadTimetable("timetablesql2");
			}else if(mode==4){
				tts.initsolAddProject(project_id);
			}
			//System.out.println(tts.count_offence());
			//System.out.println(tts.valuateTimeTableSQL());
			StopWatch sp = new StopWatch();
			sp.start();
			//局所探索の実行
			tts.local_search();
			sp.stop();
			//System.out.println("StopWatchの時間:"+sp);
			//System.out.println(tts.count_offence("timetableSQL"));
			valuate = tts.valuate();
			count_offence = tts.count_offence();
			//System.out.println("valuate:"+valuate);
			//System.out.println("count_offence:"+count_offence);
			//System.out.println("countOffence:"+tts.count_offence("timetableSQL"));
			//System.out.println("valuateDetail:"+tts.valuateDetail("timetableSQL"));
			t_stt=((SaveTimeTable)Collections.max(list));
			t_valuate = t_stt.getValuate();
			t_count_offence = t_stt.getCount_offence();
			//System.out.println("t_valuate:"+t_valuate);
			//System.out.println("t_count_offence:"+t_count_offence);
			if(count_offence < t_count_offence){
				int index = list.indexOf(new SaveTimeTable(null,t_count_offence,t_valuate));
				list.set(index,new SaveTimeTable(tts.getTimetable().getTimeTableRowsCopy(),count_offence,valuate));
			}else if(valuate < t_valuate){
				int index = list.indexOf(new SaveTimeTable(null,t_count_offence,t_valuate));
				list.set(index,new SaveTimeTable(tts.getTimetable().getTimeTableRowsCopy(),count_offence,valuate));
				//tts.TimeTableSQLDBLog("timetableSQL"+(index+1));
				//tts.SaveTimetable("timetableSQL"+(index+1));
			}
			/*
			if(valuate < min_valuate){
				min_valuate = valuate;
				tts.SaveTimetable("BesttimetableSQL");
			}
			*/
		}
		Collections.sort(list);
		Iterator i = list.iterator();
		int index = 1;
		while(i.hasNext()){
			SaveTimeTable stt = (SaveTimeTable)i.next();
			if(stt.getTimetablerows()!=null){
				stt.saveTimetablesql(tts.getCmanager());
				if(mode==4){
					tts.TimeTableSQLDBLog("./result/timetablesql"+index+"-"+project_id+".txt");
					stt.saveTimeTableAddProject(tts.getCmanager(),"timetablesql"+index,project_id);
				}else{
					tts.TimeTableSQLDBLog("timetablesql"+index+".txt");
					tts.SaveTimetable("timetablesql"+index);
				}
			}else{
				if(mode==4)tts.deleteTimetable("timetablesql"+index,project_id);
			}
			index++;
		}
		tts.close();
		return;
	}
}

