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
 * 作成日: 2007/10/22
 *
 * TODO この生成されたファイルのテンプレートを変更するには次へジャンプ:
 * ウィンドウ - 設定 - Java - コード・スタイル - コード・テンプレート
 */

/**
 * @author masayoshi
 *
 * TODO 講義時間割作成のメイン
 *
 */
public class TimeTableMain {
	private static int PORT = 3776;
	public static void main(String args[]) throws Exception{
        if (!isOnlyMe()) {
            System.out.println("既に起動されてます");
            System.exit(-1);
        }
        System.out.println("単独起動されました");
        //作成日時の取得
        Timestamp create_date = new Timestamp(System.currentTimeMillis());
        SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMddkkmmss");
        String s_create_date = sdf.format(new Date(create_date.getTime()));
		TimeTableSchedulingTaskSeries tts = null;
		//serverhost:daisygw.daisy.ai.kyutech.ac.jp
		String host = "localhost",db = "timetable",user = "satonaka",
		pass = "asdfg",name = "valuatesql";
		//readType 1:番号付きＳＱＬのname名のファイルから 2:番号無しＳＱＬのname名のファイルから
		//         3:name名のデータベース上のテーブルから読み出す
		//mode 3:なし 4:プロジェクトを考慮 5:プロジェクトを考慮&&連続する講義を考慮
		int count = 1,readType = 3,mode = 4,list_size = 5,project_id = 1;
		//readType:3&&(mode:5or4)で起動
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
		//評価良い順list_size個に時間割案を入れるリスト
		ArrayList list = new ArrayList(list_size);
		for(int i = 0; i< list_size;i++){
			list.add(new SaveTimeTable(null,999999,999999));
		}
		SaveTimeTable t_stt,best_sst;
		int count_offence = 999999,min_count_offence=99999,t_count_offence = 999999;
		int valuate = 999999,min_valuate = 99999,t_valuate=99999;
		tts = new TimeTableSchedulingTaskSeries(host,db,user,pass,name,readType,project_id,mode);
		//BEST時間割候補の読み込み(現在未使用)
		//テーブルtimetablesqlにデータを読み込む(要望の評価)
		//tts.loadTimeTableAddProject("besttimetablesql",project_id);
		//プログラム上に時間割情報を読み込む(制約の評価)
		//tts.getTimetable().loadTaskSeriesTimeTable();
		//min_valuate = tts.valuateTimeTableSQLDB();
		//min_count_offence = tts.count_offence();
		//BESTの時間割候補の評価値を表示
		//System.out.println("besttimetablesql valueate:"+min_valuate);
		//System.out.println("besttimetablesql offence:"+min_count_offence);
		//best_sst = new SaveTimeTable(tts.getTimetable().getTaskSeriesTimeTableRowsCopy(),min_count_offence,min_valuate);
		
		//count個 時間割案の生成
		for(int i = 1; i<=count; i++){
			//初期解を生成
			if(mode==4){
				tts.initsolAddProject(project_id);
			}else if(mode==5){
				tts.initsolTaskSeriesAddProject(project_id);
				//tts.initsolNotRandomTaskSeriesAddProject(i,project_id);
			}
			//初期解の評価値(詳細)を表示
			if(mode==5) System.out.println(tts.task_series_count_offenceDetail());
			else System.out.println(tts.count_offenceDetail());
			System.out.println(tts.valuateTimeTableSQLDBDetail());
			
			StopWatch sp = new StopWatch();
			sp.start();
			//局所探索の実行
			System.out.println("local_serch start");
			if(mode==5){
				tts.task_series_local_search();
			}else tts.local_search();
			System.out.println("local_serch end");
			sp.stop();
			System.out.println("時間:"+sp);
			
			//時間割案の評価値計算
			if(mode==5) count_offence = tts.task_series_count_offenceDetail();
			else count_offence = tts.count_offenceDetail();
			valuate = tts.valuateTimeTableSQLDBDetail();
			//評価値を表示
			System.out.println("valuate:"+valuate);
			System.out.println("count_offence:"+count_offence);
			
			//list中の時間割案の内一番評価の悪いものを取得
			t_stt=((SaveTimeTable)Collections.max(list));
			t_valuate = t_stt.getValuate();
			t_count_offence = t_stt.getCount_offence();
			//listの更新
			if(count_offence < t_count_offence){
				int index = list.indexOf(new SaveTimeTable(null,t_count_offence,t_valuate));
				//mode:4の時　バグがあるかも
				list.set(index,new SaveTimeTable(tts.getTimetable().getTaskSeriesTimeTableRowsCopy(),count_offence,valuate));
			}else if(count_offence == t_count_offence && valuate < t_valuate){
				int index = list.indexOf(new SaveTimeTable(null,t_count_offence,t_valuate));
				//mode:4の時　バグがあるかも
				list.set(index,new SaveTimeTable(tts.getTimetable().getTaskSeriesTimeTableRowsCopy(),count_offence,valuate));
			}
			/*BEST時間割候補の更新
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
				//DBLogのため
				//sstのデータをＤＢ上のtimetablesqlに読み込む
				stt.saveTimetablesql(tts.getCmanager());
				//評価ファイルの保存："日時,project_id,候補番号.txt"
				tts.fileTimeTableSQLDBLog("./result/"+s_create_date+""+project_id+""+index+".txt");
				tts.saveTimeTableAddProject("timetable",project_id,index,create_date);
			}
			index++;
		}
		//BEST時間割候補の保存
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
