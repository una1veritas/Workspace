package tts_notdb;
import java.util.ArrayList;
import java.util.Collections;

import org.apache.commons.lang.time.StopWatch;

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
public class TimetableNotDBMain {
	public static void main(String args[]) throws Exception{
		String host = "localhost",db = "schedule",user = "yukiko",
		pass = "nishimura";
		int count = 10,local_search = 8,mode = 1;
		ArrayList list = new ArrayList(5);
		for(int i = 0; i< 5;i++){
			list.add(new Integer(99999));
		}
		//daisygw.daisy.ai.kyutech.ac.jp schedule yukiko nishimura t_valuate3.txt
		//引数 localhost schedule yukiko nishimura t_valuate3.txt 1 1 7 2
		switch (args.length) {
		case 7:
			host = args[0];db = args[1];user = args[2];pass = args[3];
			count = Integer.parseInt(args[6]);
			local_search = Integer.parseInt(args[7]);
			mode = Integer.parseInt(args[8]);
			break;
		case 6:
			host = args[0];db = args[1];user = args[2];pass = args[3];
			count = Integer.parseInt(args[5]);
			local_search = Integer.parseInt(args[6]);
			break;
		case 4:
			host = args[0];db = args[1];user = args[2];
			pass = args[3];
			break;
		case 1:
			host = args[0];
			break;
		default:
			break;
		}
		int valuate = 999999,min_valuate = 99999,t_valuate=99999;
		TimeTableSchedulingNotDB timetable = new TimeTableSchedulingNotDB(host,db,user,pass);
		long start, stop, time;
		java.text.DecimalFormat df2 =new java.text.DecimalFormat("00");
		java.text.DecimalFormat df3 =new java.text.DecimalFormat("000");
		System.out.println("全時間の計測を開始");
		start = System.currentTimeMillis();
		min_valuate = timetable.valuate("BesttimetableSQL");
		System.out.println("BesttimetableSQL:"+min_valuate);
		for(int i = 1; i<=count; i++){
			long start1, stop1, time1;
			//System.out.println("時間の計測を開始");
			start1 = System.currentTimeMillis();
			if(mode==1){
	            timetable.initsolNotRandom(i);
			}else if(mode==2){
				timetable.initsol2();
				timetable.SaveTimetable("initTimeTable");
			}else if(mode==3){
				timetable.LoadTimetable("initTimeTable");
			}
			//System.out.println(timetable.count_offence2("timetableSQL"));
			int valuate1 = timetable.valuate();
			System.out.println(valuate1);
			timetable.valuate2("timetableSQL");
			/*
			//変更
			int valuate2 = timetable.valuate2("timetableSQL");
			System.out.println(valuate2);
			if(valuate2!=valuate1){
				System.out.println("違う");
				timetable.SaveTimetable("tempTimeTable");
			}
			*/
			StopWatch sp = new StopWatch();
			sp.start();
			//局所探索の実行
			switch (local_search) {
			case 8:
				timetable.local_search4();break;
			default:
				timetable.local_search4();break;
			}
			sp.stop();
			System.out.println("StopWatchの時間:"+sp);
			System.out.println(timetable.count_offence2("timetableSQL"));
			valuate = timetable.valuate();
			System.out.println("valuate"+valuate);
			stop1 = System.currentTimeMillis();
			System.out.println("時間計測終了");
			time1 = stop1 - start1;
			int miri = (int)(time1%1000);
			int scd = (int)(time1/1000%60);
			int mini = (int)(time1/1000/60%60);
			int hour = (int)(time1/1000/60/60);
			String date = hour+":"+df2.format(mini)+":"+df2.format(scd)+"."+df3.format(miri);
			System.out.println("実行時間 : " + date);
			System.out.println("countOffence:"+timetable.count_offence2("timetableSQL"));
			System.out.println("valuateDetail:"+timetable.valuateDetail("timetableSQL"));
			t_valuate=((Integer)Collections.max(list)).intValue();
			if(valuate < t_valuate){
				int index = list.indexOf(new Integer(t_valuate));
				list.set(index,new Integer(valuate));
				timetable.SaveTimetable("timetableSQL"+(index+1));
			}
			if(valuate < min_valuate){
				min_valuate = valuate;
				timetable.SaveTimetable("BesttimetableSQL");
			}
		}
		stop = System.currentTimeMillis();
		System.out.println("全時間計測終了");
		time = stop - start;
		int miri = (int)(time%1000);
		int scd = (int)(time/1000%60);
		int mini = (int)(time/1000/60%60);
		int hour = (int)(time/1000/60/60);
		String date = hour+":"+df2.format(mini)+":"+df2.format(scd)+"."+df3.format(miri);
		System.out.println("実行時間 : " + date);
		timetable.close();
		return;
	}
}

