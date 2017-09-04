//
//  SQLTimeTabling.java
//  SQLTimeTabling
//
//  Created by ?? ?? on 07/12/12.
//  Copyright (c) 2007 __MyCompanyName__. All rights reserved.
//
import java.util.*;
import java.io.*;
import java.sql.Timestamp;
import java.text.SimpleDateFormat;

import data.Timetable;

//java -classpath ./mysql-connector-java-5.1.5-bin.jar:./commons-lang-2.3.jar:./jars/SQLTimeTabling.jar SQLTimeTabling daisygw timetable satonaka asdfg 1 1 5



public class SQLTimeTabling {	

	public static void main(String args[]) throws Exception {
         //作成日時の取得
		long started = System.currentTimeMillis();
        String s_create_date = (new SimpleDateFormat("yyyy-MM-dd HH:mm:ss")).format(
		new Timestamp(started));

		TimeTableScheduling tts = null;
		//serverhost:daisygw.daisy.ai.kyutech.ac.jp
		String host = "localhost",db = "timetable",user = "satonaka",
		pass = "asdfg", //name = "valuatesql";
		name = "rules";

		int project_id = 1, count = 3, timeLimit = 5;
		switch (args.length) {
			case 7:
				host = args[0];db = args[1];user = args[2];pass = args[3];project_id= Integer.parseInt(args[4]);
				count = Integer.parseInt(args[5]);
				timeLimit = Integer.parseInt(args[6]);
				break;
			case 6:
				host = args[0];db = args[1];user = args[2];pass = args[3];project_id= Integer.parseInt(args[4]);
				count = Integer.parseInt(args[5]);
				break;
			case 0:
				break;
			default: break;
		}
		
		tts = new TimeTableScheduling(host,db,user,pass,name, project_id /* ,5 mode */);
		
		//時間割案の生成
		for(int i = 1; i <= count && ((System.currentTimeMillis() - started)/1000/60 + 1< timeLimit) ; i++){
			//初期解を生成
			tts.makeInitialSolution(project_id);
			System.err.println("created a new initial solution.");
			
			//局所探索の実行
			System.err.print("searching for a local optima");
			tts.local_search();
			System.err.println("Finished.");
			
			//時間割案の評価値計算
			//count_offence = tts.task_series_count_offenceDetail();
			//valuate = tts.valuateTimeTableSQLDBDetail();
			//評価値を表示
			//System.out.println("valuate:"+valuate);
			//System.out.println("count_offence:"+count_offence);
			
			tts.storeTimeTableOnDBTable("results", project_id, s_create_date);
			System.err.println("Stored a result, which violates "+tts.violations()+" constraints and is imposed the total amount of penalty "+tts.cost()+".");
			System.err.println();
		}
		tts.closeConnection();
		return;
	}

}
