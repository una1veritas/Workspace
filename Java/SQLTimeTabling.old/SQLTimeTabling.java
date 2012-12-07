//
//  SQLTimeTabling.java
//  SQLTimeTabling
//
//  Created by ?? ?? on 06/12/22.
//  Copyright (c) 2006 __MyCompanyName__. All rights reserved.
//
import java.util.*;
import java.io.*;

public class SQLTimeTabling {
	
    public static void main (String args[]) throws Exception {
        // insert code here...
        System.out.println("Hello World!");
				
		String host = "localhost",
			db = "schedule",
			user = "yukiko",
			pass = "nishimura",
			file = "valuate2.txt";
		switch (args.length) {
			case 5:
				db = args[1];
				user = args[2];
				pass = args[3];
				file = args[4];
			case 1:
				host = args[0];
			default:
				break;
		}
		
		Timetable timetable = new Timetable(host,db,user,pass,file);
	    timetable.initsol();
		long start, stop, time;
        java.text.DecimalFormat df2 =new java.text.DecimalFormat("00");
        java.text.DecimalFormat df3 =new java.text.DecimalFormat("000");
        System.out.println("Measuring time...");
        start = System.currentTimeMillis();
		System.out.println(timetable.count_offence("timetableSQL"));
		System.out.println(timetable.valuateFile());
		//timetable.search_move();
		//timetable.search_change();
		//timetable.search_swap();
		//timetable.search_add();
		//timetable.search_leave();
		timetable.local_search();
		System.out.println(timetable.count_offence("timetableSQL"));
		System.out.println(timetable.valuateFile());
        stop = System.currentTimeMillis();
        System.out.println("Ending time measurement.");
        time = stop - start;
        int miri = (int)(time%1000);
        int scd = (int)(time/1000%60);
        int mini = (int)(time/1000/60%60);
        int hour = (int)(time/1000/60/60);
        String date = hour+":"+df2.format(mini)+":"+df2.format(scd)+"."+df3.format(miri);
        System.out.println("Execution time : " + date);
        timetable.close();
	
		return;
    }
}
