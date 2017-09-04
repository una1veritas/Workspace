package data;

import java.sql.*;
import java.util.*;
import java.io.*;

//import common.DBConnectionPool;


public class Timetable implements Serializable {	
	//(task, period, processor)
	HashSet triplets;

	HashMap tasks;
	HashMap periods;
	HashMap processors;
	
	HashMap taskPeriod; // opportunity
	HashMap processorPeriod; //availability
	HashMap qualifications; // qualification
	
	
	public Timetable(Collection lectures, Collection periods, Collection teachers, HashMap opportunities, HashMap availablities, HashMap qualifications) {
		Task aTask;
		Period aPeriod;
		Processor aProcessor;
		
		tasks = new HashMap(lectures.size());
		for (Iterator i = lectures.iterator(); i.hasNext(); ) {
			aTask = (Task) i.next();
			tasks.put(aTask, aTask);
		}
		processors = new HashMap(teachers.size()); 
		for (Iterator i = teachers.iterator(); i.hasNext(); ) {
			aProcessor = (Processor) i.next();
			processors.put(aProcessor, aProcessor);
		}
		this.periods = new HashMap(periods.size());
		for (Iterator i = periods.iterator(); i.hasNext(); ) {
			aPeriod = (Period) i.next();
			this.periods.put(aPeriod, aPeriod);
		}
		this.triplets = new HashSet();
		this.taskPeriod = opportunities;
		this.processorPeriod = availablities;
		this.qualifications = qualifications;
	}
	
	
	public Timetable() {
		
	}
	
	//
	public Set processors() {
		return processors.keySet();
	}
	
	public Set tasks() {
		return tasks.keySet();
	}
	
	public Set periods() {
		return periods.keySet();
	}
	
	//
	public Task getTask(Task t) {
		return (Task) tasks.get(t);
	}
	
	public Task getTask(int id) {
		return getTask(new Task(id));
	}
	
	public Processor getProcessor(Processor p) {
		return (Processor) processors.get(p);
	}
	
	public Processor getProcessor(int id) {
		return getProcessor(new Processor(id));
	}
	
	public Period getPeriod(Period p) {
		return (Period) periods.get(p);
	}
	
	public Period getPeriod(int id) {
		return getPeriod(new Period(id));
	}
	
	//
	public HashSet taskOpportunities(Task t) {
		return (HashSet) taskPeriod.get(t);
	}
	
	//
	public boolean processorQualified(Processor p, Task t) {
		if (qualifications.containsKey(p)) {
			return ((HashSet) qualifications.get(p)).contains(new Integer(t.qualification_id()));
		} else {
			return false;
		}
	}

	//
	public boolean processorAvailableAtAll(Processor p, Collection c) {
		if (processorPeriod.containsKey(p)) {
			return ((HashSet) processorPeriod.get(p)).containsAll(c);
		} else {
			return false;
		}
	}
	
	//
	public boolean processorAvailableAt(Processor p, Period d) {
		if (processorPeriod.containsKey(p)) {
			return ((HashSet) processorPeriod.get(p)).contains(d);
		} else {
			return false;
		}
	}
	
	public boolean taskAvailableAt(Task t, Period d) {
		if (taskPeriod.containsKey(t)) {
			return ((HashSet) taskPeriod.get(t)).contains(d);
		} else {
			return false;
		}
	}
	
	public boolean taskAvailableAtAll(Task t, Collection c) {
		if (taskPeriod.containsKey(t)) {
			return ((HashSet) taskPeriod.get(t)).containsAll(c);
		} else {
			return false;
		}
	}
	
	public int getNextPeriod(int id){
		Period dp = getPeriod(id);
		if( !(dp.next() < 0) ){
			return dp.next();
		}
		return -1;
	}
	
	public ArrayList getFollowingPeriodsInTheSameDay(Period startp, int max) {
		ArrayList result = new ArrayList();
		Iterator i = periods().iterator();
		Period dp = getPeriod(startp);
		if ( dp != null && result.size() < max ) {
			result.add(dp);
		}

		while ( dp != null && dp.next() < 0 && (result.size() < max) ) {
			Period nextp = getPeriod(dp.next());
			result.add(nextp);
		}
		return result;
	}

	public ArrayList allPeriodsNecessaryFor(Triplet t) {
		int len = (getTask(t.task())).length();
		return getFollowingPeriodsInTheSameDay(t.period(), len);
	}
	
	
	//period に task を担当している processor を返す
	public Set assignedProcessorsOn(Task tsk, Period prd){
		HashSet result = new HashSet();
		Iterator i = triplets.iterator();
		while(i.hasNext()){
			Triplet t = (Triplet) i.next();
			if ( tsk.equals(t.task()) && prd.equals(t.period())) {
				result.add(t.processor());
			}
		}
		retur