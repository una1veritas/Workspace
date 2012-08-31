package data;

//import java.sql.*;
import java.util.*;
//import java.io.*;

//import common.DBConnectionPool;


public class Timetable {	
	//(task, period, processor)
	HashSet<Triplet> triplets;

	HashMap<Task,Task> tasks;
	HashMap<Period,Period> periods;
	HashMap<Processor,Processor> processors;
	
	HashMap<Task,HashSet<Period>> taskPeriod; // opportunity
	HashMap<Processor,HashSet<Period>> processorPeriod; //availability
	HashMap<Processor,HashSet<Integer>> qualifications; // qualification
	
	
	public Timetable(Collection<Task> lectures, 
					Collection<Period> periods, 
					Collection<Processor> teachers, 
					HashMap<Task,HashSet<Period>> opportunities, 
					HashMap<Processor,HashSet<Period>> availablities, 
					HashMap<Processor,HashSet<Integer>> qualifications) {
		Task aTask;
		Period aPeriod;
		Processor aProcessor;
		
		tasks = new HashMap<Task,Task>(lectures.size());
		for (Iterator<Task> i = lectures.iterator(); i.hasNext(); ) {
			aTask = i.next();
			tasks.put(aTask, aTask);
		}
		processors = new HashMap<Processor,Processor>(teachers.size()); 
		for (Iterator<Processor> i = teachers.iterator(); i.hasNext(); ) {
			aProcessor = i.next();
			processors.put(aProcessor, aProcessor);
		}
		this.periods = new HashMap<Period,Period>(periods.size());
		for (Iterator<Period> i = periods.iterator(); i.hasNext(); ) {
			aPeriod = i.next();
			this.periods.put(aPeriod, aPeriod);
		}
		this.triplets = new HashSet<Triplet>();
		this.taskPeriod = opportunities;
		this.processorPeriod = availablities;
		this.qualifications = qualifications;
	}
	
	
	public Timetable() {
		
	}
	
	//
	public Set<Processor> processors() {
		return processors.keySet();
	}
	
	public Set<Task> tasks() {
		return tasks.keySet();
	}
	
	public Set<Period> periods() {
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
	public Set<Period> taskOpportunities(Task t) {
		return taskPeriod.get(t);
	}
	
	//
	public boolean processorQualified(Processor p, Task t) {
		if (qualifications.containsKey(p)) {
			return qualifications.get(p).contains(new Integer(t.qualification_id()));
		} else {
			return false;
		}
	}

	//
	public boolean processorAvailableAtAll(Processor p, Collection<Period> c) {
		if (processorPeriod.containsKey(p)) {
			return processorPeriod.get(p).containsAll(c);
		} else {
			return false;
		}
	}
	
	//
	public boolean processorAvailableAt(Processor p, Period d) {
		if (processorPeriod.containsKey(p)) {
			return processorPeriod.get(p).contains(d);
		} else {
			return false;
		}
	}
	
	public boolean taskAvailableAt(Task t, Period d) {
		if (taskPeriod.containsKey(t)) {
			return taskPeriod.get(t).contains(d);
		} else {
			return false;
		}
	}
	
	public boolean taskAvailableAtAll(Task t, Collection<Period> c) {
		if (taskPeriod.containsKey(t)) {
			return taskPeriod.get(t).containsAll(c);
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
	
	public ArrayList<Period> getFollowingPeriodsInTheSameDay(Period startp, int max) {
		ArrayList<Period> result = new ArrayList<Period>();
//		Iterator<Period> i = periods().iterator();
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

	public ArrayList<Period> allPeriodsNecessaryFor(Triplet t) {
		return getFollowingPeriodsInTheSameDay(t.period(), getTask(t.task()).length());
	}
	
	//period に task を担当している processor を返す
	public HashSet<Processor> assignedProcessorsOn(Task tsk, Period prd){
		HashSet<Processor> result = new HashSet<Processor>();
		Iterator<Triplet> i = triplets.iterator();
		while(i.hasNext()){
			Triplet t = i.next();
			if ( tsk.equals(t.task()) && prd.equals(t.period())) {
				result.add(t.processor());
			}
		}
		return result;
	}
	
	public void clear() {
		triplets.clear();
		tasks.clear();
		periods.clear();
		processors.clear();		
		taskPeriod.clear();
		processorPeriod.clear();
		qualifications.clear();
	}
	
	public HashSet<Triplet> triplets() {
		return triplets;
	}
	
	public boolean has(Triplet triple) {
		return triplets.contains(triple);
	}
}