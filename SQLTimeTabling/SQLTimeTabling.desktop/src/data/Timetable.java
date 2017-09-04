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
		return result;
    }

    public Set assignedProcessorsOn(Task tsk, Period prd, Collection c)
    {
        HashSet result = new HashSet();
        Iterator i = c.iterator();
        do
        {
            if(!i.hasNext())
                break;
            Triplet t = (Triplet)i.next();
            if(tsk.equals(t.task()) && prd.equals(t.period()))
                result.add(t.processor());
        } while(true);
        return result;
    }

    public String toString()
    {
        StringBuffer buf = new StringBuffer("Timetable(");
        for(Iterator i = triplets.iterator(); i.hasNext(); buf.append(", "))
        {
            Triplet t = (Triplet)i.next();
            buf.append(t.toString());
        }
        buf.append(") ");
        return buf.toString();
    }

    public String toStoreString()
    {
        StringBuffer buf = new StringBuffer("");
        Triplet t;
        for(Iterator i = triplets.iterator(); i.hasNext(); buf.append(t.toString() + System.getProperty("line.separator")))
            t = (Triplet)i.next();

        return buf.toString();
    }

    public void clear()
        throws Exception
    {
        triplets.clear();
    }

    public void insertTriplet(Task task, Period period, Processor processor)
    {
        triplets.add(new Triplet(task, period, processor));
    }

    public void insertTriple(int taskid, int periodid, int processorid)
    {
        triplets.add(new Triplet(getTask(taskid), getPeriod(periodid), getProcessor(processorid)));
    }

    public void deleteTriplet(Task task, Period period, Processor processor)
    {
        triplets.remove(new Triplet(task, period, processor));
    }

    public void deleteTriple(int taskid, int periodid, int processorid)
    {
        deleteTriplet(getTask(taskid), getPeriod(periodid), getProcessor(processorid));
    }

    public HashSet triplets()
    {
        return triplets;
    }

    public boolean has(Triplet t)
    {
        return triplets.contains(t);
    }

    public int countViolations()
    {
        return countViolations(null);
    }

    public int countViolations(PrintWriter pw)
    {
        int notQualified = 0;
        int notAtAvailableProcessor = 0;
        int notAtAvailableTask = 0;
        int duplicatedTasks = 0;
        int insufficientTasks = 0;
        int ignoredTasks = 0;
        int multipleTeach = 0;
        ArrayList allTasks = new ArrayList(tasks());
        ArrayList teacherlist = new ArrayList(processors());
        HashMap counters = new HashMap(tasks().size());
        Iterator i;
        for(i = allTasks.iterator(); i.hasNext(); counters.put((Task)i.next(), new HashSet()));
        Triplet tp;
        for(i = triplets.iterator(); i.hasNext(); ((HashSet)counters.get(tp.task())).add(tp.period()))
        tp = (Triplet)i.next();
        i = allTasks.iterator();
        do
        {
            if(!i.hasNext())
                break;
            Task t = (Task)i.next();
            int sufficientCount = 0;
            int insufficientCount = 0;
            HashSet periodByTask = (HashSet)counters.get(t);
            if(periodByTask.size() == 0)
            {
                ignoredTasks++;
                if(pw != null)
                    pw.print("ignored: " + t + ", ");
            } else
            {
                for(Iterator j = periodByTask.iterator(); j.hasNext();)
                {
                    Period pd = (Period)j.next();
                    if(assignedProcessorsOn(t, pd).size() < t.getProcessors_lb())
                        insufficientCount++;
                    else
                        sufficientCount++;
                }

                if(sufficientCount > 1)
                {
                    duplicatedTasks += sufficientCount - 1;
                    if(pw != null)
                        pw.print("duplicating: " + t + " in " + sufficientCount + " time, ");
                }
            }
        } while(true);
        counters.clear();
        counters = new HashMap(processors().size());
        for(i = triplets.iterator(); i.hasNext();)
        {
            Triplet t = (Triplet)i.next();
            Processor tch = t.processor();
            if(counters.containsKey(tch))
            {
                Iterator ip = allPeriodsNecessaryFor(t).iterator();
                while(ip.hasNext()) 
                {
                    Period p = (Period)ip.next();
                    if(((Set)counters.get(tch)).contains(p))
                    {
                        multipleTeach++;
                        if(pw != null)
                            pw.print("do more than one in the same time: " + t + ", ");
                    }
                    ((Set)counters.get(tch)).add(p);
                }
            } else
            {
                HashSet set = new HashSet();
                set.addAll(allPeriodsNecessaryFor(t));
                counters.put(tch, set);
            }
        }

        counters.clear();
        i = triplets.iterator();
        do
        {
            if(!i.hasNext())
                break;
            Triplet t = (Triplet)i.next();
            Processor tch = t.processor();
            Task lec = t.task();
            if(tch != null && lec != null)
            {
                if(!processorQualified(tch, lec))
                {
                    notQualified++;
                    if(pw != null)
                        pw.print("without qualification: " + t + ", ");
                }
                if(!processorAvailableAtAll(tch, allPeriodsNecessaryFor(t)))
                {
                    notAtAvailableProcessor++;
                    if(pw != null)
                        pw.print("do not teach at: " + t + ", ");
                }
                if(!taskAvailableAtAll(lec, allPeriodsNecessaryFor(t)))
                {
                    notAtAvailableTask++;
                    if(pw != null)
                        pw.print("cannot place lecture at: " + t + ", ");
                }
            }
        } while(true);
        return notQualified + notAtAvailableProcessor + notAtAvailableTask + ignoredTasks + duplicatedTasks + multipleTeach;
    }
}
