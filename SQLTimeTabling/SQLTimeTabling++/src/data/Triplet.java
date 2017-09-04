package data;

import java.io.*;

public class Triplet implements Comparable {
	private Task task;
	private Period period;
	private Processor processor;
	
	
	private Triplet() {}
	
	public Triplet(Task l, Period p, Processor t) {
		task = l;
		period = p;
		processor = t;
	}
	
	
	public Triplet(Triplet t){
		task = t.task;
		period = t.period;
		processor = t.processor;
	}
	
	public Period period() {
		return period;
	}
	
	private void period(Period p) {
		this.period = p;
	}
	
	public Processor processor() {
		return processor;
	}

	private void processor(Processor p) {
		this.processor = p;
	}
	
	public Task task() {
		return task;
	}
	
	private void task(Task t) {
		this.task = t;
	}
	
	public String toString(){
		return "("+task.id()+", "+period.id()+", "+processor.id()+")";
	}
	
	public boolean equals(Object o){
		if(!(o instanceof Triplet))
			return false;
		Triplet ttr = (Triplet) o;
		return ttr.task.equals(task) && ttr.period.equals(period) && ttr.processor.equals(processor);
	}
	
	public int compareTo(Object o) {
		if (! (o instanceof Triplet) )
			return -((Comparable) o).compareTo(this);
		if (task == null && ((Triplet) o).task != null)
			return -1;
		else if (task != null && ((Triplet) o).task == null)
			return 1;
		if ( !(task == null && ((Triplet) o).task == null) 
			&& !task.equals(((Triplet) o).task) )
			return task.id() - ((Triplet) o).task.id();
		if (period == null && ((Triplet) o).period != null)
			return -1;
		else if (period != null && ((Triplet) o).period == null)
			return 1;
		if (  !(period == null && ((Triplet) o).period == null) 
			&& !period.equals(((Triplet) o).task) )
			return period.compareTo(((Triplet) o).period);
		if (processor == null && ((Triplet) o).processor != null)
			return -1;
		else if (processor != null && ((Triplet) o).processor == null)
			return 1;
		if ( !(processor == null && ((Triplet) o).processor == null) 
			&& !processor.equals(((Triplet) o).processor) )
			return processor.id() - ((Triplet) o).processor.id();
		return 0;
	}
	
	public int hashCode(){
		return (task.id() << 12) ^ (period.id() << 7) ^ processor.id();
	}
}
