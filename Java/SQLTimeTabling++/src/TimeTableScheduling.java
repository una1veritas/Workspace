import java.io.*;
import data.*;
import java.sql.*;
import java.util.*;

//import org.apache.commons.lang.time.StopWatch;

//import common.DBConnectionPool;

public class TimeTableScheduling {
	private ArrayList<SQLRule> rules;
	private Timetable timetable;
	
	transient Connection db;
	transient Statement state;
	transient PreparedStatement prestate;
	transient ResultSet results;
	transient int updates;
	transient int issuedQueries;
	
	//コンストラクタ
	public TimeTableScheduling(String host,String dbname, String user, String passwd, String ruleTableName, int project_id) throws Exception{
		String url = "jdbc:mysql://"+host+"/"+dbname+"?useUnicode=true&characterEncoding=utf8";
		openConnection(url, user, passwd);
		timetable = readTimetableFromDBTable(project_id);
		initializeDBTimeTable(project_id);
		readRulesFromDBTable(ruleTableName, project_id);
	}
	
	
	public void storeTimeTableOnDBTable(String tbl, int project_id, String tstamp ) throws Exception {
		StringWriter w = new StringWriter();
		reportTo(new PrintWriter(w));
		
		ByteArrayOutputStream byteout = new ByteArrayOutputStream();
		ObjectOutputStream objout = new ObjectOutputStream(byteout);
		objout.writeObject(timetable);
		ByteArrayInputStream bytein = new ByteArrayInputStream(byteout.toByteArray());
		
		String sql = "INSERT INTO "+ tbl + " (project_id, totalpenalty, violations, object, timetable, note, datasource) "+" VALUES (?, ?, ?, ?, ?, ?, ?)";
		//System.err.println(sql);
		setPreparedStatement(sql);
		setStatementValue(1, project_id);
		setStatementValue(2, cost());
		setStatementValue(3, violations());
		setStatementValue(4, bytein, byteout.size());
		setStatementValue(5, timetable.toString());
		setStatementValue(6, w.toString());
		setStatementValue(7, tstamp);
		executePreparedUpdate();
				
		return;
	}
	
	void initializeDBTimeTable(int project_id) throws Exception {
		String sql = "CREATE TEMPORARY TABLE IF NOT EXISTS timetablesql (task_id int, period_id int, processor_id int, UNIQUE KEY task_id (task_id, period_id, processor_id))";
		executeUpdate(sql); // an empty time table.
		
		sql = "SET @project_id = '"+project_id+"'";
		executeUpdate(sql);
		
	}
	
	public void loadTimeTableOnDBTable(String tblName, int project_id) throws Exception {
		//sql = "select * from rules where project_id = "+project_id+" and use_flag = 1 order by sql_id";
		String sql = "select * from "+ tblName + " where project_id = "+project_id+" order by serial";
		executeQuery(sql);
		while(results().next()){
			//System.err.println(results().getInt("weight"));
			//System.err.println(results().getString("description"));
			String lines = results().getString("note");
			//String lines = rs.getString("description");
			//System.err.println("No. "+results().getInt("serial")+", note: "+lines.length());
			//System.err.println();
		}
		//System.err.println("Issued. ");
		closeResults();
	}
	
	
	
	private Timetable readTimetableFromDBTable(int project_id) throws Exception{
		HashMap<Processor,Processor> teachers = new HashMap<Processor,Processor>();
		HashMap<Task,Task> lectures = new HashMap<Task,Task>();
		ArrayList<Period> periods = new ArrayList<Period>();
		
		HashMap<Processor,HashSet<Period>> availabilities = new HashMap<Processor,HashSet<Period>>();
		HashMap<Processor,HashSet<Integer>> qualifications = new HashMap<Processor,HashSet<Integer>>();
		HashMap<Task,HashSet<Period>> opportunities = new HashMap<Task,HashSet<Period>>();
		String sql;
		
		//講義の情報の読み取り
		//属性が入力されている講義の情報の取得
		sql = "select * from tasks a, task_properties b where a.task_id = b.task_id and a.project_id = "+project_id+" order by a.task_id";
		executeQuery(sql);
		while(results().next()){
			Task templ = new Task(results().getInt("task_id"), 
										results().getInt("required_processors_lb"), 
										results().getInt("required_processors_ub"), 
										results().getInt("qualification_id"),
										results().getInt("task_series_num"));
			lectures.put(templ, templ);
		}
		
		//時間の読み取り
		periods = new ArrayList<Period>();
		sql = "select pp.period_id, pp.day_id , h.number as hint from period_properties pp, days d, hours h where pp.day_id = d.day_id and pp.hour_id = h.hour_id and pp.project_id = "+project_id+" order by d.number,d.day_id,h.number,h.hour_id";
		executeQuery(sql);
		while(results().next()) {
			Period dp = new Period(results().getInt("period_id"), results().getInt("day_id"), results().getInt("hint"), -1);
			//System.err.println(dp);
			periods.add(dp);
		}
		for (int ix = 0; ix + 1 < periods.size(); ix++) {
			if ( ((Period) periods.get(ix)).day() == ((Period) periods.get(ix+1)).day() )
				((Period) periods.get(ix)).next( ((Period) periods.get(ix+1)).id()) ;
		}

		//講師の情報の初期化
		//属性が入力されている講師のprocessor_idを取得
		sql = "select a.processor_id from processors a, processor_properties b where a.processor_id = b.processor_id and a.project_id ="+project_id+" order by a.processor_id";
		executeQuery(sql);
		while(results().next()) {
			Processor tmpt = new Processor(results().getInt("processor_id") /*, new HashSet(), new HashSet() */);
			teachers.put(tmpt,tmpt);
		}
		
		//processor_idが担当可能な時間を取得
		for (Iterator<Processor> i = teachers.keySet().iterator(); i.hasNext(); ) {
			Processor teacher = i.next();
			availabilities.put(teacher, new HashSet<Period>());
			qualifications.put(teacher, new HashSet<Integer>());
			String sql2 = "select period_id, preferred_level_proc from processor_schedules where processor_id = '" + teacher.id() + "' order by period_id";
			executeQuery(sql2);
			while(results().next()){
				Period pd = new Period(results().getInt("period_id"));
				//System.err.println(periods);
				Period ppp = (Period) periods.get(periods.indexOf(pd));
				//System.err.println(ppp);
				availabilities.get(teacher).add(ppp);
			}
			//processor_idが担当可能な講義の種類を取得
			sql2 = "select qualification_id from processor_qualification where processor_id = '" + teacher.id() + "' order by qualification_id";
			executeQuery(sql2);
			while(results().next()){
				Integer qid = new Integer(results().getInt("qualification_id"));
				//System.err.println(qid);
				qualifications.get(teacher).add(qid);
			}
			//t.setQualifications(qs);
			//System.err.println(t);
		}
		System.err.print("Loading teachers' info, ");
		
		
		//task_idが開講可能な時間を取得
		for (Iterator<Task> i = lectures.keySet().iterator(); i.hasNext(); ) {
			Task l = i.next();
			opportunities.put(l, new HashSet<Period>());
			String sql2 = "select period_id , preferred_level_task from task_opportunities where task_id = '" + l.id() + "' order by period_id";
			executeQuery(sql2);
			while(results().next()){
				Period pd = (Period) periods.get(periods.indexOf(new Period(results().getInt("period_id"))));
				opportunities.get(l).add(pd);
			}
		}
		System.err.print("lectures' info, ");

		//if(debug) System.out.println("時間の初期化終了");
		System.err.println("periods' info completed.");
		closeResults();
		//System.err.println("Processors: "+teachers.values());
		//System.err.println("Tasks: "+teachers.values());
		//System.err.println("Periods: "+periods);
		return new Timetable(lectures.values(), periods, teachers.values(), 
							 opportunities, availabilities, qualifications);
	}

	//データベースから評価SQLを取得する
	private void readRulesFromDBTable(String table, int project_id) throws Exception {
		rules = new ArrayList<SQLRule>();
		ArrayList create_list = new ArrayList();
		
		String sql = "select * from "+ table + " where project_id = "+project_id+" and use_flag = 1 order by weight desc, sql_id";
		executeQuery(sql);
		while(results().next()){
			SQLRule temp = new SQLRule();
			temp.weight(results().getInt("weight"));
			temp.comment(results().getString("description"));
			String line = results().getString("valuate_sql");
			String [] sqls = line.split(";\\s*");
			for (int lnum = 0; lnum < sqls.length; lnum++) {
				//System.err.println("SQL "+ lnum + ": " + sqls[lnum]);
				if(sqls[lnum].equals(""))
					continue;
				temp.addSQL(sqls[lnum]); 
			}
			//System.err.println();
			if(temp.queries().size()!=0) {
				rules.add(temp);
			}
		}
	}


	//解の評価値を	
	public void reportTo(PrintWriter pw) throws Exception{
		SQLRule t = null;
		
		pw.println("Total violations of rules: " + violations());
		timetable.countViolations(pw);
		pw.println();
		for (Iterator<SQLRule> i = rules.iterator(); i.hasNext(); ){
			t = i.next();
			int num = 0;
			for (Iterator<String> s = t.queries().iterator(); s.hasNext(); ) {
				if ( !execute(s.next()) )
					continue;
				if (results().next()) {
					num = num + results().getInt("penalties");
					pw.println(""+num+ " unsatisfied request \"" + t + "\". ");
					pw.println("Subtotal: "+ (num*t.weight()) + " " +t.comment());
					pw.println();
				}
			}
			//}
		}
	}
	
	
	//初期解作成
	public void makeInitialSolution(int project_id) throws Exception {
		String sql;

		//時間割の初期化
		timetable.clear();
		sql = "delete from timetablesql";
		executeUpdate(sql);
		
		closeResults();
	}
	
	
	//局所探索
	public int local_search() throws Exception {
		int i;
		for (i=1; true; i++) {
			System.err.println("violation: "+ violations() + /* ", offences: " + timetable.countOffences() + */ ", cost: "+cost());
			System.err.println(timetable);
			System.err.println();
			if (add_task()
				|| change_component() 
				// || change_processor()
				// || change_task()
				|| swap_processors()
				|| delete_task()
				// || search_leave_task_series()
				) {
				continue;
			}
			break;
		}
		return i;
	}
	
	
	public boolean add_task() throws Exception{
		boolean changed = false;
		Random rand = new Random(Calendar.getInstance().getTimeInMillis());
		int curr_cost = cost();
		int curr_viols = violations();
		
		HashMap missingTasks = new HashMap(timetable.tasks().size());
		for (Iterator i = timetable.tasks().iterator(); i.hasNext(); ) {
			Task l = (Task) i.next();
			missingTasks.put(l, new Integer(l.getProcessors_ub()));
		}
		
		ArrayList candidates = new ArrayList();
		for (Iterator i = missingTasks.keySet().iterator(); i.hasNext(); ) {
			Task aTask = (Task) i.next(); //System.err.println("For each missing: "+ aTask);
			for (Iterator t = timetable.processors().iterator(); t.hasNext(); ) {
				Processor aProcessor = (Processor) t.next();
				for(Iterator p = timetable.taskOpportunities(aTask).iterator(); p.hasNext(); ) {
					Period aPeriod = (Period) p.next();
					//
					Triplet newTablet = new Triplet(aTask, aPeriod, aProcessor);
					if ( timetable.has(newTablet) )
						continue;
					if ( timetable.processorQualified(aProcessor, aTask) 
						&& timetable.processorAvailableAtAll(aProcessor, timetable.allPeriodsNecessaryFor(newTablet)) ) {
						candidates.add(newTablet);
					}
				}
			}
		}
		if ( candidates.isEmpty() ) 
			return false; // changed == false		
		int i, s;
		for (i = 0, s = rand.nextInt(candidates.size()); i < candidates.size(); i++) {
			Triplet aTablet = (Triplet) candidates.get((s+i) % candidates.size());
			insertTriplet(aTablet);
			int updatedViols = violations();
			if (updatedViols < curr_viols ||
				(updatedViols == curr_viols && cost() < curr_cost) ) {
				System.err.println("Adding "+aTablet);
				changed = true;
				return changed;
			} else {
				deleteTriplet(aTablet);
			}
		}
		return changed;
	}
	
	
	public boolean change_processor() throws Exception{
		boolean changed = false;
		Random rand = new Random(Calendar.getInstance().getTimeInMillis());
		int curr_viols = violations();
		int curr_cost = cost();
				
		ArrayList candidates = new ArrayList();
		for (Iterator orgt = timetable.tasks().iterator(); orgt.hasNext(); ) {
			Processor aProcessor = (Processor) orgt.next();
			for (Iterator tblt = timetable.triplets().iterator(); tblt.hasNext(); ) {
				Triplet aTablet = (Triplet) tblt.next();
				if (aTablet.processor().id() != aProcessor.id())
					continue;
				Task aTask = (Task) timetable.getTask(aTablet.task());
				Iterator t = timetable.processors().iterator(); 
				while ( t.hasNext() ) {
					Processor anotherProcessor = (Processor) t.next();
					if (anotherProcessor.equals(aProcessor))
						continue;
					if ( timetable.has(new Triplet(aTablet.task(), aTablet.period(), anotherProcessor)) )
						continue;
					if (timetable.processorQualified(anotherProcessor, aTask) 
						&& timetable.processorAvailableAtAll(anotherProcessor, timetable.allPeriodsNecessaryFor(aTablet)) ) {
						candidates.add(aTablet);
						candidates.add(new Triplet(aTablet.task(), aTablet.period(), anotherProcessor));
					}
				}
			}
		}
		if ( candidates.isEmpty() ) 
			return false; // changed == false		
		//for (Iterator i = candidates.iterator(); i.hasNext(); ) {
		Triplet oldTablet = null;
		Triplet newTablet = null;
		int i, s;
		for (i = 0, s = (rand.nextInt(candidates.size()) / 2)*2; i < candidates.size(); i += 2) {
			Triplet orgTablet = (Triplet) candidates.get((s+i) % candidates.size());
			Triplet anotherTablet = (Triplet) candidates.get((s+i+1) % candidates.size());
			deleteTriplet(orgTablet);
			insertTriplet(anotherTablet);
			int updatedViols = violations();
			int updatedCost = cost();
			if (updatedViols < curr_viols ||
				(updatedViols == curr_viols && updatedCost < curr_cost) ) {
				oldTablet = orgTablet;
				newTablet = anotherTablet;
				changed = true;
				curr_viols = updatedViols;
				curr_cost = updatedCost;
				break;
			}
			deleteTriplet(anotherTablet);
			insertTriplet(orgTablet);
		}
		if (changed) {
		//	deleteTablet(oldTablet);
		//	insertTablet(newTablet);
			System.err.println("Processor switching "+oldTablet+" to "+ newTablet);
		} 
		return changed;
	}
	
	
	public boolean change_task() throws Exception{
		boolean changed = false;
		Random rand = new Random(Calendar.getInstance().getTimeInMillis());
		int curr_viols = violations();
		int curr_cost = cost();
		
		ArrayList candidates = new ArrayList();
		for (Iterator orgt = timetable.processors().iterator(); orgt.hasNext(); ) {
			Processor aProcessor = (Processor) orgt.next();
			for (Iterator tblt = timetable.triplets().iterator(); tblt.hasNext(); ) {
				Triplet aTablet = (Triplet) tblt.next();
				if (aTablet.processor().id() != aProcessor.id())
					continue;
				Task aTask = (Task) timetable.getTask(aTablet.task());
				Iterator l = timetable.tasks().iterator(); 
				while ( l.hasNext() ) {
					Task anotherTask = (Task) l.next();
					if (anotherTask.equals(aTask))
						continue;
					if ( timetable.has(new Triplet(anotherTask, aTablet.period(), aTablet.processor())) )
						continue;
					if ( timetable.processorQualified(aProcessor, anotherTask) 
						&& timetable.processorAvailableAtAll(aProcessor, timetable.allPeriodsNecessaryFor(new Triplet(anotherTask, aTablet.period(), aTablet.processor())))) {
						candidates.add(aTablet);
						candidates.add(new Triplet(anotherTask, aTablet.period(), aTablet.processor()));
					}
				}
			}
		}
		if ( candidates.isEmpty() ) 
			return false; // changed == false		
		//for (Iterator i = candidates.iterator(); i.hasNext(); ) {
		Triplet oldTablet = null;
		Triplet newTablet = null;
		int i, s;
		for (i = 0, s = (rand.nextInt(candidates.size()) / 2)*2; i < candidates.size(); i += 2) {
			Triplet orgTablet = (Triplet) candidates.get((s+i) % candidates.size());
			Triplet anotherTablet = (Triplet) candidates.get((s+i+1) % candidates.size());
			deleteTriplet(orgTablet);
			insertTriplet(anotherTablet);
			int updatedViols = violations();
			int updatedCost = cost();
			if (updatedViols < curr_viols ||
				(updatedViols == curr_viols && updatedCost < curr_cost) ) {
				oldTablet = orgTablet;
				newTablet = anotherTablet;
				changed = true;
				curr_viols = updatedViols;
				curr_cost = updatedCost;
				break;
			}
			deleteTriplet(anotherTablet);
			insertTriplet(orgTablet);
		}
		if (changed) {
			//	deleteTablet(oldTablet);
			//	insertTablet(newTablet);
			System.err.println("Task switching "+oldTablet+" to "+ newTablet);
		} 
		return changed;
	}
	
	
	public boolean change_component() throws Exception {
		boolean changed = false;
		Random rand = new Random(Calendar.getInstance().getTimeInMillis());
		//
		int curr_viols = violations();
		int curr_cost = cost();
		Triplet triple = null, newTriplet = null;
		
		ArrayList tarray = new ArrayList(timetable.processors());
		ArrayList parray = new ArrayList(timetable.periods());
		ArrayList larray = new ArrayList(timetable.tasks());
		ArrayList triplets = new ArrayList(timetable.triplets());
		Collections.shuffle((List) triplets, rand);
		
		ArrayList candidates = new ArrayList();
		// purturb someting...
		for (Iterator cti = triplets.iterator(); !changed && cti.hasNext(); ) {
			triple = (Triplet) cti.next();
			candidates.clear();
			// try to change teacher
			for (Iterator i = tarray.iterator(); i.hasNext(); ) {
				Processor another = (Processor) i.next();
				if ( another.equals(triple.processor()) )
					continue;
				if ( timetable.processorAvailableAt(another, triple.period()) 
					&& timetable.processorQualified(another, triple.task()))
					continue;
				candidates.add(new Triplet(triple.task(), triple.period(), another));
			}
			// change period
			for (Iterator i = parray.iterator(); i.hasNext(); ) {
				Period another = (Period) i.next();
				if ( another.equals(triple.period()) )
					continue;
				if ( timetable.processorAvailableAt(triple.processor(), another)
					&& timetable.taskAvailableAt(triple.task(), another) )
					candidates.add(new Triplet(triple.task(), another, triple.processor()) );
			}
			// change lecture
			for (Iterator i = larray.iterator(); i.hasNext(); ) {
				Task another = (Task) i.next();
				if ( another.equals(triple.task()) )
					continue;
				if ( timetable.taskAvailableAt(another, triple.period()) 
					&& timetable.processorQualified(triple.processor(), another) )
					continue;
				candidates.add(new Triplet(another, triple.period(), triple.processor()) );
			}
			//
			if ( candidates.isEmpty() ) 
				continue;
			int i, s;
			for (i = 0, s = rand.nextInt(candidates.size()); i < candidates.size(); i++) {
				newTriplet = (Triplet) candidates.get((s+i) % candidates.size());
				if ( timetable.has(newTriplet) )
					continue;
				deleteTriplet(triple);
				insertTriplet(newTriplet);
				int updatedViols = violations();
				int updatedCost = cost();
				if (updatedViols < curr_viols ||
					(updatedViols == curr_viols && updatedCost < curr_cost) ) {
					changed = true;
					//curr_viols = updatedViols;
					//curr_cost = updatedCost;
					break;
				}
				deleteTriplet(newTriplet);
				insertTriplet(triple);
			}
		}
		
		if (changed) {
			//	deleteTablet(oldTablet);
			//	insertTablet(newTablet);
			System.err.println("Change something from "+triple+" to "+ newTriplet);
		} 
		return changed;
	}
	
	
	public boolean swap_processors() throws Exception{
		boolean changed = false;
		Random rand = new Random(Calendar.getInstance().getTimeInMillis());
		int curr_viols = violations();
		int curr_cost = cost();
		
		ArrayList candidates = new ArrayList();
		for (Iterator i1 = timetable.triplets().iterator() ; i1.hasNext(); ) {
			Triplet t1 = (Triplet) i1.next();
			for (Iterator i2 = timetable.triplets().iterator() ; i2.hasNext(); ) {
				Triplet t2 = (Triplet) i2.next();
				if ( (t1.processor().id() >= t2.processor().id()) 
					|| (t1.task().id() >= t2.task().id()) ) 
					continue;
				Task l1 = (Task) timetable.getTask(t1.task());
				Task l2 = (Task) timetable.getTask(t2.task());
				Processor p1 = (Processor) timetable.getProcessor(t1.processor());
				Processor p2 = (Processor) timetable.getProcessor(t2.processor());
				if (! timetable.processorQualified(p1, l2) 
					|| ! timetable.processorAvailableAtAll(p1, timetable.allPeriodsNecessaryFor(t2))
					|| ! timetable.processorQualified(p2, l1) )
					continue;
				if ( timetable.has(new Triplet(t1.task(), t1.period(), t2.processor())) || timetable.has(new Triplet(t2.task(), t2.period(), t1.processor())) ) 
					continue;
				candidates.add(t1);
				candidates.add(t2);
			}
		}
		
		if ( candidates.isEmpty() ) 
			return false; // changed == false		
		//for (Iterator i = candidates.iterator(); i.hasNext(); ) {
		Triplet t1 = null;
		Triplet t2 = null;
		int i, s;
		for (i = 0, s = (rand.nextInt(candidates.size()) / 2)*2; i < candidates.size(); i += 2) {
			t1 = (Triplet) candidates.get((s+i) % candidates.size());
			t2 = (Triplet) candidates.get((s+i+1) % candidates.size());
			deleteTriplet(t1);
			deleteTriplet(t2);
			insertTriplet(new Triplet(t1.task(), t1.period(), t2.processor()));
			insertTriplet(new Triplet(t2.task(), t2.period(), t1.processor()));
			int updatedViols = violations();
			int updatedCost = cost();
			if (updatedViols < curr_viols ||
				(updatedViols == curr_viols && updatedCost < curr_cost) ) {
				changed = true;
				curr_viols = updatedViols;
				curr_cost = updatedCost;
				break;
			}
			deleteTriplet(new Triplet(t1.task(), t1.period(), t2.processor()));
			deleteTriplet(new Triplet(t2.task(), t2.period(), t1.processor()));
			insertTriplet(t1);
			insertTriplet(t2);
		}
		if (changed) {
			//	deleteTablet(oldTablet);
			//	insertTablet(newTablet);
			System.err.println("Swaping between "+t1+" and "+ t2);
		} 
		return changed;
	}
	
	
	public boolean delete_task() throws Exception{
		boolean changed = false;
		Random rand = new Random(Calendar.getInstance().getTimeInMillis());
		int curr_viols = violations();
		int curr_cost = cost();
		
		ArrayList candidates = new ArrayList();
		for (Iterator i1 = timetable.triplets().iterator() ; i1.hasNext(); ) {
			Triplet t1 = (Triplet) i1.next();
			candidates.add(t1);
		}
		
		if ( candidates.isEmpty() ) 
			return false; // changed == false		
		//for (Iterator i = candidates.iterator(); i.hasNext(); ) {
		Triplet t1 = null;
		int i, s;
		for (i = 0, s = rand.nextInt(candidates.size()); i < candidates.size(); i++) {
			t1 = (Triplet) candidates.get((s+i) % candidates.size());
			deleteTriplet(t1);
			int updatedViols = violations();
			int updatedCost = cost();
			if (updatedViols < curr_viols ||
				(updatedViols == curr_viols && updatedCost < curr_cost) ) {
				changed = true;
				curr_viols = updatedViols;
				curr_cost = updatedCost;
				break;
			}
			insertTriplet(t1);
		}
		if (changed) {
			//	deleteTablet(oldTablet);
			//	insertTablet(newTablet);
			System.err.println("Removed "+t1);
		} 
		return changed;
	}
	
	
	//解の評価値を要望別に数えて合計を返す
	public int cost() throws Exception {
		return costBySQL();
	}
	
	
	//解の評価値を要望別に数えて合計を返す(timetablesqlに対して行う)
	public int costBySQL() throws Exception {
		int penalties = 0;
		
		for (Iterator i = rules.iterator(); i.hasNext(); ){
			SQLRule r = (SQLRule) i.next();
			for (Iterator sql = r.queries().iterator(); sql.hasNext(); ) {
				if ( execute((String) sql.next()) ) {
					while (results().next()) {
						penalties += results().getInt("penalties") * r.weight();
					}
				}
			}
		}
		return penalties;
	}
	

	//解の制約違反を種類別に数えて合計を返す
	public int violations(){
		return timetable.countViolations();
	}
	
	
	
	private void insertTriplet(Triplet t) throws Exception {
		insertTriple(t.task().id(), t.period().id(), t.processor().id());
	}
	
	//(task_id,period_id,processor_id)を時間割timetablesqlに追加
	private void insertTriple(int task_id,int period_id,int processor_id) throws Exception {
		//追加
		timetable.insertTriple(task_id, period_id, processor_id);
		
		Task l = timetable.getTask(new Task(task_id));		
		for (int c = 0; c < Math.max(l.length(), 1); c++) {
			insertTripleIntoDBTable(task_id, period_id, processor_id);
			period_id = timetable.getNextPeriod(period_id);
		}
	}
	
	private void insertTripleIntoDBTable(int task_id,int period_id,int processor_id) throws Exception {
		//追加
		String sql="insert into timetablesql (task_id,period_id,processor_id) values ('" + task_id + "','" +period_id + "','" + processor_id+"')";
		executeUpdate(sql);
	}
	
	private void deleteTriplet(Triplet t) throws Exception {
		deleteTriple(t.task().id(), t.period().id(), t.processor().id());
	}
	
	private void deleteTriple(int task_id,int period_id, int processor_id) throws Exception {
		timetable.deleteTriple(task_id, period_id, processor_id);

		Task l = timetable.getTask(task_id);
		for (int c = 0; c < Math.max(l.length(), 1); c++) {
			deleteTabletFromDBTable(task_id, period_id, processor_id);
			period_id = timetable.getNextPeriod(period_id);
		}
	}
	
	//(task_id,period_id,processor_id)をtimetablesqlから削除
	private void deleteTabletFromDBTable(int task_id,int period_id, int processor_id) throws Exception{
		//削除
		String sql = "delete from timetablesql where processor_id ='" + processor_id+"' and period_id = '" + period_id+"' and task_id = '"+task_id +"'";
		executeUpdate(sql);
	}
	
	
	public String toStoreString() throws Exception{
		StringWriter sw = new StringWriter();
		PrintWriter w = new PrintWriter(sw);
		
		w.println(toString());
		return sw.toString();
	}
	
	
	public String toString(){
		StringBuffer sb = new StringBuffer();
		
		sb.append("TimeTableScheduling(");
		sb.append(timetable.processors());
		sb.append(", ");
		sb.append(timetable.tasks());
		sb.append(", ");
		sb.append(timetable.periods());
		sb.append(", ");
		sb.append(rules);
		sb.append(", ");
		sb.append(timetable);
		sb.append(")");
		return sb.toString();
	}
	
	
	private void openConnection(String url, String user, String pass) throws Exception {
		if ( db != null && (!db.isClosed()) ) // これ以外は何かのエラー
			return;
		try {
			Class.forName("com.mysql.jdbc.Driver"); 
			db = DriverManager.getConnection(url, user, pass);
		} catch (SQLException e) {
			System.err.println("Couldn't get connection: " + e);
			throw e;
		}
		state = db.createStatement();
		
		issuedQueries = 0;
		return;
	}

	/**
	 * すべてのコネクションを開放
	 */
	public synchronized void closeConnection() throws SQLException {
		if (state!=null) {
			state.close();
		}
		if (prestate!=null) {
			prestate.close();
		}
		// The ResultSet instances associated w/ the statements will be closed by the above.
		if (db!=null) {
			db.close();
		}
		System.err.println("Queries: "+issuedQueries);
	}
	
	public ResultSet results() {
		return results;
	}
	
	public void closeResults() throws SQLException {
		if (results != null)
			results.close();
		return;
	}

	public boolean execute(String sql) throws SQLException  {
		boolean qtype;
		try {
			qtype = state.execute(sql);
		} catch (SQLException e) {
			System.err.println("SQL: "+sql);
			throw e;
		}
		if (qtype) {
			results = state.getResultSet();
		}
		issuedQueries++;
		return qtype;
	}
	
	
	public ResultSet executeQuery(String sql) throws SQLException  {
		try {
			results = state.executeQuery(sql);
		} catch (SQLException e) {
			System.err.println("SQL: "+sql);
			throw e;
		}
		issuedQueries++;
		return results;
	}
	
	
	public int executeUpdate(String sql) throws SQLException  {
		try {
			updates = state.executeUpdate(sql);
		} catch (SQLException e) {
			System.err.println("SQL: "+sql);
			throw e;
		}
		issuedQueries++;
		return updates;
	}
	
	
	public void setPreparedStatement(String sql) throws SQLException  {
		prestate = db.prepareStatement(sql);
	}
	
	public void setStatementValue(int pos, int val) throws SQLException {
		prestate.setInt(pos, val);
	}
	
	public void setStatementValue(int pos, String val) throws SQLException {
		prestate.setString(pos, val);
	}
	
	public void setStatementValue(int pos, Timestamp val) throws SQLException {
		prestate.setTimestamp(pos, val);
	}
	
	public void setStatementValue(int pos, InputStream stream, int size) throws SQLException {
		prestate.setBinaryStream(pos, stream, size ); //-(3)
}	
	
	public void setStatementValue(int pos, byte b[]) throws SQLException {
		prestate.setBytes(pos, b);
	}

	//
	public int executePreparedUpdate() throws Exception  {
		updates = prestate.executeUpdate();
		
		issuedQueries++;
		
		return updates;
	}
	
	public ResultSet executePreparedQuery() throws Exception  {
		results = prestate.executeQuery();
		
		issuedQueries++;
		
		return results;
	}
	
}
