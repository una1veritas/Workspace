public class PrintJob {
	private String jobId;
	private String user;
	private int priority;

	private static int created = 0;
	
	public PrintJob(String id, String uname, int prio) {
		jobId = id;
		user = uname;
		priority = prio;
		
		created++;
	}
}
