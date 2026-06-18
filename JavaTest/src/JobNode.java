
public class JobNode {
	private PrintJob job;
	public JobNode left;
	public JobNode right;
	
	public JobNode(PrintJob pj) {
		job = pj;
		left = null;
		right = null;
	}
}