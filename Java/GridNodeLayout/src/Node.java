import HideYoshi.Point;


public class Node {
	public int id;
	public int x, y;
	
	public Node(int id) {
		this.id = id;
	}
	
	public Node(int id, int px, int py) {
		this.id = id;
		x = px;
		y = py;
	}
	
	public Node(Node v) {
		id = v.id;
		x = v.x;
		y = v.y;
	}
	
	public void replaceWith(Node a){
		x = a.x;
		y = a.y;
		id = a.id;
	}
}
