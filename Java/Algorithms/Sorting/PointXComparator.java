import java.awt.Point;
import java.util.Comparator;

class PointXComparator implements Comparator {
    
    public int compare(Object o1, Object o2) {
	Point p1, p2;
	p1 = (Point) o1;
	p2 = (Point) o2;

	if (p1.x == p2.x) {
	    return (p1.y - p2.y);
	} else {
	    return (p1.x - p2.x);
	}
    }

}

