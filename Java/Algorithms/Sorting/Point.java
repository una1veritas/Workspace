import java.awat.*;

class PointXComparator {
    public PointXComparator(Object o1, Object o2) {
	Point p1, p2;
	p1 = (Point) o1;
	p2 = (Point) o2;

	if (o1.x == o2.x) {
	    return (o1.y - o2.y);
	} else {
	    return (o1.x - o2.x);
	}
    }
}
