public class Polygon {
    int xpoints[];
    int ypoints[];

    Polygon() {
    }

    public Polygon(int[] x, int[] y) {
	xpoints = new int[x.length];
	for (int i = 0; i < x.length; i++)
	    xpoints[i] = x[i];
	ypoints = new int[y.length];
	for (int i = 0; i < y.length; i++)
	    ypoints[i] = y[i];
    }

    private int size() {
	return xpoints.length;
    }

    public void drawOn(java.awt.Graphics g) {
	g.drawPolygon(xpoints, ypoints, this.size());
    }

    public void move(int x, int y) {
	for(int i = 0; i < size(); i++) {
	    xpoints[i] += x;
	    ypoints[i] += y;
	}
    }
}

