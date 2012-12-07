public class RegularTriangle extends Polygon {
  public RegularTriangle(int x, int y, int size) {
    xpoints = new int[3];
    ypoints = new int[3];
    xpoints[0] = x;
    ypoints[0] = y;
    xpoints[1] = x + size;
    ypoints[1] = y;
    xpoints[2] = x + (size / 2);
    ypoints[2] = y - (int)(size * (Math.sqrt(3.0) / 2));
  }
}
