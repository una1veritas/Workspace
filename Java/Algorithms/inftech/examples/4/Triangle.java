public class Triangle extends Polygon {
  public Triangle(int left, int top, int length, int height) {
    xpoints = new int[3];
    ypoints = new int[3];
    xpoints[0] = left;
    ypoints[0] = top + height;
    xpoints[1] = left + length;
    ypoints[1] = top + height;
    xpoints[2] = left + (length / 2);
    ypoints[2] = top;
  }
}
