public class Square extends Polygon {
  public Square(int left, int top, int length) {
    xpoints = new int[4];
    ypoints = new int[4];
    xpoints[0] = left;
    ypoints[0] = top;
    xpoints[1] = left + length;
    ypoints[1] = top;
    xpoints[2] = left + length;
    ypoints[2] = top + length;
    xpoints[3] = left;
    ypoints[3] = top + length;
  }
}
