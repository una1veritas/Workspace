package common;
import java.awt.*;
//import java.util.*;

public class GraphCanvas6 extends Canvas {
  int[] y = new int [41];
  int ind;
  int Max;
  int number;
  int dx;
  int sx;

  public GraphCanvas6() { }

  public void setData(int[] m, int max, int index, int d) { 
    dx = d;
    Max = max + 10 - (max % 10);
    number = (int) (40 / dx) + 1;
    for (int i = 0; i < number; i++) {
      y[i] = 120 * m[i] / Max;
    }
    ind = index;
    dx = (int) (180 * dx / 40);
    sx = 20 - dx / 2;
    repaint();
  }

  public void paint(Graphics g) {
    for (int i = 0; i < number; i++) {
      if (i != ind) {
	g.setColor(Color.white);
      } else {
	g.setColor(Color.green);
      }
      g.fillRect(sx+dx*i, 140-y[i], dx, y[i]);
      g.setColor(Color.black);
      g.drawRect(sx+dx*i, 140-y[i], dx, y[i]);
    }

    g.drawLine(200, 145, 200, 10);
    g.drawLine(195, 20, 203, 20);
    g.drawString("" + Max, 205, 25);
    g.drawString("4.0", 192, 155);

    g.drawLine(20, 140, 20, 145);
    g.drawString("0.0", 15, 155);

    g.drawLine(110, 140, 110, 145); 
    g.drawString("2.0", 105, 155);
  }
}
