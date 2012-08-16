package common;
import java.awt.*;
//import java.util.*;

public class GraphCanvas5 extends Canvas {
  int[] y;
  int ind;
  int Max;
  int number;
  int dx;
  int sx;

  public GraphCanvas5() { }

  public void setData(int[] m, int max, int index, int d) { 
    dx = d;
    y = new int[m.length];
    Max = max + 10 - (max % 10);
    number = (int) (100 / dx) + 1;
    for (int i = 0; i < number; i++) {
      y[i] = 120 * m[i] / Max;
    }
    ind = index;
    dx = (int) (450 * dx / 100);  
    sx = 20 - dx / 2;            
    repaint();
  }

  public void paint(Graphics g) {
    g.setColor(Color.black);
    g.drawLine(470, 145, 470, 10);
    g.drawLine(20,  145, 20,  10);
    g.drawLine(20,  140, 470, 140);
    for (int i = 0; i < number; i++) {
      if (i == 0) {
	g.setColor(Color.pink);
      } else if (i == 1) {
	g.setColor(Color.yellow);
      } else {
	g.setColor(Color.white);
      }
      if (i == ind) {
	g.setColor(Color.green);
      } 
      g.fillRect(sx+dx*i, 140-y[i], dx, y[i]);
      g.setColor(Color.black);
      g.drawRect(sx+dx*i, 140-y[i], dx, y[i]);
    }
    g.drawLine(465, 20, 473, 20);
    g.drawString("" + Max, 475, 25);
    g.drawString("100", 462, 155);
    g.drawString("0", 18, 155);

    g.drawLine(290, 140, 290, 145);
    g.drawString("60", 285, 155);

    g.drawLine(380, 140, 380, 145); 
    g.drawString("80", 375, 155);
  }
}
