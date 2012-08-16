package common;
import javax.swing.*;
import java.awt.*;
import java.util.*;

public class GraphPanel6 extends JPanel {
  GraphCanvas6       can;
  JTextField    title0 = new JTextField();
  JTextField    title1 = new JTextField();
  JTextField    title2 = new JTextField();

  public GraphPanel6() {
    setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
    add(title0);
    add(title1);
    add(title2);
    can = new GraphCanvas6();
    can.setSize(250, 180);
    can.setBackground(new Color(245, 245, 220));
    add(can);
  }

  public void setGraph(int dx, String data, int gpa_ind,
		       String text0, String text1, String text2) {
    int max = 0;
    int sum = 0;
    int num = (int) (40 / dx) + 1;
    int i, tot;
    int[] m = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

    StringTokenizer stk = new StringTokenizer(data, "$");
    while (stk.hasMoreTokens()) {
      String ss = stk.nextToken();
      StringTokenizer sstk = new StringTokenizer(ss, "|");
      i = Integer.parseInt(sstk.nextToken());
      tot = Integer.parseInt(sstk.nextToken());
      if (i < num) {
	if (tot > max) {
	  max = tot;
	}
	sum = sum + tot;
	m[i] = tot;
      }
    }
    can.setData(m, max, gpa_ind, dx);
    title0.setText(text0);
    title1.setText(text1);
    title2.setText(text2 + " " + sum);
  }
}
