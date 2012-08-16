package common;
import javax.swing.*;
import java.awt.*;
import java.util.*;

public class GraphPanel5 extends JPanel {
  GraphCanvas5       can;
  JTextField title1 = new JTextField();
  JTextField title2 = new JTextField();
  JTextField title3 = new JTextField();

  public GraphPanel5() {
    setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
    add(title1);
    add(title2);
    add(title3);
    can = new GraphCanvas5();
    can.setSize(510, 160);
    can.setBackground(new Color(245, 245, 220));
    add(can);
  }

  public void setGraph(int dx, String data, int gains,
		       String text1, String text2, String title) {
    int max = 0;
    int sum = 0;
    int num = (int) (100 / dx) + 1;
    int i, tot;
    int[] m = new int[num];
    for (int k = 0; k < num; k++) {
      m[k] = 0;
    }

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
    int ind = (int) (gains / dx);
    can.setData(m, max, ind, dx);
    title1.setText(text1);
    title2.setText(text2);
    title3.setText(title + " " + sum);
  }

}
