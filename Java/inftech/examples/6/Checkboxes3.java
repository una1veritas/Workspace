import java.applet.Applet;
import java.awt.*;
import java.awt.event.*;

/*
  <applet code="Checkboxes3" width=400 height=80>
  </applet>
  */

public class Checkboxes3 extends Applet
  implements ItemListener {

  Label label;
  Checkbox chkbox1, chkbox2, chkbox3;

  public void init() {
    chkbox1 = new Checkbox("Apple");
    chkbox1.addItemListener(this);
    add(chkbox1);
    chkbox2 = new Checkbox("Grapefruit");
    chkbox2.addItemListener(this);
    add(chkbox2);
    chkbox3 = new Checkbox("Orange");
    chkbox3.addItemListener(this);
    add(chkbox3);

    label = new Label("No fruits.");
    add(label);
  }

  public void itemStateChanged(ItemEvent event) {
    String s = "";

    if (chkbox1.getState()) {
      s += chkbox1.getLabel();
    }
    if (chkbox2.getState()) {
      s += chkbox2.getLabel();
    }
    if (chkbox3.getState()) {
      s += chkbox3.getLabel();
    }
    label.setText(s);
  }
}
