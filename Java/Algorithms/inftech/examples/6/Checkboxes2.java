import java.applet.Applet;
import java.awt.*;
import java.awt.event.*;

/*
  <applet code="Checkboxes2" width=400 height=80>
  </applet>
  */

public class Checkboxes2 extends Applet
  implements ActionListener {

  Label label;
  Checkbox chkbox1, chkbox2, chkbox3;

  public void init() {
    chkbox1 = new Checkbox("Apple");
    add(chkbox1);
    chkbox2 = new Checkbox("Grapefruit");
    add(chkbox2);
    chkbox3 = new Checkbox("Orange");
    add(chkbox3);

    Button bttn = new Button("OK");
    bttn.addActionListener(this);
    add(bttn);

    label = new Label("No fruits.");
    add(label);
  }

  public void actionPerformed(ActionEvent event) {
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
