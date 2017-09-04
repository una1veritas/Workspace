import java.applet.Applet;
import java.awt.*;
import java.awt.event.*;

/*
  <applet code="Checkboxes" width=400 height=80>
  </applet>
  */

public class Checkboxes extends Applet {
  Label label;

  public void init() {
    Checkbox chkbox1 = new Checkbox("Apple");
    add(chkbox1);
    Checkbox chkbox2 = new Checkbox("Grapefruit");
    add(chkbox2);
    Checkbox chkbox3 = new Checkbox("Orange");
    add(chkbox3);

    label = new Label("No fruits.");
    add(label);
  }
}
