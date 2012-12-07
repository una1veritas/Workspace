import java.applet.Applet;
import java.awt.*;
import java.awt.event.*;

/*
  <applet code="Buttons2" width=200 height=200>
  </applet>
  */

public class Buttons2 extends Applet
  implements ActionListener {

  Label label;

  public void init() {
    Button bttn1 = new Button("Apple");
    bttn1.addActionListener(this);
    add(bttn1);
    Button bttn2 = new Button("Banana");
    bttn2.addActionListener(this);
    add(bttn2);
    Button bttn3 = new Button("Orange");
    bttn3.addActionListener(this);
    add(bttn3);

    label = new Label("No fruits");
    add(label);
  }

  public void actionPerformed(ActionEvent event) {
    label.setText(event.getActionCommand());
  }
}
