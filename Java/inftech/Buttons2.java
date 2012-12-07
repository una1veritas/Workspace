import java.applet.*;
import java.awt.*;
import java.awt.event.*;
/*
  <applet code="Buttons" width=400 height=80>
  </applet>
 */

public class Buttons extends Applet implements ActionListener {
   Label label;

   public void init() {
      Button bttn1 = new Button("Apple");
      add(bttn1);
      Button bttn2 = new Button("Banana");
      add(bttn2);
      Button bttn3 = new Button("Orange");
      add(bttn3);

      bttn1.addActionListener(this);
      bttn2.addActionListener(this);
      bttn3.addActionListener(this);

      label = new Label("No fruits");
      add(label);
   }

    public void actionPerformed(ActionEvent event) {
	label.setText(event.getActionCommand());
    }
}
