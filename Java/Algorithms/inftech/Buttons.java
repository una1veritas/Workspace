import java.applet.*;
import java.awt.*;
import java.awt.event.*;
/*
  <applet code="Buttons" width=400 height=80>
  </applet>
 */

public class Buttons extends Applet {
   Label label;

   public void init() {
      Button bttn1 = new Button("Apple");
      add(bttn1);
      Button bttn2 = new Button("Banana");
      add(bttn2);
      Button bttn3 = new Button("Orange");
      add(bttn3);

      label = new Label("No fruits");
      add(label);
   }
}
