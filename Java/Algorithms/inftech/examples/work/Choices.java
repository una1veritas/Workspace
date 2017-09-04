import java.applet.*;
import java.awt.*;
import java.awt.event.*;

/*
  <applet code="Choices" width=400 height=80>
  </applet>
  */

public class Choices extends Applet
  implements ItemListener {
  Label label;
  Choice choice1, choice2;

  public void init() {
    choice1 = new Choice();
    choice1.addItem("Kitakyushu");
    choice1.addItem("Fukuoka");
    choice1.addItem("Iizuka");
    choice1.addItem("Yukuhashi");
    choice1.addItemListener(this);
    add(choice1);

    choice2 = new Choice();
    choice2.addItem("Nishitetsu Bus");
    choice2.addItem("JR");
    choice2.addItem("TAXI");
    choice2.addItemListener(this);
    add(choice2);

    label = new Label("Kitakyushu, Nishitetsu Bus");
    add(label);
  }

  public void itemStateChanged(ItemEvent event) {
/*    Choice choice = (Choice) event.getItemSelectable();
    if (choice1.getState()) {
    } else if (choice2.getState()) {
    }*/
    label.setText(choice1.getSelectedItem() + ", " + choice2.getSelectedItem());
  }
}
