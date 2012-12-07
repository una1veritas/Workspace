import java.applet.*;
import java.awt.*;
import java.awt.event.*;

/*
  <applet code="Choices" width=400 height=80>
  </applet>
  */

public class Choices extends Applet {
  Label label;

  public void init() {
    Choice choice1 = new Choice();
    choice1.addItem("Kitakyushu");
    choice1.addItem("Fukuoka");
    choice1.addItem("Iizuka");
    choice1.addItem("Yukuhashi");
    add(choice1);
    Choice choice2 = new Choice();
    choice2.addItem("JR");
    choice2.addItem("Nishitetsu Bus");
    choice2.addItem("TAXI");
    add(choice2);

    label = new Label("Kitakyushu, Nishitetsu Bus");
    add(label);
  }
}
