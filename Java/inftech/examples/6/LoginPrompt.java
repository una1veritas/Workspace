import java.applet.*;
import java.awt.*;
import java.awt.event.*;

/*
  <applet code="LoginPrompt" width=250 height=80>
  </applet>
  */

public class LoginPrompt extends Applet {
  TextField nfield, pfield;
  Label name, password;

  public void init() {
    name = new Label("Name: ");
    add(name);
    nfield = new TextField(20);
    nfield.setText("guest");
    add(nfield);
    password = new Label("Password: ");
    add(password);
    pfield = new TextField(12);
    pfield.setEchoChar('*');
    add(pfield);
  }
}
