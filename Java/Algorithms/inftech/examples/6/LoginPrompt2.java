import java.applet.*;
import java.awt.*;
import java.awt.event.*;

/*
  <applet code="LoginPrompt2" width=250 height=70>
  </applet>
  */

public class LoginPrompt2 extends Applet {
  TextField nfield, pfield;
  Label nlabel, plabel;

  public void init() {
    setLayout(null);
    
    nlabel = new Label("Name: ");
    nlabel.setBounds(10, 10, 60, 20);
    add(nlabel);
    
    nfield = new TextField(20);
    nfield.setText("guest");
    nfield.setBounds(80, 10, 150, 20);
    add(nfield);
    
    plabel = new Label("Password: ");
    plabel.setBounds(10, 40, 60, 20);
    add(plabel);
    
    pfield = new TextField(12);
    pfield.setEchoChar('*');
    pfield.setBounds(80, 40, 150, 20);
    add(pfield);
  }
}
