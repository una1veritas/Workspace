import java.applet.Applet;
import java.awt.Graphics;
/*
  <applet code="MyApplet" width=200 height=200>
  </applet>
  */
public class MyApplet extends Applet {
  public void paint(Graphics g) {
    g.drawString("This is my applet.", 20, 100);
  }
}
