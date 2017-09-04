import java.applet.Applet;
import java.awt.*;
import java.awt.event.*;

/*
  <applet code="MouseDemoApplet" width=200 height=200>
  </applet>
 */

public class MouseDemoApplet extends Applet 
  implements MouseListener {

  public void init() {
     addMouseListener(this);
  }

  public void mouseClicked(MouseEvent e) {
     System.out.println("Clicked.");
     setBackground(Color.blue);
     repaint();
  }

  public void mouseEntered(MouseEvent e) {
     System.out.println("Mouse enterd.");
     setBackground(Color.green);
     repaint();
  }

  public void mouseExited(MouseEvent e) {
     System.out.println("Mouse exited.");
     setBackground(Color.red);
     repaint();
  }

  public void mousePressed(MouseEvent e) {
     System.out.println("Button pressed.");
     setBackground(Color.white);
     repaint();
  }

  public void mouseReleased(MouseEvent e) {
     System.out.println("Button released.");
     setBackground(Color.yellow);
     repaint();
  }
}
