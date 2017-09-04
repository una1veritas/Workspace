import java.applet.Applet;
import java.awt.*;
import java.awt.event.*;

/*
  <applet code="MouseDragApplet2" width=200 height=200>
  </applet>
  */

public class MouseDragApplet2 extends Applet
  implements MouseListener, MouseMotionListener {

  int xs, ys, xe, ye;
  boolean pressed = false;

  public void init() {
    addMouseListener(this);
    addMouseMotionListener(this);
  }

  public void mouseClicked(MouseEvent e) {
  }

  public void mouseEntered(MouseEvent e) {
  }

  public void mouseExited(MouseEvent e) {
  }

  public void mousePressed(MouseEvent e) {
    System.out.println("Button pressed.");
    xs = xe = e.getX();
    ys = ye = e.getY();
    pressed = true;
    repaint();
  }

  public void mouseReleased(MouseEvent e) {
    System.out.println("Button released.");
    pressed = false;
    repaint();
  }

  public void mouseDragged(MouseEvent e) {
    xe = e.getX();
    ye = e.getY();
    repaint();
  }

  public void mouseMoved(MouseEvent e) {
  }

  public void paint(Graphics g) {
    if (pressed) {
      g.drawLine(xs, ys, xe, ye);
    }
  }
}
