import java.applet.Applet;
import java.awt.*;
import java.awt.event.*;

/*
  <applet code="MouseDragApplet" width=200 height=200>
  </applet>
 */

public class MouseDragApplet extends Applet 
  implements MouseListener, MouseMotionListener, KeyListener {

  int x, y;
    int xorg, yorg;
  boolean pressed = false;

  public void init() {
     addMouseListener(this);
     addMouseMotionListener(this);
     Label b = new Label("push me!");
     b.addKeyListener(this);
     add(b);
     b.requestFocus();
  }

    public void keyTyped(KeyEvent e) {
    }

    public void keyPressed(KeyEvent e) {
	System.out.println("!!");
    }
    
    public void keyReleased(KeyEvent e) {
	System.out.print("Key ");
	System.out.print(e.getKeyChar());
	System.out.print(" Pressed.\n" );
    }

  public void mouseClicked(MouseEvent e) {
  }

  public void mouseEntered(MouseEvent e) {
  }

  public void mouseExited(MouseEvent e) {
  }

  public void mousePressed(MouseEvent e) {
     System.out.println("Button pressed.");
     x = e.getX();
     y = e.getY();
     xorg = x;
     yorg = y;
     pressed = true;
     repaint();
  }

  public void mouseReleased(MouseEvent e) {
     System.out.println("Button released.");
     pressed = false;
     repaint();
  }

  public void mouseDragged(MouseEvent e) {
     x = e.getX();
     y = e.getY();
     repaint();
  }

  public void mouseMoved(MouseEvent e) {
  }

  public void paint(Graphics g) {
     if (pressed) {
        g.drawLine(xorg, yorg,x,y);
     }
  }
}
