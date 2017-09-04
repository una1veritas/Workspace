import java.applet.*;
import java.awt.*;
import java.awt.event.*;

/*
  <applet code="GameApplet" width=320 height=320>
  </applet>
*/

public class GameApplet extends Applet 
    implements KeyListener, MouseListener, MouseMotionListener, Runnable {
    
    int x, y;
    int xorg, yorg;
    boolean pressed = false;

    Thread frames;
    int posx = 0;

    public void init() {
	addKeyListener(this);
	addMouseListener(this);
	addMouseMotionListener(this);
    }

    public void start() {
	if ((frames = new Thread(this)) != null) {
	    frames.start();
	}
    }

    public void stop() {
	System.out.println("Stopped.");
	if (frames != null) {
	    /* frames.stop(); */
	    frames = null;
	}
    }

    public void destroy() {
	System.out.println("Destroyed.");
    }

    public void run() {
	while (true) {
	    repaint();
	    /* update screen */
	    posx = (posx + 1) % 100;
	    try {
		Thread.sleep(100);
	    } catch (Exception x) {
	    }
	}
    }

    public void paint(Graphics g) {
	System.out.print(".");
	if (pressed) {
	    g.drawLine(xorg, yorg,x,y);
	}
	g.fillRect(posx+20,20,80,100);
    }

    public void keyPressed(KeyEvent e) {
    }

    public void keyReleased(KeyEvent e) {
	System.out.println("Key " + e.getKeyText(e.getKeyCode()));
    }
    
    public void keyTyped(KeyEvent e) {
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
	/* repaint(); */
    }
    
    public void mouseReleased(MouseEvent e) {
	System.out.println("Button released.");
	pressed = false;
	/* repaint(); */
    }
    
    public void mouseDragged(MouseEvent e) {
	x = e.getX();
	y = e.getY();
	/* repaint(); */
    }
    
    public void mouseMoved(MouseEvent e) {
    }

}

