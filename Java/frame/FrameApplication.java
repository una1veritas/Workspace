import java.awt.*;
import java.awt.event.*;

class MyFrame extends Frame {
    class MyKeyAdapter extends KeyAdapter {
	public void KeyReleased(KeyEvent e) {
	    System.out.println("Key " + e.getKeyChar());
	}
    }

    class MyWindowListener extends WindowAdapter {
	public void windowClosing(WindowEvent e) {
	    System.exit(0);
	}
    }

    MyFrame(String title) {
	super(title);
	addWindowListener(new MyWindowListener());
    }

    public void paint(Graphics g) {
	g.setColor(Color.blue);
	g.fillRect(10,25,230,128);
    }
}

public class FrameApplication {
    public static void main(String args[]) {
	MyFrame fr = new MyFrame("Mine!");
	fr.setSize(400,400);
	fr.show();
    }
}
