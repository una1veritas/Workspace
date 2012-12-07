import java.applet.*;
import java.awt.*;
import java.awt.event.*;

/*
  <applet code="MyApplet" width=240 height=200>
  </applet>
*/

public class MyApplet extends Applet 
    implements ActionListener {
    
    public void init() {
	Button b = new Button("Open Frame");
	b.addActionListener(this);
	add(b);
    }

    public void actionPerformed(ActionEvent e) {
	MainFrame f = new MainFrame("New Frame");
	f.setSize(400,400);
	f.show();
    }
}

class MainFrame extends Frame {
    MainFrame(String title) {
	super(title);
	addWindowListener(new MyWindowAdapter());
	addKeyListener(new MyKeyAdapter());
    }

    class MyWindowAdapter extends WindowAdapter {
	public void windowClosing(WindowEvent e) {
	    dispose();
	}
    }

    class MyKeyAdapter extends KeyAdapter {
	public void keyReleased(KeyEvent e) {
	    System.out.println("Key " + e.getKeyChar());
	}
    }
    
}
