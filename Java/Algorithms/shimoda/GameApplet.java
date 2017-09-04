import java.applet.Applet;
import java.awt.Graphics;
import java.awt.Color;
import java.awt.event.*;

/*
<applet code="GameApplet" width=400 height=400>
</applet>
*/

public class GameApplet extends Applet
    implements KeyListener{
    Soukoban map;
    int width = 32;
    
    public void init(){
	addKeyListener(this);
	map = new Soukoban();
    }
    
    public void paint(Graphics g){
	int x, y;
	
	for(y=0;y<10;y++){
	    for(x=0;x<10;x++){
		switch(map.getElm(x,y)){
		case 'B':
		    g.setColor(new Color(128,61,0));
		    g.fillRect(x*width,y*width,width,width);
		    break;
		case 'M':
		    g.setColor(new Color(0,0,0));
		    g.fillRect(x*width,y*width,width,width);
		    break;
		case 'W':
		    g.setColor(new Color(127,127,127));
		    g.fillRect(x*width,y*width,width,width);
		    break;
		}
	    }
	}
	//System.out.println("Owari.");
    }
    
    public void keyReleased(KeyEvent e){
	//System.out.println();
    }
    
    public void keyPressed(KeyEvent e){
	switch(e.getKeyCode()){
	case KeyEvent.VK_UP:
	    map.moveUp();
	    repaint();
	    break;
	case KeyEvent.VK_DOWN:
	    map.moveDown();
	    repaint();
	    break;
	case KeyEvent.VK_RIGHT:
	    map.moveRight();
	    repaint();
	    break;
	case KeyEvent.VK_LEFT:
	    map.moveLeft();
	    repaint();
	    break;
	}
	//System.out.print(".");
    }
    
    public void keyTyped(KeyEvent e){
    }
}
