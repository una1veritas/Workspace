import java.applet.Applet;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.Color;
import java.awt.event.*;

/*
<applet code="GameApplet2" width=400 height=400>
</applet>
*/

public class GameApplet2 extends GameApplet {
    Image image;
    int iwidth, iheight;

    public void init(){
	super.init();
	iwidth = getWidth();
	iheight = getHeight();
	image = createImage(iwidth, iheight);
    }
    
    public void paint(Graphics g) {
	Graphics offscreen;

	offscreen = image.getGraphics();
	offscreen.setColor(getBackground());
	offscreen.fillRect(0,0,iwidth,iheight);
	
	super.paint(offscreen);

	g.drawImage(image, 0, 0, this);
    }

    public void update(Graphics g) {
	paint(g);
    }
}
