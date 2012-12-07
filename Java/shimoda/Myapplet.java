import java.applet.Applet;
import java.awt.Graphics;
import java.awt.Color;
import java.awt.event.*;

/*
<applet code="MyApplet" width=200 height=200>
</applet>
*/

public class MyApplet extends Applet{
	Soukoban map;
	int width = 20;

	public void init(){
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
	}
}