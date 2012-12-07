import java.applet.Applet;
import java.awt.Graphics;

/*
  <applet code="MyApplet" width=200 height=200>
  </applet>
 */

public class MyApplet extends Applet {
    public void paint(Graphics g) {
	int x[] = {100, 150, 150, 125, 100};
	int y[] = {100, 100, 150, 180, 150};

	g.drawString("This is my applet.", 20, 100);
	Polygon p = new Polygon(x, y);
	p.drawOn(g);
	p.move(13, -5);
	p.drawOn(g);
	Square s = new Square(30, 40, 50);
	s.drawOn(g);
	s.say();

	Messenger msgr = new Messenger("Sin", "Father");
	g.drawString(msgr.sayHello(), 130, 20);
    }
}

