// Dot.java

import java.awt.*;

class Dot {
  int x, y, num, id;

  Dot() {
    x = 0;
    y = 0;
    num = 0;
    id = 0;
  		//{{INIT_CONTROLS
		//}}
}

  Dot(int px, int py, int pn, int pid) {
    x = px;
    y = py;
    num = pn;
    id = pid;
  }

  Dot(int pn) {
    x = (int)(Math.random()*70)*10 + 65;
    y = (int)(Math.random()*70)*10 + 65;
    num = pn;
    id = 0;
  }
	//{{DECLARE_CONTROLS
	//}}
}
