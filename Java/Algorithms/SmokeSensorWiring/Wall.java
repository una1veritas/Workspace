// Wall.java

import java.util.*;

class Wall {
  int sx, sy, gx, gy;

  Wall() {
    sx = 0;
    sy = 0;
    gx = 0;
    gy = 0;
		//{{INIT_CONTROLS
		//}}
}

  Wall(int i) {
    int random = (int)(Math.random()*700)+65;  
    if (i == 0) {
      sx = random;
      gx = random;
      sy = (int)(Math.random()*700)+65;
      gy = (int)(Math.random()*700)+65;
    } else if (i == 1) {
      sy = random;
      gy = random;
      sx = (int)(Math.random()*700)+65;
      gx = (int)(Math.random()*700)+65;
    }
  }

  Wall(int a, int b, int c, int d) {
    sx = a;
    sy = b;
    gx = c;
    gy = d;
  }
	//{{DECLARE_CONTROLS
	//}}
}
