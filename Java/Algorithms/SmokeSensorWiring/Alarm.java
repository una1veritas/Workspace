// Alarm.java

import java.util.*;
import java.awt.*;

import Dot;
import Wall;

class Alarm {
  Vector point;
  Vector between;
  int init;
  boolean sort;
  int value[] = new int[8];
  Vector wall_h, wall_l;

  Alarm() {
    point = new Vector(0,1);
    between = new Vector(0,1);
    init = 0;
    sort = false;
    resetValue();
    wall_h = new Vector(0,1);
    wall_l = new Vector(0,1);
//    wall_h.addElement(new Wall( 60, 60,770, 60));
//    wall_h.addElement(new Wall( 60, 60, 60,770));
//    wall_h.addElement(new Wall(770, 60,770,770));
//    wall_h.addElement(new Wall( 60,770,770,770));
  }

  Alarm(Alarm a) {
    point = (Vector)a.point.clone();
    between = (Vector)a.between.clone();
    init = a.init;
    sort = a.sort;
    resetValue();
    wall_h = (Vector)a.wall_h.clone();
    wall_l = (Vector)a.wall_l.clone();
  }

  void resetValue() {
    for (int i = 0; i < 8; i++)
      value[i] = 0;
  }

  void resetPoint() {
    point.removeAllElements();
    between.removeAllElements();
    init = 0;
    sort = false;
    resetValue();
  }

  void randomPoint(int n) {
    resetPoint();
    int r = (int)(Math.random()*n) + 2;
    for (int i = 0; i < r; i++) {
      int s = point.size();
      point.addElement(new Dot(s));
      System.out.println("pt[" + getnum(s) + "] is (" + getx(s) + ", " + gety(s) + ")");
    }
  }
  
  void resetWall_l() {
    wall_l.removeAllElements();
    sort = false;
  }

  void resetWall_h() {
//    int n = wall_h.size();
//    for (int i = 0; i < n-4; i++)
//      wall_h.removeElementAt(4);
    wall_h.removeAllElements();
    sort = false;
  }

  void resetWall() {
    resetWall_l();
    resetWall_h();
  }

  void randomWall_l(int n) {
    resetWall_l();
    int q = (int)(Math.random()*n);
    for (int i = 0; i < q; i++) {
      int z = (int)(Math.random()*2);
      wall_l.addElement(new Wall(z));
    }
  }

  void randomWall_h(int n) {
    resetWall_h();
    int p = (int)(Math.random()*n);
    for (int i = 0; i< p; i++) {
      int z = (int)(Math.random()*2);     
      wall_h.addElement(new Wall(z));
    }
  }

  void randomWall(int m, int n) {
    randomWall_h(m);
    randomWall_l(n);
  }

  void start_x() {
    if (getx(0) != getx(1))
      init = 3;
    else
      init = 0;
  }

  void start_y() {
    if (gety(0) != gety(1))
      init = 0;
    else
      init = 3;
  }

  void sort_x() {
    start_x();
    normalsort();
    path();
  }

  void sort_y() {
    start_y();
    normalsort();
    path();
  }

  int getx(int i) { 
    return ((Dot)point.elementAt(i)).x;
  }
  int gety(int i) { 
    return ((Dot)point.elementAt(i)).y;
  }
  int getnum(int i) { 
    return ((Dot)point.elementAt(i)).num;
  }
  int getid(int i) { 
    return ((Dot)point.elementAt(i)).id;
  }
  int getbx(int i) { 
    return ((Point)between.elementAt(i)).x;
  }
  int getby(int i) { 
    return ((Point)between.elementAt(i)).y;
  }
  int getsx(Vector vector, int i) {
    return ((Wall)vector.elementAt(i)).sx;
  }
  int getsy(Vector vector, int i) {
    return ((Wall)vector.elementAt(i)).sy;
  }
  int getgx(Vector vector, int i) {
    return ((Wall)vector.elementAt(i)).gx;
  }
  int getgy(Vector vector, int i) {
    return ((Wall)vector.elementAt(i)).gy;
  }

  boolean smaller(int i, int j, int k) {
    return (i < j) && (j < k);
  }
  boolean smaller2(int i, int j, int k) {
    return (i <= j) && (j < k);
  }

  void print() {  // みちじゅん と ひょうか を しゅつりょく
    int n = point.size();
    for (int i = 0; i < n-1; i++)
      System.out.print(getnum(i) + " -> ");
    System.out.println(getnum(n-1));
    for (int i = 0; i < 7; i++)
      System.out.print(value[i]+", ");
    System.out.println(value[7]);
  }

  void path() {
    resetValue();
    int n = point.size();
    between.removeAllElements();
    for (int i = 0; i < n-1; i++) {
      between.addElement(new Point(direction_x(i,i+1),direction_y(i,i+1)));
      value[4] += distance(i,i+1);
      value[6] += (distance(i,i+1) - duplication(i,i+1));
    }
    cross0();
    cross1();
    cross2();
    roomchange();
    cornering();
  }

  int ccw(Point pt0, Point pt1, Point pt2) {
    int dx1 = pt1.x - pt0.x, dy1 = pt1.y - pt0.y,
        dx2 = pt2.x - pt0.x, dy2 = pt2.y - pt0.y;

    if (dx1*dy2 > dy1*dx2)
      return 1;
    else if (dx1*dy2 < dy1*dx2)
      return -1;
    else
      if ((dx1*dx1 + dy1*dy1) > (dx2*dx2 + dy2*dy2))
	return 0;
      else
	return 100;
  }

/*  boolean intersect(Vector p, Vector q) {
    boolean tmp = ((ccw(p.
			
*/
/*  void cross05() { //Alarm vs. Wall_H
    int n = point.size();
    int m = wall_h.size();
    value[0] = 0;

    for (int i = 0; i < n-1; i++)
      for (int j = 0; j < m; j++) {
	
*/
  void cross0() { // Alarm vs. Wall_H
    int n = point.size();
    int m = wall_h.size();
    value[0] = 0;

    for (int i = 0; i < n-1; i++)
      for (int j = 0; j < m; j++) {

	int gsx = getsx(wall_h,j), ggx = getgx(wall_h,j),
	    gsy = getsy(wall_h,j), ggy = getgy(wall_h,j),
	    xi = getx(i), bxi = getbx(i), xii = getx(i+1),
	    yi = gety(i), byi = getby(i), yii = gety(i+1);

	if (gsx == ggx) { // tate
	  if (smaller(gsy,byi,ggy)||smaller(ggy,byi,gsy)) {
	    if (yi == byi) // yoko
	      if (smaller(xi,gsx,bxi)||smaller(bxi,gsx,xi))
		value[0]++;
	    if (byi == yii)  // yoko
	      if (smaller(bxi,gsx,xii)||smaller(xii,gsx,bxi))
		value[0]++;
	  }
	} else if (gsy == ggy) // yoko
	  if (smaller(gsx,bxi,ggx)||smaller(ggx,bxi,gsx)) {
	    if (xi == bxi) // tate
	      if  (smaller(yi,gsy,byi)||smaller(byi,gsy,yi))
		value[0]++;
	    if (bxi == xii) // tate
	      if (smaller(byi,gsy,yii)||smaller(yii,gsy,byi))
		value[0]++;
	  }
      }
  }  

  void cross1() { // Alarm vs. Alarm
    int n = point.size();
    value[1] = 0;

    for (int i = 0; i < n-2; i++) {
	int xi = getx(i), xbi = getbx(i), xii = getx(i+1),
	    yi = gety(i), ybi = getby(i), yii = gety(i+1),
	    xj = getx(i+1), xbj = getbx(i+1), xjj = getx(i+1+1),
	    yj = gety(i+1), ybj = getby(i+1), yjj = gety(i+1+1);

	if (xi == xbi) // tate
	  if (smaller(yi,ybj,ybi)||smaller(ybi,ybj,yi)) {
	    if (yj == ybj) // yoko
	      if (smaller(xj,xbi,xbj)||smaller(xbj,xbi,xj))
		value[1]++;
	    if (yjj == ybj) // yoko
	      if (smaller(xjj,xbi,xbj)||smaller(xbj,xbi,xjj))
		value[1]++;
	  }
	if (yi == ybi) // yoko
	  if (smaller(xi,xbj,xbi)||smaller(xbi,xbj,xi)) {
	    if (xj == xbj) // tate
	      if (smaller(yj,ybi,ybj)||smaller(ybj,ybi,yj))
		value[1]++;
	    if (xbj == xjj) // tate
	      if (smaller(ybj,ybi,yjj)||smaller(yjj,ybi,ybj))
		value[1]++;
	  }
	if (xbi == xii) // tate
	  if (smaller(ybi,ybj,yii)||smaller(yii,ybj,ybi)) {
	    if (yj == ybj) // yoko
	      if (smaller(xj,xbi,xbj)||smaller(xbj,xbi,xj))
		value[1]++;
	    if (yjj == ybj) // yoko
	      if (smaller(xbj,xbi,xjj)||smaller(xjj,xbi,xbj))
		value[1]++;
	  }
	if (ybi == yii) // yoko
	  if (smaller(xbi,xbj,xii)||smaller(xii,xbj,xbi)) {
	    if (xj == xbj) // tate
	      if (smaller(yj,ybi,ybj)||smaller(ybj,ybi,yj))
		value[1]++;
	    if (xbj == xjj) // tate
	      if (smaller(ybj,ybi,yjj)||smaller(yjj,ybi,ybj))
		value[1]++;
	  }
	for (int j = i+2; j < n-1; j++) {

	   xi = getx(i); xbi = getbx(i); xii = getx(i+1);
	      yi = gety(i); ybi = getby(i); yii = gety(i+1);
	      xj = getx(j); xbj = getbx(j); xjj = getx(j+1);
	      yj = gety(j); ybj = getby(j); yjj = gety(j+1);

	  if (xi == xbi) // tate
	    if (smaller2(yi,ybj,ybi)||smaller2(ybi,ybj,yi)) {
	      if (yj == ybj) // yoko
		if (smaller2(xj,xbi,xbj)||smaller2(xbj,xbi,xj))
		  value[1]++;
	      if (yjj == ybj) // yoko
		if (smaller2(xjj,xbi,xbj)||smaller2(xbj,xbi,xjj))
		  value[1]++;
	    }
	  if (yi == ybi) // yoko
	    if (smaller2(xi,xbj,xbi)||smaller2(xbi,xbj,xi)) {
	      if (xj == xbj) // tate
		if (smaller2(yj,ybi,ybj)||smaller2(ybj,ybi,yj))
		  value[1]++;
	      if (xbj == xjj) // tate
		if (smaller2(ybj,ybi,yjj)||smaller2(yjj,ybi,ybj))
		  value[1]++;
	    }
	  if (xbi == xii) // tate
	    if (smaller2(ybi,ybj,yii)||smaller2(yii,ybj,ybi)) {
	      if (yj == ybj) // yoko
		if (smaller2(xj,xbi,xbj)||smaller2(xbj,xbi,xj))
		  value[1]++;
	      if (yjj == ybj) // yoko
		if (smaller2(xbj,xbi,xjj)||smaller2(xjj,xbi,xbj))
		  value[1]++;
	    }
	  if (ybi == yii) // yoko
	    if (smaller2(xbi,xbj,xii)||smaller2(xii,xbj,xbi)) {
	      if (xj == xbj) // tate
		if (smaller2(yj,ybi,ybj)||smaller2(ybj,ybi,yj))
		  value[1]++;
	      if (xbj == xjj) // tate
		if (smaller2(ybj,ybi,yjj)||smaller2(yjj,ybi,ybj))
		  value[1]++;
	    }
	}
      }
  }

  boolean cross15() { // Alarm vs. Alarm
    int n = point.size();
    value[1] = 0;

    for (int i = 0; i < n-3; i++)
      for (int j = i+2; j < n-1; j++) {
	int xi = getx(i), xbi = getbx(i), xii = getx(i+1),
	    yi = gety(i), ybi = getby(i), yii = gety(i+1),
	    xj = getx(j), xbj = getbx(j), xjj = getx(j+1),
	    yj = gety(j), ybj = getby(j), yjj = gety(j+1);

	if (xi == xbi) // tate
	  if (smaller(yi,ybj,ybi)||smaller(ybi,ybj,yi)) {
	    if (yj == ybj) // yoko
	      if (smaller(xj,xbi,xbj)||smaller(xbj,xbi,xj)) {
		reverse(i+1, j);
		return true;
	      }
	    if (yjj == ybj) // yoko
	      if (smaller(xjj,xbi,xbj)||smaller(xbj,xbi,xjj)){
		reverse(i+1, j);
		return true;
	      }
	  }
	  if (yi == ybi) // yoko
	    if (smaller(xi,xbj,xbi)||smaller(xbi,xbj,xi)) {
	      if (xj == xbj) // tate
		if (smaller(yj,ybi,ybj)||smaller(ybj,ybi,yj)){
		  reverse(i+1, j);
		  return true;
		}
	      if (xbj == xjj) // tate
		if (smaller(ybj,ybi,yjj)||smaller(yjj,ybi,ybj)) {
		  reverse(i+1, j);
		  return true;
		}
	    }
	  if (xbi == xii) // tate
	    if (smaller(ybi,ybj,yii)||smaller(yii,ybj,ybi)) {
	      if (yj == ybj) // yoko
		if (smaller(xj,xbi,xbj)||smaller(xbj,xbi,xj)) {
		  reverse(i+1, j);
		  return true;
	}
	      if (yjj == ybj) // yoko
		if (smaller(xbj,xbi,xjj)||smaller(xjj,xbi,xbj)) {
		  reverse(i+1, j);
		  return true;
		}
	    }
	  if (ybi == yii) // yoko
	    if (smaller(xbi,xbj,xii)||smaller(xii,xbj,xbi)) {
	      if (xj == xbj) // tate
		if (smaller(yj,ybi,ybj)||smaller(ybj,ybi,yj)) {
		  reverse(i+1, j);
		  return true;
		}
	      if (xbj == xjj) // tate
		if (smaller(ybj,ybi,yjj)||smaller(yjj,ybi,ybj)) {
		  reverse(i+1, j);
		  return true;
		}
	    }
      }
    return false;
  }

  void cross2() { // Alarm vs. Wall_L
    int n = point.size();
    int m = wall_l.size();
    value[2] = 0;

    for (int i = 0; i < n-1; i++)
      for (int j = 0; j < m; j++) {

	int gsx = getsx(wall_l,j), ggx = getgx(wall_l,j),
	    gsy = getsy(wall_l,j), ggy = getgy(wall_l,j),
	    xi = getx(i), bxi = getbx(i), xii = getx(i+1),
	    yi = gety(i), byi = getby(i), yii = gety(i+1);

	if (gsx == ggx) { // tate
	  if (smaller(gsy,byi,ggy)||smaller(ggy,byi,gsy)) {
	    if (yi == byi) // yoko
	      if (smaller(xi,gsx,bxi)||smaller(bxi,gsx,xi))
		value[2]++;
	    if (byi == yii)  // yoko
	      if (smaller(bxi,gsx,xii)||smaller(xii,gsx,bxi))
		value[2]++;
	  }
	} else if (gsy == ggy) // yoko
	  if (smaller(gsx,bxi,ggx)||smaller(ggx,bxi,gsx)) {
	    if (xi == bxi) // tate
	      if  (smaller(yi,gsy,byi)||smaller(byi,gsy,yi))
		value[2]++;
	    if (bxi == xii) // tate
	      if (smaller(byi,gsy,yii)||smaller(yii,gsy,byi))
		value[2]++;
	  }
      }
  }

  void roomchange() {
    value[3] = 0;
    int n = point.size();
    for (int i = 0; i < n-1; i++)
      if (getid(i) != getid(i+1))
	value[3]++;
  }

  void cornering() {
    int n = point.size()-1;
    for (int i = 0; i < n; i++)
      if ((getx(i) != getx(i+1)) && (gety(i) != gety(i+1)))
	value[5]++;
    for (int i = 0; i < n-1; i++)
      if ((getbx(i) != getbx(i+1)) && (getby(i) != getby(i+1)))
	value[5]++;
  }

  void exchange(int m, int n) {
    if (m > n) { int tmp1 = m; m = n; n = tmp1; }
    Dot tmp = new Dot(getx(m), gety(m), getnum(m), getid(m));
    point.insertElementAt(tmp, n);
    point.removeElementAt(m);
    tmp = new Dot(getx(n), gety(n), getnum(n), getid(n));
    point.insertElementAt(tmp, m);
    point.removeElementAt(n+1);
  }

  void reverse(int i, int j) {
    int tmp = j - i;
    if ((tmp % 2) == 0)
      for (int p = 0; p < (tmp / 2); p++)
	exchange(i+p, j-p);
    else
      for (int p = 0; p < ((tmp+1) / 2); p++)
	exchange(i+p, j-p);
  }

  int distance(int pt1, int pt2) {
    return Math.abs(getx(pt1) - getx(pt2)) + Math.abs(gety(pt1) - gety(pt2));
  }

  int direction_x(int pt1, int pt2) {
    int tmp = 0;

    if (pt1 == 0)
      switch(init) {
      case 0:
	tmp = getx(pt1);
	break;
      case 3:
	if (gety(pt1) == gety(pt2))
	  tmp = getx(pt1);
	else
	  tmp = getx(pt2);
	break;
      default:
	System.err.println("ERROR.");
      }
    else
      if (getby(pt1-1) == gety(pt1))
	if (gety(pt1) == gety(pt2))
/**/	  tmp = (getx(pt1) + getx(pt2)) / 2;
	else if (getx(pt1) == getx(pt2))
	  tmp = getx(pt1);
	else if (smaller(getbx(pt1-1),getx(pt1),getx(pt2))
		 ||smaller(getx(pt2),getx(pt1),getbx(pt1-1)))
	  tmp = getx(pt2);
	else
	  if (getbx(pt1-1) == getx(pt2))
	    tmp = getx(pt2);
/**///	  else if (getx(pt1-1) == getx(pt2))
//	    tmp = getx(pt2);
	  else
	    tmp = getx(pt1);
      else if (getbx(pt1-1) == getx(pt1))
	if (getx(pt1) == getx(pt2))
	  tmp = getx(pt1);
	else if (gety(pt1) == gety(pt2))
/**/	  tmp = (getx(pt1) + getx(pt2)) / 2;
	else if (smaller(getby(pt1-1),gety(pt1),gety(pt2))
		 ||smaller(gety(pt2),gety(pt1),getby(pt1-1)))
	  tmp = getx(pt1);
	else
	  if (getby(pt1-1) == gety(pt2))
	    tmp = getx(pt1);
/**///	  else if (gety(pt1-1) == gety(pt2))
//	    tmp = getx(pt1);
	  else
	    tmp = getx(pt2);
      else
	System.out.println("ERROR.-direction_x");
    return tmp;
  }

  int direction_y(int pt1, int pt2) {
    int tmp = 0;

    if (pt1 == 0)
      switch(init) {
      case 0:
	if (getx(pt1) == getx(pt2))
	  tmp = gety(pt1);
	else
	  tmp = gety(pt2);
	break;
      case 3:
	tmp = gety(pt1);
	break;
      default:
	System.err.println("ERROR.");
      }
    else
      if (getby(pt1-1) == gety(pt1))
	if (gety(pt1) == gety(pt2))
	  tmp = gety(pt1);
	else if (getx(pt1) == getx(pt2))
/**/	  tmp = (gety(pt1) + gety(pt2)) / 2;
	else if (smaller(getbx(pt1-1),getx(pt1),getx(pt2))
		 ||smaller(getx(pt2),getx(pt1),getbx(pt1-1)))
	  tmp = gety(pt1);
	else
	  if (getbx(pt1-1) == getx(pt2))
	    tmp = gety(pt1);
/**///	  else if (getx(pt1-1) == getx(pt2))
//	    tmp = gety(pt1);
	  else
	    tmp = gety(pt2);
      else if (getbx(pt1-1) == getx(pt1))
	if (getx(pt1) == getx(pt2))
/**/	  tmp = (gety(pt1) + gety(pt2)) / 2;
	else if (gety(pt1) == gety(pt2))
	  tmp = gety(pt1);
	else if (smaller(getby(pt1-1),gety(pt1),gety(pt2))
		 ||smaller(gety(pt2),gety(pt1),getby(pt1-1)))
	  tmp = gety(pt2);
	else
	  if (getby(pt1-1) == gety(pt2))
	    tmp = gety(pt2);
/**///	  else if (gety(pt1-1) == gety(pt2))
//	    tmp = gety(pt2);
	  else
	    tmp = gety(pt1);
      else
	System.out.println("ERROR.-direction_y");
    return tmp;
  }

  int duplication(int pt1, int pt2) {
    int tmp = 0;

    if (pt1 == 0)
      tmp = 0;
    else if (pt1 > 0) {
      if ((getbx(pt1-1) == getbx(pt1)) && (getby(pt1-1) == getby(pt1)))
	tmp = Math.abs(getx(pt1)-getbx(pt1)) + Math.abs(gety(pt1)-gety(pt1-1));
      else if ((getx(pt1-1) == getbx(pt1)) && (gety(pt1-1) == getby(pt1)))
	tmp = Math.abs(getx(pt1-1)-getx(pt1)) + Math.abs(gety(pt1-1)-gety(pt1));
      else if ((getbx(pt1-1) == getx(pt2)) && (getby(pt1-1) == gety(pt2)))
	tmp = Math.abs(getx(pt1)-getx(pt2)) + Math.abs(gety(pt1)-gety(pt2));
    } else
      System.err.println("ERROR.-duplication");
    return tmp;
  }

  void normalsort() {
    int n = point.size();
    for (int i = 0; i < n-1; i++) {
      int tmp_dist = distance(i,i+1);
      for (int j = i+2; j < n; j++)
	if (tmp_dist > distance(i,j)) {
	  tmp_dist = distance(i,j);
	  exchange(i+1, j);
	}
    }
    sort = true;
  }

  void randomsort() {
    int n = point.size();
    for (int i = 0; i < n-1; i++) {
      int tmp_dist = distance(i,i+1);
      for (int j = 0; j < (n-i-2) - (n-i-2)/2; j++) {
	int random = (int)(Math.random()*(n-i-2)) + (i+2);
	if (tmp_dist > distance(i,random)) {
	  tmp_dist = distance(i,random);
	  exchange(i+1, random);
	}
      }
    }
    sort = true;
  }

  void cross() { // Alarm vs. Alarm
    int n = point.size();

    for (int i = 0; i < n-3; i++)
      for (int j = i+2; j < n-1; j++) {

	int xi = getx(i), xbi = getbx(i), xii = getx(i+1),
	    yi = gety(i), ybi = getby(i), yii = gety(i+1),
	    xj = getx(j), xbj = getbx(j), xjj = getx(j+1),
	    yj = gety(j), ybj = getby(j), yjj = gety(j+1);

	if (xi == xbi) // tate
	  if (smaller(yi,ybj,ybi)||smaller(ybi,ybj,yi)) {
	    if (yj == ybj) // yoko
	      if (smaller(xj,xbi,xbj)||smaller(xbj,xbi,xj)) {
		reverse(i+1,j);
		return;
	      }
	    if (yjj == ybj) // yoko
	      if (smaller(xjj,xbi,xbj)||smaller(xbj,xbi,xjj)) {
		reverse(i+1,j);
		return;
	      }
	  }
	if (yi == ybi) // yoko
	  if (smaller(xi,xbj,xbi)||smaller(xbi,xbj,xi)) {
	    if (xj == xbj) // tate
	      if (smaller(yj,ybi,ybj)||smaller(ybj,ybi,yj)) {
		reverse(i+1,j);
		return;
	      }
	    if (xbj == xjj) // tate
	      if (smaller(ybj,ybi,yjj)||smaller(yjj,ybi,ybj)) {
		reverse(i+1,j);
		return;
	      }
	  }
	if (xbi == xii) // tate
	  if (smaller(ybi,ybj,yii)||smaller(yii,ybj,ybi)) {
	    if (yj == ybj) // yoko
	      if (smaller(xj,xbi,xbj)||smaller(xbj,xbi,xj)) {
		reverse(i+1,j);
		return;
	      }
	    if (yjj == ybj) // yoko
	      if (smaller(xbj,xbi,xjj)||smaller(xjj,xbi,xbj)) {
		reverse(i+1,j);
		return;
	      }
	  }
	if (ybi == yii) // yoko
	  if (smaller(xbi,xbj,xii)||smaller(xii,xbj,xbi)) {
	    if (xj == xbj) // tate
	      if (smaller(yj,ybi,ybj)||smaller(ybj,ybi,yj)) {
		reverse(i+1,j);
		return;
	      }
	    if (xbj == xjj) // tate
	      if (smaller(ybj,ybi,yjj)||smaller(yjj,ybi,ybj)) {
		reverse(i+1,j);
		return;
	      }
	  }
      }
  }

}
