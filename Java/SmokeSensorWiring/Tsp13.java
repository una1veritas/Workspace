// Tsp13.java

import java.io.*;
import java.awt.*;
import java.util.*;


import Alarm;
import Wall;

class Tsp13 extends Frame{
  static Alarm alarm = new Alarm();

  long start_t, end_t;

  public Tsp13() {
    super("Tsp13");

    Menu exMenu = new Menu("Menu");
    exMenu.add(new MenuItem("RandomChange"));
    exMenu.add(new MenuItem("RandomChangei2"));
    exMenu.add(new MenuItem("RandomChangei2Z"));
    exMenu.add(new MenuItem("RandomChangei2Z10"));
    exMenu.add(new MenuItem("r16"));
    exMenu.add(new MenuItem("r17"));
    exMenu.add(new MenuItem("r18"));
    exMenu.add(new MenuItem("r19"));
    exMenu.add(new MenuItem("r20"));
    exMenu.add(new MenuItem("Reset"));
    exMenu.add(new MenuItem("RandomSet"));
    exMenu.add(new MenuItem("RandomSet_Point"));
    exMenu.add(new MenuItem("RandomSet_Wall"));
    exMenu.add(new MenuItem("RandomSet_Wall_l"));
    exMenu.add(new MenuItem("RandomSet_Wall_h"));
    exMenu.add(new MenuItem("ResetPoint"));
    exMenu.add(new MenuItem("ResetWall"));
    exMenu.add(new MenuItem("ResetWall_h"));
    exMenu.add(new MenuItem("ResetWall_l"));
    exMenu.addSeparator();
    exMenu.add(new MenuItem("Quit"));

    Menu exMenu2 = new Menu("Operate");
    exMenu2.add(new MenuItem("Best_Change_Cross"));
    exMenu2.add(new MenuItem("Best_Change"));
    exMenu2.add(new MenuItem("Best_Change2"));
    exMenu2.add(new MenuItem("Best_Changei2"));
    exMenu2.add(new MenuItem("Change"));
    exMenu2.add(new MenuItem("Change2"));
    exMenu2.add(new MenuItem("Cross"));
    exMenu2.add(new MenuItem("Cross1"));
    exMenu2.add(new MenuItem("Cross15"));
    exMenu2.add(new MenuItem("Cross2"));
    exMenu2.add(new MenuItem("Best"));

    Menu SortChangeMenu = new Menu("Sort_Change");
    exMenu2.add(SortChangeMenu);
    SortChangeMenu.add(new MenuItem("Sort_x_Change"));
    SortChangeMenu.add(new MenuItem("Sort_y_Change"));

    exMenu2.add(new MenuItem("RandomSort"));

    Menu SortMenu = new Menu("Sort");
    exMenu2.add(SortMenu);
    SortMenu.add(new MenuItem("Sort_x"));
    SortMenu.add(new MenuItem("Sort_y"));

    Menu StartMenu = new Menu("Start");
    exMenu2.add(StartMenu);
    StartMenu.add(new MenuItem("Start_x"));
    StartMenu.add(new MenuItem("Start_y"));

    exMenu2.add(new MenuItem("Print"));
 
    MenuBar mb = new MenuBar();
    mb.add(exMenu);
    mb.add(exMenu2);
    setMenuBar(mb);
  		//{{INIT_CONTROLS
		setLayout(null);
		setSize(430,270);
		setVisible(false);
		setTitle("–³‘è");
		//}}
		//{{INIT_MENUS
		//}}
}

  public void begin() {
    resize(830, 800);
    System.out.println("Window is opening.");
    show();
  }

  public boolean handleEvent(Event e) {
    if (e.id == Event.WINDOW_DESTROY) {
      dispose();
      System.exit(0);
      return true;
    } else
      return super.handleEvent(e);
  }

  public boolean action(Event evt, Object arg) {
    if (evt.target instanceof MenuItem) {
      MenuItem mi = (MenuItem)evt.target;
      if ("Quit".equals(mi.getLabel())) {
	System.out.println("Exiting... See you.");
	dispose();
	System.exit(0);
	return true;

      } else if ("r17".equals(mi.getLabel())) {
	try {
	  alarm.resetPoint();
	  String line_str;
	  int line_number;
	  FileInputStream is = new FileInputStream("r171");
	  DataInputStream ds = new DataInputStream(is);
	  for (int i = 0; i < 17; i++) {
	    line_str = ds.readLine();
	    int tmp_x = Integer.parseInt(line_str);
	    line_str = ds.readLine();
	    int tmp_y = Integer.parseInt(line_str);
	    line_str = ds.readLine();
	    int tmp_id = Integer.parseInt(line_str);
	    alarm.point.addElement(new Dot(tmp_x, tmp_y, i, tmp_id));
	  }
	  ds.close();
	} catch (IOException e) {
	  System.out.println("File error: " + e);
	}
	repaint();
      } else if ("r19".equals(mi.getLabel())) {
	try {
	  alarm.resetPoint();
	  String line_str;
	  int line_number;
	  FileInputStream is = new FileInputStream("r191");
	  DataInputStream ds = new DataInputStream(is);
	  for (int i = 0; i < 19; i++) {
	    line_str = ds.readLine();
	    int tmp_x = Integer.parseInt(line_str);
	    line_str = ds.readLine();
	    int tmp_y = Integer.parseInt(line_str);
	    line_str = ds.readLine();
	    int tmp_id = Integer.parseInt(line_str);
	    alarm.point.addElement(new Dot(tmp_x, tmp_y, i, tmp_id));
	  }
	  ds.close();
	} catch (IOException e) {
	  System.out.println("File error: " + e);
	}
	repaint();
       } else if ("r20".equals(mi.getLabel())) {
	try {
	  alarm.resetPoint();
	  String line_str;
	  int line_number;
	  FileInputStream is = new FileInputStream("r201");
	  DataInputStream ds = new DataInputStream(is);
	  for (int i = 0; i < 20; i++) {
	    line_str = ds.readLine();
	    int tmp_x = Integer.parseInt(line_str);
	    line_str = ds.readLine();
	    int tmp_y = Integer.parseInt(line_str);
	    line_str = ds.readLine();
	    int tmp_id = Integer.parseInt(line_str);
	    alarm.point.addElement(new Dot(tmp_x, tmp_y, i, tmp_id));
	  }
	  ds.close();
	} catch (IOException e) {
	  System.out.println("File error: " + e);
	}
	repaint();
      } else if ("r18".equals(mi.getLabel())) {
	try {
	  alarm.resetPoint();
	  String line_str;
	  int line_number;
	  FileInputStream is = new FileInputStream("r181");
	  DataInputStream ds = new DataInputStream(is);
	  for (int i = 0; i < 18; i++) {
	    line_str = ds.readLine();
	    int tmp_x = Integer.parseInt(line_str);
	    line_str = ds.readLine();
	    int tmp_y = Integer.parseInt(line_str);
	    line_str = ds.readLine();
	    int tmp_id = Integer.parseInt(line_str);
	    alarm.point.addElement(new Dot(tmp_x, tmp_y, i, tmp_id));
	  }
	  ds.close();
	} catch (IOException e) {
	  System.out.println("File error: " + e);
	}
	repaint();
      } else if ("r16".equals(mi.getLabel())) {
	try {
	  alarm.resetPoint();
	  String line_str;
	  int line_number;
	  FileInputStream is = new FileInputStream("r161");
	  DataInputStream ds = new DataInputStream(is);
	  for (int i = 0; i < 16; i++) {
	    line_str = ds.readLine();
	    int tmp_x = Integer.parseInt(line_str);
	    line_str = ds.readLine();
	    int tmp_y = Integer.parseInt(line_str);
	    line_str = ds.readLine();
	    int tmp_id = Integer.parseInt(line_str);
	    alarm.point.addElement(new Dot(tmp_x, tmp_y, i, tmp_id));
	  }
	  ds.close();
	} catch (IOException e) {
	  System.out.println("File error: " + e);
	}
	repaint();
     } else if ("RandomChange".equals(mi.getLabel())) {
//	int c[] = new int[20];
	int p[] = new int[30];
	for (int j = 2; j < 30; j++) {
	  System.out.print(j+" ");
	  p[j] = 0;
	  for (int z = 0; z < 20; z++) {
//	    alarm.randomWall_h(5);
//	    alarm.randomWall_l(10);
	    alarm.resetPoint();
	    for (int i = 0; i < j; i++) {
	      int s = alarm.point.size();
	      alarm.point.addElement(new Dot(s));
	    }
	    start_t = System.currentTimeMillis();
	    alarm.sort_x();
	    Alarm tmp_a = new Alarm(alarm);
	    tmp_a.sort_y();
	    if (!hyoka(alarm, tmp_a))
	      alarm = tmp_a;
	    int t = changei();
//	    c[z] = changei();
	    end_t = System.currentTimeMillis();
	    repaint();
	    System.out.print(" "+(end_t - start_t));
	    p[j] += (end_t - start_t);
	  }
	  System.out.println();
	  System.out.println((p[j]/20));
	}
	return true;

     } else if ("RandomChangei2".equals(mi.getLabel())) {
//	int c[] = new int[20];
	int p[] = new int[30];
	for (int j = 2; j < 30; j++) {
	  System.out.print(j+" ");
	  p[j] = 0;
	  for (int z = 0; z < 20; z++) {
//	    alarm.randomWall_h(5);
//	    alarm.randomWall_l(10);
	    alarm.resetPoint();
	    for (int i = 0; i < j; i++) {
	      int s = alarm.point.size();
	      alarm.point.addElement(new Dot(s));
	    }
	    start_t = System.currentTimeMillis();
	    alarm.sort_x();
	    Alarm tmp_a = new Alarm(alarm);
	    tmp_a.sort_y();
	    int t = 1;
	    while (((alarm.value[0] != 0) || (alarm.value[1] != 0)) && (t != 0))
	      t = changei2();
	    end_t = System.currentTimeMillis();
	    repaint();
	    System.out.print(" "+(end_t - start_t));
	    p[j] += (end_t - start_t);
	  }
	  System.out.println();
	  System.out.println((p[j]/20));
	}
	return true;

      } else if ("RandomChangei2Z".equals(mi.getLabel())) {
	int h[] = new int[8];
	int c[] = new int[20];
	int p[] = new int[30];
	for (int j = 2; j < 30; j++) {
	  for (int d = 0; d < 8; d++)
	    h[d] = 0;
	  for (int z = 0; z < 20; z++) {
	    alarm.randomWall_h(5);
	    alarm.randomWall_l(10);
	    alarm.resetPoint();
	    for (int i = 0; i < j; i++) {
	      int s = alarm.point.size();
	      alarm.point.addElement(new Dot(s));
	    }
	    alarm.sort_x();
	    Alarm tmp_a = new Alarm(alarm);
	    tmp_a.sort_y();
	    int t = 1;
	    while (((alarm.value[0] != 0) || (alarm.value[1] != 0)) && (t != 0))
	      t = changei2();
	    repaint();
	    for (int d = 0; d < 8; d++)
	      h[d] += alarm.value[d];
	  }
	  System.out.println();
	  for (int d = 0; d < 8; d++)
	    System.out.print(h[d]+" ");
	  System.out.println();
	  for (int d = 0; d < 8; d++)
	    System.out.print(h[d]/20+" ");
	  System.out.println();
	}
	return true;

      } else if ("RandomChangei2Z10".equals(mi.getLabel())) {
	int h[] = new int[8];
	int c[] = new int[20];
	int p[] = new int[30];
	for (int j = 2; j < 30; j++) {
	  for (int d = 0; d < 8; d++)
	    h[d] = 0;
	  for (int z = 0; z < 10; z++) {
	    alarm.randomWall_h(5);
	    alarm.randomWall_l(10);
	    alarm.resetPoint();
	    for (int i = 0; i < j; i++) {
	      int s = alarm.point.size();
	      alarm.point.addElement(new Dot(s));
	    }
	    alarm.sort_x();
	    Alarm tmp_a = new Alarm(alarm);
	    tmp_a.sort_y();
	    int t = 1;
	    while (((alarm.value[0] != 0) || (alarm.value[1] != 0)) && (t != 0))
	      t = changei2();
	    repaint();
	    for (int d = 0; d < 8; d++)
	      h[d] += alarm.value[d];
	  }
	  System.out.println();
	  for (int d = 0; d < 8; d++)
	    System.out.print(h[d]+" ");
	  System.out.println();
	  for (int d = 0; d < 8; d++)
	    System.out.print(h[d]/20+" ");
	  System.out.println();
	}
	return true;

      } else if ("Reset".equals(mi.getLabel())) {
	alarm.resetPoint();
	alarm.resetWall();
	repaint();
	System.out.println("All Clear.");
	return true;
      } else if ("RandomSet".equals(mi.getLabel())) {
	alarm.randomPoint(30);
	alarm.randomWall(5, 10);
	repaint();
	return true;
      } else if ("RandomSet_Point".equals(mi.getLabel())) {
	alarm.randomPoint(30);
	repaint();
	return true;
      } else if ("RandomSet_Wall".equals(mi.getLabel())) {
	alarm.randomWall(5,10);
	repaint();
	return true;
      } else if ("RandomSet_Wall_l".equals(mi.getLabel())) {
	alarm.randomWall_l(10);
	repaint();
	return true;
      } else if ("RandomSet_Wall_h".equals(mi.getLabel())) {
	alarm.randomWall_h(5);
	repaint();
	return true;
      } else if ("ResetPoint".equals(mi.getLabel())) {
	alarm.resetPoint();
	repaint();
	System.out.println("Point Clear.");
	return true;
      } else if ("ResetWall".equals(mi.getLabel())) {
	alarm.resetWall();
	repaint();
	System.out.println("Wall Clear.");
	return true;
      } else if ("ResetWall_h".equals(mi.getLabel())) {
	alarm.resetWall_h();
	repaint();
	System.out.println("Wall_h Clear.");
	return true;
      } else if ("ResetWall_l".equals(mi.getLabel())) {
	alarm.resetWall_l();
	repaint();
	System.out.println("Wall_l Clear.");
	return true;
      } else if (alarm.point.size() > 0) {
	if ("Best".equals(mi.getLabel())) {
	  alarm.sort_x(); alarm.print();
	  Alarm tmp_a = new Alarm(alarm);
	  tmp_a.sort_y(); tmp_a.print();
	  if (!hyoka(alarm, tmp_a)) {
	    System.out.println("Start_y is better.");
	    alarm = tmp_a;
	  } else
	    System.out.println("Start_x is better.");
	  alarm.print();
	  repaint();
	  return true;
	} else if ("Best_Change_Cross".equals(mi.getLabel())) {
	  start_t = System.currentTimeMillis();
	  alarm.sort_x(); alarm.print();
	  Alarm tmp_a = new Alarm(alarm);
	  tmp_a.sort_y(); tmp_a.print();
	  if (!hyoka(alarm, tmp_a)) {
	    System.out.println("Start_y is better.");
	    alarm = tmp_a;
	  } else
	    System.out.println("Start_x is better.");
	  System.out.println("Changing time.");
	  change();
	  System.out.println("Change is finished.");
	  alarm.print();
	  boolean flag = true;
	  while(flag){
	    Alarm tmp = new Alarm(alarm);
	    if (tmp.cross15()) {
	      tmp.path();
	      if (!hyoka(alarm, tmp))
		alarm = tmp;
	      else
		flag = false;
	    } else
	      flag = false;
	  }
	  end_t = System.currentTimeMillis();
	  alarm.print();
	  repaint();
	  System.out.println("--- "+(end_t - start_t)+" msec passed.");
	  return true;
	} else if ("Best_Change".equals(mi.getLabel())) {
	  start_t = System.currentTimeMillis();
	  alarm.sort_x(); alarm.print();
	  Alarm tmp_a = new Alarm(alarm);
	  tmp_a.sort_y(); tmp_a.print();
	  if (!hyoka(alarm, tmp_a)) {
	    System.out.println("Start_y is better.");
	    alarm = tmp_a;
	  } else
	    System.out.println("Start_x is better.");
	  System.out.println("Changing time.");
	  change();
	  end_t = System.currentTimeMillis();
	  alarm.print();
	  repaint();
	  System.out.println("--- "+(end_t - start_t)+" msec passed.");
	  return true;
	} else if ("Best_Change2".equals(mi.getLabel())) {
	  start_t = System.currentTimeMillis();
	  alarm.sort_x(); alarm.print();
	  Alarm tmp_a = new Alarm(alarm);
	  tmp_a.sort_y(); tmp_a.print();
	  if (!hyoka(alarm, tmp_a)) {
	    System.out.println("Start_y is better.");
	    alarm = tmp_a;
	  } else
	    System.out.println("Start_x is better.");
	  System.out.println("Changing time.");
	  change15();
	  end_t = System.currentTimeMillis();
	  repaint();
	  alarm.print();
	  System.out.println("--- "+(end_t - start_t)+" msec passed.");
	  return true;
	} else if ("Best_Changei2".equals(mi.getLabel())) {
	  start_t = System.currentTimeMillis();
	  alarm.sort_x(); alarm.print();
	  Alarm tmp_a = new Alarm(alarm);
	  tmp_a.sort_y(); tmp_a.print();
	  if (!hyoka(alarm, tmp_a)) {
	    System.out.println("Start_y is better.");
	    alarm = tmp_a;
	  } else
	    System.out.println("Start_x is better.");
	  System.out.println("Changing time.");
	  int t = 1;
	  while (((alarm.value[0] != 0) || (alarm.value[1] != 0)) && (t != 0))
	    t = changei2();
	  end_t = System.currentTimeMillis();
	  repaint();
	  alarm.print();
	  System.out.println("--- "+(end_t - start_t)+" msec passed.");
	  return true;
	} else if ("Print".equals(mi.getLabel())) {
	  for (int i = 0; i < alarm.point.size()-1; i++)
	    System.out.println("distance(pt[" + alarm.getnum(i) + "], pt[" + alarm.getnum(i+1) + "] is " + alarm.distance(i, i+1) + " - " + alarm.duplication(i, i+1) + " = " + (alarm.distance(i,i+1)-alarm.duplication(i,i+1)));
	  alarm.print();
	  repaint();
	  return true;
	} else if ("Sort_x_Change".equals(mi.getLabel())) {
	  alarm.sort_x();
	  change();
	  alarm.print();
	  repaint();
	  return true;
	} else if ("Sort_y_Change".equals(mi.getLabel())) {
	  alarm.sort_y();
	  change();
	  alarm.print();
	  repaint();
	  return true;
	} else if ("Sort_x".equals(mi.getLabel())) {
	  alarm.sort_x();
	  alarm.print();
	  repaint();
	  return true;
	} else if ("Sort_y".equals(mi.getLabel())) {
	  alarm.sort_y();
	  alarm.print();
	  repaint();
	  return true;
	} else if ("RandomSort".equals(mi.getLabel())) {
	  alarm.randomsort(); alarm.path();
	  alarm.print();
	  repaint();
	  return true;
	} else if ("Cross".equals(mi.getLabel())) {
	  if (!alarm.sort)
	    System.err.println("Can't cross yet.");
	  else {
	    alarm.cross();
	    alarm.path();
	    repaint();
	    alarm.print();
	  }
	  return true;
	} else if ("Cross1".equals(mi.getLabel())) {
	  if (!alarm.sort)
	    System.err.println("Can't cross1 yet.");
	  else {
	    alarm.path();
	    start_t = System.currentTimeMillis();
	    alarm.cross1();
	    end_t = System.currentTimeMillis();
	    System.out.println("Cross1 = "+alarm.value[1]);
	    System.out.println("--- "+(end_t - start_t)+" msec passed.");
/*	    start_t = System.currentTimeMillis();
	    cross15(alarm);
	    end_t = System.currentTimeMillis();
	    System.out.println("Cross15 = "+alarm.value[1]);
	    System.out.println("--- "+(end_t - start_t)+" msec passed.");
*/	  }
	  return true;
	} else if ("Cross15".equals(mi.getLabel())) {
	  if (!alarm.sort)
	    System.err.println("Can't cross1 yet.");
	  else {
	    alarm.path();
	    start_t = System.currentTimeMillis();

	    boolean flag = true;
	    while(flag){
	      Alarm tmp = new Alarm(alarm);
	      if (tmp.cross15()) {
		tmp.path();
		if (!hyoka(alarm, tmp))
		  alarm = tmp;
		else
		  flag = false;
	      } else
		flag = false;
	    }

	    end_t = System.currentTimeMillis();
	    System.out.println("Cross15 = "+alarm.value[1]);
	    System.out.println("--- "+(end_t - start_t)+" msec passed.");
	    repaint();
	  }
	  return true;

	} else if ("Cross2".equals(mi.getLabel())) {
	  if (!alarm.sort)
	    System.err.println("Can't cross yet.");
	  else {
	    alarm.path();
	    start_t = System.currentTimeMillis();
	    alarm.cross2();
	    end_t = System.currentTimeMillis();
	    System.out.println("Cross2 = "+alarm.value[2]);
	    System.out.println("--- "+(end_t - start_t)+" msec passed.");
/*	    start_t = System.currentTimeMillis();
	    cross25(alarm,wall_l,alarm.value[2]);
	    end_t = System.currentTimeMillis();
	    System.out.println("Cross25 = "+alarm.value[2]);
	    System.out.println("--- "+(end_t - start_t)+" msec passed.");
*/	  }
	  return true;
	} else if ("Change".equals(mi.getLabel())) {
	  if (!alarm.sort)
	    System.err.println("Can't change yet.");
	  else {
	    System.out.println("Changing time.");
	    change();
	    alarm.print();
	    repaint();
	  }
	  return true;
	} else if ("Change2".equals(mi.getLabel())) {
	  if (!alarm.sort)
	    System.err.println("Can't change2 yet.");
	  else {
	    System.out.println("Changing time.");
	    change2();
	    alarm.print();
	    repaint();
	  }
	  return true;
	} else if ("Start_x".equals(mi.getLabel())) {
	  alarm.start_x();alarm.path();
	  alarm.sort = true;
	  alarm.print();
	  repaint();
	  return true;
	} else if ("Start_y".equals(mi.getLabel())) {
	  alarm.start_y(); alarm.path();
	  alarm.sort = true;
	  alarm.print();
	  repaint();
	  return true;
	} else
	  return false;
      } else
	System.err.println("No Alarm.");
    }
    return false;
  }

  public boolean mouseDown(Event evt, int px, int py) {
    alarm.sort = false;
    int pn = alarm.point.size();
    int pid = 0;
    alarm.point.addElement(new Dot(px, py, pn, pid));
    System.out.println("pt[" +pn+ "] is (" +px+ ", "  +py+ ", " +pid+ ")");
    repaint();
    return true;
  }

  void paint_wall(Graphics g, Vector v, Color c) {
    int s = v.size();
    if (s > 0) {
      g.setColor(c);
      for (int i = 0; i < s; i++) {
	g.drawLine(alarm.getsx(v,i), alarm.getsy(v,i),alarm.getgx(v,i),alarm.getgy(v,i));
      }
    }
  }

  public void paint(Graphics g) {
    paint_wall(g, alarm.wall_h, Color.black); // forbidden obstacle line
    paint_wall(g, alarm.wall_l, Color.lightGray);
    int s = alarm.point.size();
    if (s > 0) {
      g.setColor(Color.red);
      g.fillRect(alarm.getx(0)-5, alarm.gety(0)-5, 10, 10);
      for (int i = 1; i < s; i++) {
	g.setColor(Color.black);
	g.drawOval(alarm.getx(i)-5, alarm.gety(i)-5, 10, 10);
	g.drawString("" + alarm.getnum(i), alarm.getx(i)+5, alarm.gety(i)+5);
      }
      if (alarm.sort) {
	g.setColor(Color.red);
	for (int i = 0; i < s-1; i++) {
	  g.drawLine(alarm.getx(i),alarm.gety(i),alarm.getbx(i),alarm.getby(i));
	  g.drawLine(alarm.getbx(i),alarm.getby(i),alarm.getx(i+1),alarm.gety(i+1));
	}
      }
    }
  }

  static void inputPoint() {
    try {
      System.out.print("Please input number of point : ");
      System.out.flush();
      String line = new DataInputStream(System.in).readLine();
      inputPoint(line);
    } catch (java.io.IOException e) {
      System.out.println("Could not get input. Sorry.");
      System.exit(1);
    }
  }

  static void inputPoint(String count) {
    int c = 0;
    try {
      c = Integer.parseInt(count);
    } catch (java.lang.NumberFormatException e) {
      System.err.println("Could not get input. Sorry.");
      System.exit(1);
    }
    if (c <= 0) {
      System.out.println("Please input integer more than 0");
      System.exit(1);
    } else
      askPoint(c);
  }

  static void askPoint(int n) {
    try {
      int i = 0;
      while(i < n) {
	System.out.print("x[" +i+ "] is ? ");
	System.out.flush();
	String line = new DataInputStream(System.in).readLine();
	int lx = Integer.parseInt(line);
	System.out.print("y[" +i+ "] is ? ");
	System.out.flush();
	line = new DataInputStream(System.in).readLine();
	int ly = Integer.parseInt(line);
	System.out.print("id[" +i+ "] is ? ");
	System.out.flush();
	line = new DataInputStream(System.in).readLine();
	int lid = Integer.parseInt(line);
	if ((lx < 0) || (ly < 0) || (lid < 0))
	  System.err.println("Please input more than 0");
	else if ((lx > 800) || (ly > 800))
	  System.err.println("Please input less than 800");
	else {
	  alarm.point.addElement(new Dot(lx, ly, i, lid));
	  i++;
	}
      }
    } catch (java.io.IOException e) {
      System.err.println("Could not get input. Sorry.");
      System.exit(1);
    } catch (java.lang.NumberFormatException e) {
      System.err.println("Could not get input. Sorry.");
      System.exit(1);
    }
  }

  static void inputWall() {
    try {
      System.out.print("Please input number of wall_h : ");
      System.out.flush();
      String line_h = new DataInputStream(System.in).readLine();
      System.out.print("Please input number of wall_l : ");
      System.out.flush();
      String line_l = new DataInputStream(System.in).readLine();
      System.out.println("Wall_H");
      inputWall(alarm.wall_l, line_l);
      System.out.println("Wall_L");
      inputWall(alarm.wall_h, line_h);
    } catch (java.io.IOException e) {
      System.out.println("Could not get input. Sorry.");
      System.exit(1);
    }
  }

  static void inputWall(Vector vector, String count) {
    int c = 0;
    try {
      c = Integer.parseInt(count);
    } catch (java.lang.NumberFormatException e) {
      System.err.println("Could not get input. Sorry.");
      System.exit(1);
    }
    if (c <= 0) {
      System.out.println("Please input integer more than 0");
      System.exit(1);
    } else {
      askWall(vector, c);
    }
  }

  static void askWall(Vector vector, int n) {
    try {
      int i = 0;
      while(i < n) {
	System.out.print("sx[" +i+ "] is ? ");
	System.out.flush();
	String line = new DataInputStream(System.in).readLine();
	int lsx = Integer.parseInt(line);
	System.out.print("sy[" +i+ "] is ? ");
	System.out.flush();
	line = new DataInputStream(System.in).readLine();
	int lsy = Integer.parseInt(line);
	System.out.print("gx[" +i+ "] is ? ");
	System.out.flush();
	line = new DataInputStream(System.in).readLine();
	int lgx = Integer.parseInt(line);
	System.out.print("gy[" +i+ "] is ? ");
	System.out.flush();
	line = new DataInputStream(System.in).readLine();
	int lgy = Integer.parseInt(line);
	if ((lsx < 0) || (lsy < 0) || (lgx < 0) || (lgy < 0))
	  System.err.println("Please input more than 0");
	else if ((lsx > 800) || (lsy > 800) || (lgx > 800) || (lgy > 800))
	  System.err.println("Please input less than 800");
	else {
	  vector.addElement(new Wall(lsx, lsy, lgx, lgy));
	  i++;
	}
      }
    } catch (java.io.IOException e) {
      System.err.println("Could not get input. Sorry.");
      System.exit(1);
    } catch (java.lang.NumberFormatException e) {
      System.err.println("Could not get input. Sorry.");
      System.exit(1);
    }
  }

  boolean hyoka(Alarm p, Alarm q) { // (p<=q) --> true; (p>q) --> false
    for (int i = 0; i < 8; i++)
      if (p.value[i] < q.value[i])
	return true;
      else if (p.value[i] > q.value[i])
	return false;
    return true;
  }

  int changei() {
    boolean flag = false;
    int n = alarm.point.size();
    int c = 0;
    while(!flag) {
      flag = true;
      for (int i = 0; (i < n-2) && flag; i++)
	for (int j = i+2; (j < n) && flag; j++) {
	  Alarm tmp = new Alarm(alarm);
	  tmp.reverse(i+1, j);
	  tmp.path();
	  if (!hyoka(alarm, tmp)) {
	    alarm = tmp;
	    c++;
	    flag = false;
	  }
	}
    }
    return c;
  }
  int changei2() {
    int n = alarm.point.size();
    int c = 0;
    for (int i = 0; (i < n-2); i++) {
      boolean flag = true;
      for (int j = i+2; (j < n); j++) {
	Alarm tmp = new Alarm(alarm);
	tmp.reverse(i+1, j);
	tmp.path();
	  int k = 0;
	  while((k < 8)) {
	    if (tmp.value[k] < alarm.value[k]) {
	      alarm = tmp;
//	      System.out.println("CHANGED-"+k+"!!! ("+i+","+(i+1)+") <-> ("+j+","+(j+1)+")");
	      flag = false;
	    } else if (tmp.value[k] == alarm.value[k])
	      k++;
	    else
	      break;
	  }
      }
      if (!flag)
	c++;
    }
    return c;
  }

  void change() {
    boolean flag = false;
    int n = alarm.point.size();
    while(!flag) {
      flag = true;
      for (int i = 0; (i < n-2) && flag; i++)
	for (int j = i+2; (j < n) && flag; j++) {
	  Alarm tmp = new Alarm(alarm);
	  tmp.reverse(i+1, j);
	  tmp.path();

	  int k = 0;
	  while(flag && (k < 8)) {
	    if (tmp.value[k] < alarm.value[k]) {
	      alarm = tmp;
	      System.out.println("CHANGED-"+k+"!!! ("+i+","+(i+1)+") <-> ("+j+","+(j+1)+")");
	      flag = false;
	    } else if (tmp.value[k] == alarm.value[k])
	      k++;
	    else
	      break;
	  }
/*	  if (!hyoka(alarm, tmp))
	    alarm = tmp;
	    flag = false;
*/
	}
    }
  }
  void change15() {
    int n = alarm.point.size();
    for (int i = 0; (i < n-2); i++)
      for (int j = i+2; j < n; j++) {
	Alarm tmp = new Alarm(alarm);
	tmp.reverse(i+1, j);
	tmp.path();
	int k = 0;
	while(k < 8) {
	  if (tmp.value[k] < alarm.value[k]) {
	    alarm = tmp;
	    System.out.println("CHANGED-"+k+"!!! ("+i+","+(i+1)+") <-> ("+j+","+(j+1)+")");
	    break;
	  } else if (tmp.value[k] == alarm.value[k])
	    k++;
	  else
	    break;
	}
    }
  }

  void change2() {
    int p = 2;
    int n = alarm.point.size();
    boolean flag = false;

    while(!flag) {
      flag = true;
      for (int i = 0; ((i < (n-2)) && flag); i++) {
	for (int j = p; (j < n) && flag; j++) {
	  Alarm tmp = new Alarm(alarm);
	  tmp.reverse(p+1,j);
	  tmp.path();
	  int k = 0;
	  while((k < 8) && flag) {
	    if (tmp.value[k] < alarm.value[k]) {
	      alarm = tmp;
	      System.out.println("CHANGED2-"+k+"!!! ("+i+","+(i+1)+") <-> ("+j+","+(j+1)+")");
	      p = i;
	      flag = false;
	    } else if (tmp.value[k] == alarm.value[k])
	      k++;
	    else
	      break;
	  }
	}
	if (flag)
	  if ((p-i) == 2)
	    p++;	
      }
    }
  }

  public static void main(String args[]) {
    Tsp13 route = new Tsp13();

    int count = args.length;

    if (count != 0) {
      if ("-p".equals(args[0])) {
	if (count == 1)
	  inputPoint();
	else if (count == 2)
	  inputPoint(args[1]);
	else {
	  System.err.println("Argument error!");
	  System.exit(1);
	}
      } else if ("-w".equals(args[0])) {
	if (count == 1)
	  inputWall();
	else if (count == 2)
	  inputWall(alarm.wall_h, args[1]);
	else if (count == 3) {
	  inputWall(alarm.wall_h, args[1]);
	  inputWall(alarm.wall_l, args[2]);
	} else {
	  System.err.println("Argument error!");
	  System.exit(1);
	}
      } else if ("-k".equals(args[0])) {
	if (count == 1) {
	  inputPoint();
	  inputWall();
	} else if (count == 4) {
	  inputPoint(args[1]);
	  inputWall(alarm.wall_h, args[2]);
	  inputWall(alarm.wall_l, args[3]);
	} else {
	  System.err.println("Argument error!");
	  System.exit(1);
	}
      } else {
	System.err.println("Argument error!");
	System.exit(1);
      }
    }
    route.begin();
  }

	//{{DECLARE_CONTROLS
	//}}
	//{{DECLARE_MENUS
	//}}
}
