import java.applet.*;
import java.awt.*;
import java.awt.event.*;
/* 
  <applet code="TextFields" width=400 height=200>
  </applet>
 */
public class TextFields extends Applet {
   TextField tfield;
   TextArea  tarea;

   public void init() {
      tfield = new TextField(20);
      add(tfield);
      tarea = new TextArea(8,20);
      add(tarea);
   }
}
