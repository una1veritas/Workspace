import java.applet.*;
import java.awt.*;
import java.awt.event.*;
/* 
  <applet code="LoginPrompt" width=400 height=200>
  </applet>
 */
public class LoginPrompt extends Applet {
   TextField tf1, tf2;

   public void init() {
       setLayout(null);
       tf1 = new TextField("guest", 20);
       tf1.setBounds(10, 10, 200, 16);
       add(tf1);
       tf2 = new TextField(12);
       tf2.setEchoChar('*');
       tf2.setBounds(10, 50, 120, 18);
       add(tf2);
   }
}
