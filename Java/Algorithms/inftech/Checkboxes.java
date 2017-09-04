import java.applet.*;
import java.awt.*;
import java.awt.event.*;
/*
  <applet code="Checkboxes" width=400 height=80>
  </applet>
*/

public class Checkboxes extends Applet implements ActionListener {
    Label label;
    Button bttn;
    Checkbox chkbox1, chkbox2, chkbox3;
    
    public void init() {
	chkbox1 = new Checkbox("Apple");
	add(chkbox1);
	chkbox2 = new Checkbox("Grapefruit");
	add(chkbox2);
	chkbox3 = new Checkbox("Orange");
	add(chkbox3);
	
	bttn = new Button("Summary");
	bttn.addActionListener(this);
	add(bttn);
	label = new Label("**** No fruits ****");
	add(label);
    }
    
    public void actionPerformed(ActionEvent event) {
	String s = "";
	if (chkbox1.getState()) {
	    s = s + chkbox1.getLabel();
	}
	if (chkbox2.getState()) {
	    s = s + " " + chkbox2.getLabel();
	}
	if (chkbox3.getState()) {
	    s = s + " " + chkbox3.getLabel();
	}
	label.setText(s);
    }
}
