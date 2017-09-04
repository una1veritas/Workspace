import java.applet.*;
import java.awt.*;
import java.awt.event.*;
/* 
  <applet code="CheckboxGroups" width=400 height=80>
  </applet>
 */
public class CheckboxGroups extends Applet {
   Label label;

   public void init() {
      CheckboxGroup grp = new CheckboxGroup();
      Checkbox cbox1 = new Checkbox("Sedan", grp, true);
      add(cbox1);
      Checkbox cbox2 = new Checkbox("Wagon", grp, false);
      add(cbox2);
      Checkbox cbox3 = new Checkbox("Coupe", grp, false);
      add(cbox3);
      Checkbox cbox4 = new Checkbox("Compact", grp, false);
      add(cbox4);

      label = new Label("                     ");
      add(label);
   }
}
