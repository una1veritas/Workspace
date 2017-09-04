import java.applet.*;
import java.awt.*;

/*
  <applet code="Labels" width=200 height=120>
  </applet>
  */

public class Labels extends Applet {

  public void init() {
    Label label1 = new Label("Default alignment.");
    add(label1);
    Label label2 = new Label("A center aligned label.", Label.CENTER);
    add(label2);
    Label label3 = new Label("Aligned right:", Label.RIGHT);
    add(label3);
  }
}
    