import java.applet.*;
import java.awt.*;
import java.awt.event.*;
import java.net.*;

/*
  <applet code="WebSearch" width=400 height=400>
  </applet>
  */

public class WebSearch extends Applet implements ActionListener {
  Choice engine;
  TextField query;
  TextArea result;
  Button bttn;

  public void init() {
    engine = new Choice();
    engine.addItem("Yahoo!");
    engine.addItem("Altavista");
    engine.addItem("HotBot");
    add(engine);

    query = new TextField(20);
    add(query);

    bttn = new Button("Search");
    bttn.addActionListener(this);
    add(bttn);

    result = new TextArea(50, 300);
    add(result);
  }

  public void actionPerformed(ActionEvent event) {
    try {
      URL url = new URL("");
    //    result.setText();
    } catch (Exception e) {
      System.out.println(e);
    }
  }
}
