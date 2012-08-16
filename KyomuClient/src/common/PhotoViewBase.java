package common;
import clients.*;
import java.util.*;
//import java.net.*;
import javax.swing.*;
import javax.swing.event.*;
import java.awt.*;
//import java.awt.event.*;

public class PhotoViewBase extends JPanel {
  private CommonInfo commonInfo;
  private TabbedPaneBase tabbedPane;
  public  JDesktopPane desktop = new JDesktopPane();
  private HashMap<String, Object> studentMap = new HashMap<String, Object>();
  private ImageFrameListener frameListener = new ImageFrameListener();
  private Integer firstLayer = new Integer(1);
  private int count = 0;

  public PhotoViewBase(String serviceName,
		       String nodePath, String panelID,
		       CommonInfo commonInfo, 
		       TabbedPaneBase tabbedPane,
		       DataPanelBase dataPanel) {
    this.commonInfo = commonInfo;
    this.tabbedPane = tabbedPane;
    setMainComponent();      
  }
    
  public void setMainComponent() { 
    setLayout(new BorderLayout());
    add("Center", desktop);    
  }

  public void pageOpened() { 
    ArrayList<String> studentList = tabbedPane.getStudentListToPass();
    for (int i = 0; i < studentList.size(); i++) {
      String line = studentList.get(i);
      String[] tokens = line.split("\\|");
      String dname = tokens[0];
      String gname = tokens[1];
      String code = tokens[2];
      String name = tokens[3];
      if (!studentMap.containsKey(code)) {
	addStudentPhoto(i, code, name, dname);
      }
    }
    count++;
  }

  public void addStudentPhoto(int n, String code, String name, String dname) {
    ImageIcon icon = commonInfo.commonInfoMethods.getStudentPhoto(code);
    if (icon != null) {
      int w = icon.getIconWidth();
      int h = icon.getIconHeight();
      if (w < 10) {
	w = 120;
	h = 200;
      }
      addInternalFrame(icon, code, name, dname,  w, h, n);
    }
  }

  private void addInternalFrame(ImageIcon icon, 
				String code, String name, String dname,
				int width, int height, int cnt) {
    JInternalFrame iframe = new JInternalFrame(name);
    iframe.setClosable(true);
    iframe.setIconifiable(true);
    iframe.setResizable(true);
    iframe.setBounds(count*5 + 60*cnt, count*5 + 30*cnt, width + 13, height + 53);
    JPanel panel = new JPanel();
    panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
    panel.add(new ImageScroller(icon));
    JLabel label = new JLabel();
    label.setFont(new Font("DialogInput", Font.PLAIN, 11));  
    label.setText(code + ":" + dname);
    panel.add(label);
    iframe.setContentPane(panel);
    iframe.addInternalFrameListener(frameListener);
    desktop.add(iframe, firstLayer);
    iframe.show();
    studentMap.put(code, iframe);
  }

  public void clearPhotoView() {
    ArrayList<String> list = new ArrayList<String>(studentMap.keySet());
    for (String code : list) {
      JInternalFrame iframe = (JInternalFrame)studentMap.get(code);
      iframe.dispose();
    }
  }

  class ImageFrameListener extends InternalFrameAdapter {
    public void internalFrameClosed(InternalFrameEvent e) {
      JInternalFrame iframe = (JInternalFrame) e.getSource();
//      int w = iframe.getWidth();
//      int h = iframe.getHeight();
//      String title = iframe.getTitle();
      ArrayList<String> list = new ArrayList<String>(studentMap.keySet());
      for (int i = 0; i < list.size(); i++) {
	String code = (String) list.get(i);
	if (iframe == studentMap.get(code)) {
	  studentMap.remove(code);
	  return;
	}
      }
    }
  }

  class ImageScroller extends JScrollPane {   

    ImageScroller(ImageIcon icon) {
      super();
      JPanel p = new JPanel();
      p.setBackground(Color.white);
      p.setLayout(new BorderLayout());      
      p.add(new JLabel(icon), BorderLayout.CENTER);      
      getViewport().add(p);
      getHorizontalScrollBar().setUnitIncrement(10);
      getVerticalScrollBar().setUnitIncrement(10);
    }
    
    public Dimension getMinimumSize() {
      return new Dimension(25, 25);
    }    
  }  
}
