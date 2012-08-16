package syllabusNew;
import java.io.*;
import javax.swing.tree.*;
import org.w3c.dom.*;

public final class UFile {
  
  public static void createSyllabusFile(int year) {
    String dir = "/home/maginu/KYOMU-INFO/SYLLABUS/";
    String fname = dir + year + ".xml";
    try {
      PrintWriter fout = 
        new PrintWriter( new BufferedWriter(new FileWriter(fname)) );
      fout.println("<?xml version=\"1.0\" encoding=\"EUC-JP\"?>");
      fout.println("<教授要目>");      

      File directory = new File(dir + year);      
      File[] list = directory.listFiles();
      String line;
      for (int i = 0; i < list.length; i++) { 
	BufferedReader fin =
	  new BufferedReader( new FileReader( list[i] ) );        
        while((line = fin.readLine()) != null) {
          if (line.startsWith("<?") != true)
            fout.println(line);
        }
        fin.close();
      }      
      fout.println("</教授要目>");
      fout.close();
    } catch (Exception ex) {
      ex.printStackTrace();
    }
  }

  public static Node makeElement(Node node, Document doc) {
    return (_makeElement(node, doc));
  }

  private static Node _makeElement(Node node, Document doc) {
    // DOM の Node [Text]
    if (node instanceof Text) {
      Text text_ = (Text)node;
      Text text = doc.createTextNode(text_.getData());
      return text;
    }
    
    // DOM の Node [Element]
    if (node instanceof Element) {
      Element element_ = (Element)node;
      Element element = doc.createElement(element_.getTagName());
      
      // 属性
      NamedNodeMap attrs = element_.getAttributes();
      int size = attrs.getLength();
      for (int i = 0; i < size; i++) {
        Attr attr = (Attr)attrs.item(i);
        String name = attr.getName();
        String value = attr.getValue();
        element.setAttribute(name, value);
      }
      
      // 子ノード
      NodeList children = element_.getChildNodes();
      int nchildren = children.getLength();
      for (int i = 0; i < nchildren; i++) {
        if (_makeElement(children.item(i), doc) != null)
          element.appendChild(_makeElement(children.item(i), doc));
      }
      return (element);
      
    } else {
      throw (new InternalError());
    }
  }
}
