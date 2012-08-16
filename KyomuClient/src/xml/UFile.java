package xml;
//import java.io.*;
//import java.util.*;
import javax.swing.tree.*;
import org.w3c.dom.*;

public final class UFile {
  
  public static String makenullxmltext(String kyojuID, String kamokuID) {
    String xmltext = "<講義 科目コード=\"" + kamokuID +
      "\" 教官コード=\"" + kyojuID + "\">\n" +
        "\t<講義内容></講義内容>\n" +
          "\t<位置付け></位置付け>\n" +
            "\t<講義項目></講義項目>\n" +
              "\t<進め方></進め方>\n" +
                "\t<評価方法></評価方法>\n" +
                  "\t<教科書></教科書>\n" +
                    "\t<参考書></参考書>\n" +
		      "\t<備考></備考>\n" +
			"\t<キーワード></キーワード>\n" +
			  "</講義>\n";
    return xmltext;
  }

  public static String makenulljabeexmltext(String kyojuID, String kamokuID) {
    String xmltext = "<講義 科目コード=\"" + kamokuID +
      "\" 教官コード=\"" + kyojuID + "\">\n" +
        "\t<授業の概要></授業の概要>\n" +
          "\t<カリキュラムにおけるこの授業の位置付け></カリキュラムにおけるこの授業の位置付け>\n" +
            "\t<授業項目></授業項目>\n" +
              "\t<授業の進め方></授業の進め方>\n" +
                "\t<授業の達成目標></授業の達成目標>\n" +
		  "\t<成績評価の基準および評価方法></成績評価の基準および評価方法>\n" +
		    "\t<キーワード></キーワード>\n" +
		      "\t<教科書></教科書>\n" +
			"\t<参考書></参考書>\n" +
			  "\t<備考></備考>\n" +
			    "</講義>\n";
    return xmltext;
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
      /*
      NamedNodeMap attrs = element_.getAttributes();
      int size = attrs.getLength();
      for (int i = 0; i < size; i++) {
        Attr attr = (Attr)attrs.item(i);
        String name = attr.getName();
        String value = attr.getValue();
        element.setAttribute(name, value);
      }
      */
      
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

  public static DefaultMutableTreeNode makeJTTree(Node node) {
    return (_makeJTNode(node));
  }
  
  public static boolean isDOMLeafNode(Element element) {
    String tag = element.getTagName();

    if (tag.equals("P") || tag.equals("LI")) {
      return (true);
    }
    return (false);
  }

  private static DefaultMutableTreeNode _makeJTNode(Node node) {
    // DOM の Node [Text]
    if (node instanceof Text) {
      return (null);
    }
    
    // DOM の Node [Element]
    if (node instanceof Element) {
      Element element = (Element)node;
      String tagname = element.getTagName();

      if (isDOMLeafNode(element)) {

        // 葉ノード
        String text = getString(element);
        DefaultMutableTreeNode leafNode =
          new DefaultMutableTreeNode(new JTElement(tagname, text, true));
        return (leafNode);
        
      } else {
        
        // 節ノード
        JTElement jtelement = new JTElement(tagname, "", false);
        DefaultMutableTreeNode elementNode =
          new DefaultMutableTreeNode(jtelement);
      
        // 属性
        NamedNodeMap attrs = element.getAttributes();
        int size = attrs.getLength();
        for (int i = 0; i < size; i++) {
          Attr attr = (Attr)attrs.item(i);
          String name = attr.getName();
          String value = attr.getValue();
          DefaultMutableTreeNode attrNode =
            new DefaultMutableTreeNode(new JTAttr(name, value));
          elementNode.add(attrNode);
          //elementNode.insert(attrNode, elementNode.getChildCount());
        }
      
        // 子ノード
        NodeList children = element.getChildNodes();
        int nchildren = children.getLength();
        for (int i = 0; i < nchildren; i++) {
          if (_makeJTNode(children.item(i)) != null)
            elementNode.add(_makeJTNode(children.item(i)));
          //elementNode.insert(_makeJTNode(children.item(i)),
          //elementNode.getChildCount());
        }
        return (elementNode);
      }
    } else {
      throw (new InternalError());
    }
  }

  private static String getString(Node node) {
    StringBuffer buffer = new StringBuffer();
    NodeList children = node.getChildNodes();
    int nChildren = children.getLength();
    for (int i = 0; i < nChildren; i++) {
      Node child = children.item(i);
      if (child instanceof Text) {
        Text text = (Text)child;
        buffer.append(text.getData());
      }
    }
    return (new String(buffer));
  }
}
