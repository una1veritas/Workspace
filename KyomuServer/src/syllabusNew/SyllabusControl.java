package syllabusNew;
import java.io.*;
import java.net.*;
import java.util.*;
import javax.xml.parsers.*;
import org.xml.sax.*;
import org.w3c.dom.*;

public class SyllabusControl {
  private String dir = "/home/maginu/KYOMU-INFO/SYLLABUS/";

  private int prevSchoolYear;
  private int thisSchoolYear;
  private int nextSchoolYear;

  private Document domtreePrevYear;
  private Document domtreeThisYear;
  private Document domtreeNextYear;

  private DocumentBuilder builder;

  public SyllabusControl(int thisSchoolYear) {
    this.prevSchoolYear = thisSchoolYear - 1;
    this.thisSchoolYear = thisSchoolYear;
    this.nextSchoolYear = thisSchoolYear + 1;

    URL urlPrevYear = null;
    URL urlThisYear = null;
    URL urlNextYear = null;

    UFile.createSyllabusFile(prevSchoolYear);
    UFile.createSyllabusFile(thisSchoolYear);
    UFile.createSyllabusFile(nextSchoolYear);
    
    try {
      urlPrevYear = new File(dir + prevSchoolYear + ".xml").toURL();
      urlThisYear = new File(dir + thisSchoolYear + ".xml").toURL();
      urlNextYear = new File(dir + nextSchoolYear + ".xml").toURL();
    } catch (MalformedURLException e) {
      e.printStackTrace();
    }

    try {
      DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
      builder = factory.newDocumentBuilder();
      builder.setErrorHandler(new SimpleErrorHandler());
      domtreePrevYear = builder.parse(urlPrevYear.toExternalForm());
      domtreeThisYear = builder.parse(urlThisYear.toExternalForm());
      domtreeNextYear = builder.parse(urlNextYear.toExternalForm());      
    } catch (ParserConfigurationException e) {
      e.printStackTrace();
    } catch (SAXException e) {
      e.printStackTrace();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
  
  private Element getElement(String yearText, 
			     String teacherCode, String subjectCode) {
    int year = Integer.parseInt(yearText);
    Element root = null;
    if (year == prevSchoolYear) {
      root = domtreePrevYear.getDocumentElement();
    } else if (year == thisSchoolYear) {
      root = domtreeThisYear.getDocumentElement();
    } else if (year == nextSchoolYear) {
      root = domtreeNextYear.getDocumentElement();
    } else {
      return null;
    }
    
    NodeList children = root.getElementsByTagName("講義");
    int nChildren = children.getLength();    
    for (int i = 0; i < nChildren; i++) {
      Node child = children.item(i);      
      if (child instanceof Element) {
        Element element = (Element)child;
        String teacher = getAttribute(element, "教官コード");
        String subject = getAttribute(element, "科目コード");        
        if (teacher.equals(teacherCode) && subject.equals(subjectCode)) {
          return element;
        }
      }
    }
    return null;
  }  
  
  public String getElementHtml(String yearText, 
			       String teacherCode, String subjectCode) {
    String htmltext = "";
    Element element = getElement(yearText, teacherCode, subjectCode);
    if (element == null) {
      return null;
    } else {
      HTMLMakerDOM maker = new HTMLMakerDOM();
      UDOMVisitor.traverse(element, maker);
      htmltext = maker.getText();
      return htmltext.trim();
    }
  }
    
  public String getElementXml(String yearText, 
			      String teacherCode, String subjectCode) {
    String xmltext = "";
    Element element = getElement(yearText, teacherCode, subjectCode);
    if (element == null) {
      return null;
    } else {      
      XMLMaker maker = new XMLMaker();
      UDOMVisitor.traverse(element, maker);
      xmltext = maker.getText();
      return xmltext.trim();
    }
  }

  public void updateElement(String yearText, String xmltext) {
    Document domtree;
    String dir2 = dir + yearText + "/";
    int year = Integer.parseInt(yearText);

    if (year == prevSchoolYear) {
      domtree = domtreePrevYear;
    } else if (year == thisSchoolYear) {
      domtree = domtreeThisYear;
    } else if (year == nextSchoolYear) {
      domtree = domtreeNextYear;
    } else {
      return;
    }

    Element newelement = null;
    boolean renewal = false;    
    try {
      StringReader reader = new StringReader(xmltext);
      InputSource source = new InputSource(reader);
      Document document = builder.parse(source);
      
      Element element = document.getDocumentElement();
      Node node = UFile.makeElement(element, domtree);
      if (node instanceof Element) {
        newelement = (Element) node;
      }
      
    } catch(SAXException e) {
      System.out.println("SAXException");
    } catch(IOException e) {
      System.out.println("IOException");
    }

    if (newelement == null) {
      System.out.println("Error: newelement can't create!!");
    }
    
    String teacher = getAttribute(newelement, "教官コード");
    String subject = getAttribute(newelement, "科目コード");

    String elementfilename = teacher + "_" + subject + ".xml";
    try {
      FileWriter writer = new FileWriter( dir2 + elementfilename);
      writer.write(xmltext.trim());
      writer.close();
    } catch (Exception ex) {
      ex.printStackTrace();
    }

    Element rootelement = domtree.getDocumentElement();
    
    NodeList children = rootelement.getElementsByTagName("講義");
    int nChildren = children.getLength();
    
    for (int i = 0; i < nChildren; i++) {
      Node child = children.item(i);      
      if (child instanceof Element) {
        Element element = (Element)child;
        
        String teacher2 = getAttribute(element, "教官コード");
        String subject2 = getAttribute(element, "科目コード");

        if (teacher2.equals(teacher) && subject2.equals(subject)) {
          rootelement.replaceChild(newelement, element);
          renewal = true;
	  break;
        }
      }
    }

    if (renewal == false) {     
      rootelement.appendChild(newelement);  
      // 新しい教授要目の要素を新たに追加する
    }
  }

  private static String getAttribute(Element element, String attrName) {
    Attr attr = element.getAttributeNode(attrName);
    if (attr == null) {
      return (null);
    }
    return (attr.getValue());
  }
}
