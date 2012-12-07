import com.ibm.xml.parser.Parser;
import java.io.FileInputStream;
import java.io.PrintWriter;
import org.w3c.dom.Node;

public class PrintDOMTree {
    /**
     * このメソッドはDOM APIのみで書かれている。
     */
    public static void printTree(Node node, PrintWriter writer,
                                 int currIndent, int dx) {
        for (int i = 0;  i < currIndent;  i ++)
            writer.print(" ");
        switch (node.getNodeType()) {
          case Node.DOCUMENT_NODE:
          case Node.ELEMENT_NODE:
          case Node.TEXT_NODE:
          case Node.CDATA_SECTION_NODE:
            writer.println(node.getNodeName());
            break;
          case Node.PROCESSING_INSTRUCTION_NODE:
            writer.println("<?"+node.getNodeName()+"...?>");
            break;
          case Node.COMMENT_NODE:
            writer.println("<!--"+node.getNodeValue()+"-->");
            break;
          case Node.ENTITY_NODE:
            writer.println("ENTITY "+node.getNodeName());
            break;
          case Node.ENTITY_REFERENCE_NODE:
            writer.println("&"+node.getNodeName()+";");
            break;
          case Node.DOCUMENT_TYPE_NODE:
            writer.println("DOCTYPE "+node.getNodeName());
            break;
          default:
            writer.println("? "+node.getNodeName());
        }

        for (Node child = node.getFirstChild();
             child != null;
             child = child.getNextSibling()) {
            printTree(child, writer, currIndent+dx, dx);
        }
    }

    public static void main(String[] argv) {
        try {
            Parser parser = new Parser(argv[0]); // @XML4J
            PrintWriter pwriter = new PrintWriter(System.out);
            FileInputStream fis = new FileInputStream(argv[0]);
            printTree(parser.readStream(fis),    // @XML4J
                      pwriter, 0, 2);
            pwriter.flush();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

