package syllabusNew;
import org.xml.sax.SAXParseException;
import org.xml.sax.ErrorHandler;

public class SimpleErrorHandler implements ErrorHandler {
    public void error(SAXParseException e) {
	System.out.print("[Error] ");
	System.out.println(e.getMessage());
    }

    public void fatalError(SAXParseException e) {
	System.out.print("[Fatal Error] ");
	System.out.println(e.getMessage());
    }

    public void warning(SAXParseException e) {
	System.out.print("[Warning] ");
	System.out.println(e.getMessage());
    }
}
