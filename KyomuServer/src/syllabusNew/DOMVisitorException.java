package syllabusNew;
import java.io.PrintWriter;
import java.io.PrintStream;
import org.w3c.dom.*;

public class DOMVisitorException extends RuntimeException {
  private Exception cause_ = null;
  
  public DOMVisitorException(String message) {
    super(message);
  }
  
  public DOMVisitorException(Exception e) {
    super(e.getMessage());
    cause_ = e;
  }
  
  public DOMVisitorException(String message, Exception e) {
    super(message);
    cause_ = e;
  }
  
  public Exception getException() {
    if (cause_ != null) {
      return (cause_);
    } else {
      return (this);
    }
  }
  
  public Exception getCauseException() {
    return (cause_);
  }
  
  public void printStackTrace() {
    printStackTrace(new PrintWriter(System.err, true));
  }
  
  public void printStackTrace(PrintStream out) {
    printStackTrace(new PrintWriter(out));
  }
  
  public void printStackTrace(PrintWriter writer) {
    if (writer == null) {
      writer = new PrintWriter(System.err, true);
    }
    super.printStackTrace(writer);
    if (cause_ != null) {
      writer.println();
      writer.println("StackTrace of Original Exception:");
      cause_.printStackTrace(writer);
    }
  }
}
