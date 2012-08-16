package xml;

public final class UDOM{  
  public static String escapeSystemQuot(String string){
    if (string.indexOf('"') == -1) {
      return (string);
    }
    StringBuffer buffer = new StringBuffer();
    int size = string.length();
    for (int i = 0; i < size; i++) {
      char c = string.charAt(i);
      if (c == '"') {
	buffer.append("&quot;");
      } else {
	buffer.append(c);
      }
    }
    return (new String(buffer));
  }

  public static String escapeEntityQuot(String string) {
    if (string.indexOf('%') == -1 &&
	string.indexOf('&') == -1 &&
	string.indexOf('"') == -1) {

      return (string);
    }
    StringBuffer buffer = new StringBuffer();
    int size = string.length();
    for (int i = 0; i < size; i++) {
      char c = string.charAt(i);
      if (c == '%') {
	buffer.append("&---;");
      } else if (c == '&') {
	buffer.append("&amp;");
      } else if (c == '"') {
	buffer.append("&quot;");
      } else {
	buffer.append(c);
      }
    }
    return (new String(buffer));
  }

  public static String escapeAttrQuot(String string){
    if(string.indexOf('<') == -1 &&
       string.indexOf('&') == -1 &&
       string.indexOf('"') == -1)
      return string;

    StringBuffer buffer = new StringBuffer();
    int size = string.length();
    for(int i = 0; i < size; i++){
      char c = string.charAt(i);
      if(c == '<')
	buffer.append("&lt;");
      else if(c == '&')
	buffer.append("&amp;");
      else if(c == '"')
	buffer.append("&quot;");
      else
	buffer.append(c);
    }
    return(new String(buffer));
  }
  
  public static String escapeCharData(String string){
    if(string.indexOf('<') == -1 &&
       string.indexOf('&') == -1 &&
       string.indexOf("]]>") == -1)
      return string;

    StringBuffer buffer = new StringBuffer();
    int nBrackets = 0;
    int size = string.length();
    for(int i = 0; i < size; i++){
      char c = string.charAt(i);
      if(c == '<')
	buffer.append("&lt;");
      else if(c == '&')
	buffer.append("&amp;");
      else if(c == '>' && nBrackets >= 2)
	buffer.append("&gt;");
      else
	buffer.append(c);
      
      if(c == ']')
	nBrackets++;
      else
	nBrackets = 0;
    }
    return(new String(buffer));
  }
}
