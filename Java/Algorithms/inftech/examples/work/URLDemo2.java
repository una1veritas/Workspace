import java.io.*;
import java.net.*;

class URLDemo2 {
  public static void main(String args[]) {
    try {
//hotbotOK. URL url = new URL("http://www.hotbot.com/default.asp?prov=Google&query=" + args[0] + "&ps=&loc=searchbox&tab=web");
//yahoo.jp‚Î‚¯‚é. 
URL url = new URL("http://search.yahoo.co.jp/bin/search?p=" + args[0] + "&submit=%B8%A1%BA%F7");
//yahoo.comOK. URL url = new URL("http://search.yahoo.com/bin/search?p=" + args[0]);
//google.jp‚¯‚ç‚ê‚é.      URL url = new URL("http://www.google.co.jp/search?q=" + args[0] + "&ie=UTF-8&oe=UTF-8&hl=ja&btnG=Google+%E6%A4%9C%E7%B4%A2&lr=");
//goo‚Î‚¯‚é. URL url = new URL("http://search.goo.ne.jp/web.jsp?TAB=&MT=" + args[0] + "&web.x=0&web.y=0&web=%A5%A6%A5%A7%A5%D6");
//infoseek‚Î‚¯‚é. URL url = new URL("http://www.infoseek.co.jp/Titles?qt=" + args[0] + "&internet=%A5%A4%A5%F3%A5%BF%A1%BC%A5%CD%A5%C3%A5%C8&btnchk=1&lk=noframes&qp=0&nh=10&svx=100600");
//altaOK. URL url = new URL("http://www.altavista.com/web/results?q=" + args[0] + "&kgs=0&kls=0&avkw=xytx");
//google.com‚¯‚ç‚ê‚é. URL url = new URL("http://www.google.com/search?hl=en&ie=UTF-8&oe=UTF-8&q=" + args[0] + "&btnG=Google+Search");
//msn.jp‚Î‚¯‚é. URL url = new URL("http://search.msn.co.jp/results.asp?FORM=msnh&v=1&RS=CHECKED&CY=ja&cp=932&q=" + args[0]);
      InputStream in = url.openStream();
      InputStreamReader sr = new InputStreamReader(in);
      BufferedReader br = new BufferedReader(sr);

      String s;
      while ( (s = br.readLine()) != null )
        System.out.println(s);
      
      sr.close();
    } catch (Exception e) {
      System.out.println("Exception " + e + " occurred.");
    }
  }
}
