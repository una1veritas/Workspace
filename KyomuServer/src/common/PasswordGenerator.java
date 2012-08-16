package common;
import java.util.*;
import javax.swing.*;

public class PasswordGenerator {
  private Random rand;

  static char charset[] = {'a','b','c','d','e','f','g','h','i','j','k','m','n',
			   'p','q','r','s','t','u','v','w','x','y','z',
			   '1','2','3','4','5','6','7','8','9',
			   'A','B','C','D','E','F','G','H','I','J','K','L',
			   'M','P','Q','R','S','T','U','V','W','X','Y','Z'};

  public PasswordGenerator()  {
    rand = new Random(System.currentTimeMillis());
  }

  public String getPassword() {
    String str = "";
    int len = charset.length;
    for (int i = 0; i < 8; i++) {
      str = str + charset[rand.nextInt(len)];
    }
    return str;
  }

  public String getSalt() {
    String str = "";
    int len = charset.length;
    for (int i = 0; i < 2; i++) {
      str = str + charset[rand.nextInt(len)];
    }
    return str;
  }
}
