class Messenger extends Hello {
  String sender;
  
  public Messenger(String to, String from) {
    super(to);
    sender = from;
  }
  
  String sayHello() {
    return super.sayHello() + " from " + sender + ".";
  }

  public String toString() {
    return super.toString() + " " + sender;
  }

  public static void main(String args[]) {
    Messenger m = new Messenger("receiver", "sender");
    
    System.out.println(m.toString());
  }
}
