class Messenger extends Hello {
  String sender;
  
  public Messenger(String to, String from) {
    super(to);
    sender = from;
  }
  
  String sayHello() {
    return super.sayHello() + " from " + sender + ".";
  }
}
