class Hello {
  String name;

  Hello(String who) {
    name = who;
  }

  String sayHello() {
    return "Hello, " + name + "!";
  }

  public static void main(String args[]) {
    Hello messengers[] = new Hello[args.length];

    for (int i = 0; i < args.length; i++) {
      messengers[i] = new Hello(args[i]);
      System.out.println(messengers[i].sayHello());
    }
  }
}
