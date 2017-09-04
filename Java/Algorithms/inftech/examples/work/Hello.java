class Hello {
  String name;

  Hello(String who) {
    name = who;
  }

  String sayHello() {
    return "Hello, " + name + "!";
  }

  public String toString() {
    return name;
  }

  public static void main(String args[]) {
    Hello h;
    h = new Hello("Morita");

    System.out.println(h.toString());
  }
}
