class Hello {
  String name;

  Hello(String who) {
    name = who;
  }

  void sayHello() {
    System.out.println("Hello, " + name + "!");
  }

  public static void main(String args[]) {
    Hello deputy;

    deputy = new Hello("���ʤ���̾������޻���");
    deputy.sayHello();
  }
}
