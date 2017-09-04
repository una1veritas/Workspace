class Hello {
   String name;

   Hello(String who) {
      name = who;
   }

   String sayHello() {
      return "Hello, " + name + "!";
   }

   public static void main(String args[]) {
      Hello h;
      h = new Hello("あなたの名前をローマ字で");

      System.out.println(h.sayHello());
   }
}
