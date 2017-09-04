class ArrayTest {
  public static void main(String args[]) {
    String temp = "";
    int i;
    for (i = 0; i < args.length; i++) {
      System.out.println(args[i]);
      temp = temp + args[i];
    }
    System.out.println(temp);
  }
}
