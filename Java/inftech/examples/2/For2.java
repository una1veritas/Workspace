class For2 {
  public static void main(String args[]) {
    int i;

    for (i = 1; i <= 100; i++) {
      if ((i % 17) == 0) {
        System.out.print(i + " ");
      }
    }
    System.out.println();
  }
}
