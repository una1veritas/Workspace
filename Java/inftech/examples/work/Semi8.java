class Semi8 {
  public static void main(String args[]) {
    int i;

    for (i = 1900; i <= 2104; i++) {
      if (((i % 4) == 0) && ((i % 100) != 100) || ((i % 400) == 0)) {
        System.out.print(i + " ");
      }
    }
    System.out.println();
  }
}
