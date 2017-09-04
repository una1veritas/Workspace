class Semi7 {
  public static void main(String args[]) {
    int i;
    
    for (i = 1; i <= 100; i++) {
      if (((i % 3) == 0) && ((i % 2) != 0)) {
        System.out.print(i + " ");
      }
    }
    System.out.println();
  }
}
