class TestContinue {
  public static void main(String args[]) {
    int i, n;
    
    n = 5;
    for (i = 0; i < n; i++) {
      // A
      System.out.println("A !");
      if (i == 3) continue;
      // B
      System.out.println("B ?");
    }
  }
}
