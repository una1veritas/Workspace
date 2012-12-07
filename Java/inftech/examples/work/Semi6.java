class TestFor1 {
  public static void main(String args[]) {
    int i;

    for (i = 1; i <= 100; i++) {
      System.out.print(i + " ");
    }
    System.out.println();
  }
}

class TestFor2 {
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
