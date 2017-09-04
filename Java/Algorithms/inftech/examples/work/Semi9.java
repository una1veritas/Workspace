class Semi9 {
  public static void main(String args[]) {
    System.out.println(factorial(5));
  }

  static long factorial(int n) {
    int i;

    long result = 1;
    for (i = 1; i <= n; i++) {
      result = result * i;
    }
    return result;
  }
}
