class Function {
  static int target;
  
  public static void main(String args[]) {
    target = 6;
    System.out.println(factorial());
  }

  static long factorial(int n) {
    int i;
    
    long result = 1;
    for (i = 1; i <= n; i++) {
      result = result * i;
    }
    return result;
  }

  static long factorial( ) {
    return factorial(target);
  }
}
