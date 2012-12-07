class Function {
    static int target;
    
    public static void main(String args[]) {
	target = 6;
	System.out.println(factorial());
    }

  static int factorial (int n) {
    int i, result = 1;
    for (i = 1; i <= n; i++) {
      result = result * i;
    }
    return result;
  }

    static int factorial() {
	int i, result = 1;
	for (i = 1; i <= target; i++) {
	    result = result * i;
	}
	return result;
    }

}

