class IntercalaryYear {
  public static void main(String args[]) {
    int i;

    for (i = 1900; i <= 2104; i++) {
      if (!(
        ((i % 4) != 0) // (1)
        ||
        ( ((i % 100) == 0) && ((i % 400) != 0) ) // (2)
        )){
        System.out.print(i + " ");
      }
    }
    System.out.println();
  }
}
