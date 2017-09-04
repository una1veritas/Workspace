class cv_on_im {
  int id; 
  static int global = 2;

  cv_on_im(int n) {
    id = n;
  }

  static void incr() {
    global++;
  }
  
  public static void main(String args[]) {
    int i;
    cv_on_im hoge;

    hoge = new cv_on_im(1);

    for (i = 0; i < 10; i++) {
      System.out.println(hoge.test());
      cv_on_im2.incr2();
    }
  }

  String test() {
    global++;
    return id + " " + global;
  }
}

class cv_on_im2 {
  static int global = 1000;

  static void incr2() {
    global++;
  }
}