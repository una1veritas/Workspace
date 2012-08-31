class Divisors {

    public static void main(String[] args) {
        int n = 852;
        int dv;
        
        if (n >= 1) {
            System.out.print(1+", ");
        } else {
            return;
        }
        dv = 2;
        while (n >= dv) {
            if ( n % dv == 0) {
                n = n / dv;
                System.out.print(dv+", ");
            }
            dv++;
        }
        System.out.println();

    }

}
