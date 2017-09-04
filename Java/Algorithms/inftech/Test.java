class Test {
    public static void main(String args[]) {
	int s, e;

	s = Integer.parseInt(args[1]);
	e = Integer.parseInt(args[2]);
	System.out.println(args[0].substring(s,e));
    }
}
