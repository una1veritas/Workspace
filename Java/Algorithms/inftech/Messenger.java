class Messenger extends Hello {
    String sender;

    public Messenger(String sndr, String recp) {
	super(recp);
	sender = sndr;
    }

    public String sayHello() {
	return super.sayHello() + "I am " + sender + ".";
    }
}
