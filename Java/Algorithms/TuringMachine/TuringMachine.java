import java.io.*;
import java.util.*;

class TuringMachine {

    public static void main(String [] args) 
	throws Exception {

	TreeSet states = new TreeSet();
	TreeSet alphabet = new TreeSet();
	TreeSet workAlphabet = new TreeSet();
	TreeSet finalStates = new TreeSet();
	int workTapes = 1;

	String temp[];

	String buf;
	BufferedReader cin = new BufferedReader(new InputStreamReader(System.in));
	final int 
	    None = 0,
	    States = 1,
	    FinalStates = 2, 
	    Alphabet = 3,
	    WorkTapeAlphabet = 4,
	    WorkTapeNumber = 5,
	    Transitions = 6;

	int readingContext = None;
	    

	alphabet.add(new Character('^'));
	alphabet.add(new Character('$'));
	workAlphabet.add(new Character('$'));
	workAlphabet.add(new Character('B'));

	while ( (buf = cin.readLine()) != null) {
	    if (buf.startsWith("//") || buf.equals("")) {
		continue;
	    }
	    if ( buf.toLowerCase().startsWith("alphabet") ) {
		readingContext = Alphabet;
		System.out.println("Reading Input-Tape Alphabet...");
		continue;
	    } else if ( buf.toLowerCase().startsWith("worktapealphabet") ) {
		readingContext = WorkTapeAlphabet;
		System.out.println("Reading Work-Tape Alphabet...");
		continue;
	    } else if ( buf.toLowerCase().startsWith("state") ) {
		readingContext = States;
		System.out.println("Reading States...");
		continue;
	    } else if ( buf.toLowerCase().startsWith("finalstate") ) {
		readingContext = FinalStates;
		System.out.println("Reading Final States...");
		continue;
	    } else if ( buf.toLowerCase().startsWith("numberofworktape") ) {
		readingContext = WorkTapeNumber;
		System.out.println("Reading The number of Worktapes...");
		continue;
	    } else if ( buf.toLowerCase().startsWith("transition") ) {
		readingContext = Transitions;
		System.out.println("Reading Transitions...");
		continue;
	    }

	    switch (readingContext) {
	    case Alphabet:
		for (int i = 0; i < buf.length(); i++) {
		    alphabet.add(new Character(buf.charAt(i)));
		}
		break;
	    case WorkTapeAlphabet:
		for (int i = 0; i < buf.length(); i++) {
		    workAlphabet.add(new Character(buf.charAt(i)));
		}
		break;
	    case States:
		temp = buf.split("\\s*$");
		temp = temp[0].split("\\s*,\\s*");
		for (int i = 0; i < temp.length; i++) {
		    states.add(temp[i]);
		}
		states.remove("");
		break;
	    case FinalStates:
		temp = buf.split("\\s*$");
		temp = temp[0].split("\\s*,\\s*");
		for (int i = 0; i < temp.length; i++) {
		    if (states.contains(temp[i])) {
			finalStates.add(temp[i]);
		    } else {
			System.out.println("Error: Found a final state " + temp[i] + " which is not a state!!");
		    }
		}
		break;
	    case WorkTapeNumber:
		workTapes = Integer.parseInt(buf);
		break;
	    default:
		System.out.println(buf);
	    }

	}


	System.out.print("\nAlphabet: ");
	for(Iterator i = alphabet.iterator(); i.hasNext(); ) {
	    System.out.print(i.next() + ", ");
	}
	System.out.print("\nWorktape Alphabet: ");
	for(Iterator i = workAlphabet.iterator(); i.hasNext(); ) {
	    System.out.print(i.next() + ", ");
	}
	System.out.print("\nStates: ");
	for(Iterator i = states.iterator(); i.hasNext(); ) {
	    System.out.print(i.next() + ", ");
	}
	System.out.print("\nFinal States: ");
	for(Iterator i = finalStates.iterator(); i.hasNext(); ) {
	    System.out.print(i.next() + ", ");
	}
	System.out.println("\nThe number of Worktapes: " + workTapes);
	System.out.println();

	return;
    }

}
