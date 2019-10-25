//
//  MusicalNote.java
//  SMFScore
//
//  Created by Sin on 06/03/06.
//  Copyright 2006 __MyCompanyName__. All rights reserved.
//

public class MusicalNote {
	int number, noteOn;
	int duration, velocity, channel;
	
	public MusicalNote(int s, int n) {
		number = n;
		noteOn = s;
		
		duration = -1;
		velocity = -1;
		channel = -1;
	}
	
	public MusicalNote(int c, int s, int n) {
		channel = c;
		noteOn = s;
		number = n;
		
		duration = -1;
		velocity = -1;
	}
	
	MusicalNote setValues(int c, int s, int n) {
		channel = c;
		noteOn = s;
		number = n;

		duration = -1;
		velocity = -1;

		return this;
	}
	
	public boolean isEndOfTrack() {
		return (channel == -1) && (noteOn == -1) && (number == -1);
	}
	
	public String toString() {
		return "(" + channel +": " + noteOn + ", " + number + /* " > "+off+ */") ";
	}

}
