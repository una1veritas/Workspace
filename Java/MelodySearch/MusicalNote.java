


//MusicalNote.java
//SMFScore

//Created by Sin on 06/03/06.
//Copyright 2006 __MyCompanyName__. All rights reserved.


public class MusicalNote {
	int number, noteOn;
	int duration, velocity, channel;
	int tempo = 120;
	int no;
	public MusicalNote(int s, int n) {
		number = n;
		noteOn = s;

		duration = -1;
		velocity = -1;
		channel = -1;
		no=0;
	}
	
	public MusicalNote(int c,int s,int n){
		channel=c;
		noteOn=s;
		number=n;
		velocity=-1;
		duration=-1;
		//tempo=-1;
		no=0;
	}
	public MusicalNote(int c, int s, int n ,int v ,int t) {
		channel = c;
		noteOn = s;
		number = n;

		velocity=v;
		duration = -1;

		tempo=t;
		no=0;
	}

	public void setDuration(int t){
		duration=t-noteOn;
	}

	public void setNo(int n){
		no=n;
	}
	public int getNo(){
		return no;
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
		return "(" + channel +", " + noteOn + ", " + number + ", "+duration+", "+velocity +" ,"+tempo+/* " > "+off+ */") ";
	}

}
