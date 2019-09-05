//
//  SMFScore.java
//  SMFScore
//
//  Created by ?? ?? on 06/03/06.
//  Copyright (c) 2006 __MyCompanyName__. All rights reserved.
//
import java.util.*;
import java.io.*;
import java.lang.*;

public class SMFScore {
	Vector score;
	int tracks;
	int division;
	int format;
	
	//
	private final static int defaultDivision = 480;
	//public static long totalTime;
	
	private static int parseVarLenInt(InputStream stream) throws IOException {
		int oneByte, value = 0; // for storing unsigned byte
		while ( (oneByte = stream.read()) != -1 ) {
			value = value << 7;
			value += oneByte & 0x7f;
			if ( (oneByte & 0x80) == 0 )
				break;			
		}
		if ( oneByte == -1 )
			return -1;
		return value;
	}
	
	
	public SMFScore(String arg) {
		int number, noteon;
		int ch = 1;
		StringTokenizer st = new StringTokenizer(arg);
		score = new Vector();
		format = 0;
		tracks = 1;
		division = 480; // Ticks Per Quoter Note //Integer.parseInt(st.nextToken());
		noteon = 0;
		number = Integer.parseInt(st.nextToken());
		score.add(new MusicalNote(ch, noteon, number));
		while (st.hasMoreTokens()) {
			noteon += (int) (480 * 4 / Double.parseDouble(st.nextToken()));
			number = Integer.parseInt(st.nextToken());
			score.add(new MusicalNote(ch, noteon, number));
		}
		return;
	}
	
	public SMFScore(InputStream bistr) throws IOException {
		//int format;
		
		int deltaTotal = 0;
		int deltaTime;
		int velocity;
		int len;
		
		int high4bits, low4bits;
		int oneByte;
		byte buf[] = {0,0,0,0};
		
		/*
		oneByte = bistr.read();
		high4bits = (oneByte & 0xf0) >> 4;
		low4bits = oneByte & 0x0f;
		 */
		bistr.read(buf);
		//if (high4bits == 4 && low4bits == 13) {
		if ( buf[0] == 'M' && buf[1] == 'T' && buf[2] == 'h' && buf[3] == 'd' ) {
			/*
			 for (int i = 0; i < 8; i++) {
				bistr.read();
			}*/
			bistr.skip(4);
			format = bistr.read() << 8;
			format += bistr.read();
			tracks = bistr.read() << 8;
			tracks += bistr.read();
			division = bistr.read() << 8;
			division += bistr.read();
			//System.out.println("format = " + format + ", tracks = " + tracks + ", division = " + division);
		} else {
			throw new IOException("Not an SMF file.");
			//return;
		}
		score = new Vector();
		for(int i = 0; i < tracks; i++) {
			/*
			oneByte = bistr.read();
			high4bits = ((oneByte & 0xf0) >> 4);
			low4bits = (oneByte & 0x0f);
			System.out.println("track: " + high4bits + ", " + low4bits);
			if (high4bits == 4 && low4bits == 13) { */
			bistr.read(buf);
			if ( buf[2] == 'r' && buf[3] == 'k' ) {
				/*
				for (int j = 0; j < 7; j++) {
					bistr.read();
				}
				 */
				bistr.skip(4);
			} else {
				continue;
			}
			deltaTotal = 0;
			high4bits = 0; low4bits = 0;
			while ( true ) {
					// read the amount of delta time
					deltaTime = parseVarLenInt(bistr);
					if ( deltaTime == -1 )
						break;
					deltaTotal += deltaTime;
					//
					oneByte = bistr.read();
					if ( (oneByte & 0x80) != 0 ) {
						// this is status byte.
						high4bits = (oneByte & 0xf0) >> 4;
						low4bits = oneByte & 0x0f;
						//
						if ( oneByte == 0xf0 ) {
							// system exclusive event
						} else if ( oneByte == 0xf7 ) {
							// escaped system exclusive event
						} else if ( oneByte == 0xff ) {
							// meta event
							oneByte = bistr.read(); // event type
							len = parseVarLenInt(bistr); // event size
							if ( oneByte == 0x2f ) {
								// may be 'end of track'
								// len is always assumed to be zero. 
								score.add(new MusicalNote(-1,-1,-1) );
								break; // go to the next track.
							} else {
								// System.out.println("len = "+len);
								for (int j = 0; j < len; j++)
									bistr.read();
								continue;
							}
						}
						oneByte = bistr.read(); // read the first data byte.
					}
					// oneByte is the first data byte.
					switch (high4bits) {
						case 0x08: // note off
							// oneByte == note number
							bistr.read(); // velocity
							break;
						case 0x09: // note off or note on
							// oneByte == note number
							velocity = bistr.read(); // velocity
							//if ( !( 20 < velocity /* && velocity < 109 */ ) )
							//	continue;
							if ( velocity == 0x00 ) 
								break;
								// this is note off w/ velocity 64
							if ( low4bits == 9 ) {
								// a percussion note
								continue;
							}
							/*
							if ( score.size() > 0 && 
								 (((MusicalNote) score.lastElement()).noteOn == deltaTotal) ) {
								//((MusicalNote) score.lastElement()).setValues(low4bits + 1, deltaTotal, oneByte);
								//System.out.print("+");
							} else { */
								score.add( new MusicalNote(low4bits + 1, deltaTotal, oneByte & 0x7f) );
							//}
							break;
						case 0x0a: // poly. pressure
						case 0x0b: // control change
							//
							bistr.read();
							break;
						case 0x0c: // prog. change
						case 0x0d: // ch. pressure
							// 
							break;
						case 0x0e: // pitch bend change
							//
							bistr.read();
							break;
					}

			}
		}
		//
		return;
	}
	
	MusicalNote noteAt(int i) {
		return (MusicalNote) score.elementAt(i);
	}
	
	int size() {
		return score.size();
	}
	
	int division() {
		return division;
	}
	
	int dist(MusicalNote tprev, MusicalNote tcurr, MusicalNote pprev, MusicalNote pcurr) {
		if ( (tprev.channel == 0) || tcurr.channel == 0 ) {
			return Integer.MAX_VALUE;
		}
		if ( (tprev.number - tcurr.number) != (pprev.number - pcurr.number) ) {
			return Integer.MAX_VALUE;
		}
		if ( (tprev.channel != tcurr.channel) || pprev.channel != pcurr.channel ){
			return Integer.MAX_VALUE;
		}
		//returns the difference in note duration * defaultDivision
		return (int) Math.abs( (tcurr.noteOn - tprev.noteOn) * defaultDivision / division - (pcurr.noteOn - pprev.noteOn) ) ;
	}
	
	class DPTable {
		int d[][];
		
		DPTable(SMFScore text, SMFScore melody) {
			d = new int[melody.size()][text.size()];
			// tasks for initialization;
			
			int l = text.size();
			for (int i = 0; i < d.length; i++) {
				Arrays.fill(d[i],0,i,Integer.MAX_VALUE);
			}
			Arrays.fill(d[0], 0, d[0].length, 0);
			
		}
		
		public String toString() {
			StringBuffer buf = new StringBuffer("");
			for (int c = 0; c < d[0].length; c++) {
				buf.append( c + ": " + noteAt(c).toString() + " " );
				for (int r = 0; r < d.length; r++) {
					if ( d[r][c] == Integer.MAX_VALUE ) {
						buf.append("\t+++");
					} else {
						buf.append("\t"+d[r][c]);
					}
					buf.append(" ");
				}
				buf.append("\r");
			}
			return buf.toString();
		}
	}
	
	
public void skippingFillDPTable(DPTable tbl, SMFScore melody) {
	int row, col, cc; //, lb;
	int prevNote;
	int pivot[] = new int[128];
	long Dneg[] = new long[128];
	long Dpos;
	
	for (col = 0; col < size(); col++) {
		if ( noteAt(col).isEndOfTrack() ) {
			tbl.d[0][col] = Integer.MAX_VALUE;
		} else {
			tbl.d[0][col] = 0;
		}
	}
	// for row > 0
	for (row = 1; row < melody.size(); row++) {
		for (col = 0 /*row */; col < size(); ) {
			
			if ( noteAt(col).isEndOfTrack() || col == 0 ) {
				// the channel has begun, or has been changed; initialize as the first column of the table.
				for (int note = 0; note < Dneg.length; note++) {
					pivot[note] = col+1;
					Dneg[note] = Integer.MAX_VALUE;
				}
				for (cc = col; cc < col + row + 1 && cc < size() ; cc++) {
					if (noteAt(cc).isEndOfTrack())
						col = cc;
					tbl.d[row][cc] = Integer.MAX_VALUE;
				}
				col = cc;
				continue;
			}
			for (int note = 0; note < Dneg.length; note++)
				if ( Dneg[note] != Integer.MAX_VALUE ) 
					Dneg[note] = Dneg[note] + (noteAt(col).noteOn - noteAt(col-1).noteOn);
			Dpos = Integer.MAX_VALUE;
			//
			prevNote = noteAt(col).number - ( melody.noteAt(row).number - melody.noteAt(row - 1).number );
			if ( prevNote < 0 || prevNote >= 128 ) {
				tbl.d[row][col] = Integer.MAX_VALUE;
			} else {
				for (cc = pivot[prevNote]; cc < col; cc++) {
					if ( melody.noteAt(row).noteOn - melody.noteAt(row-1).noteOn < noteAt(col).noteOn - noteAt(cc).noteOn ) {
						pivot[prevNote] = cc + 1;
						long tmp = ((long) tbl.d[row-1][cc]) + dist(noteAt(cc), noteAt(col), melody.noteAt(row-1), melody.noteAt(row));
						if ( tmp < Dneg[prevNote] ) {
							Dneg[prevNote] = tmp;
						}
					} else {
						Dpos = Math.min(Dpos, ((long) tbl.d[row-1][cc]) + dist(noteAt(cc), noteAt(col), melody.noteAt(row-1), melody.noteAt(row)) );
					}
				}
				tbl.d[row][col] = (int) Math.min( Dneg[prevNote], Dpos );
			}
			col++;
		}
	}
	return; 
}
	
public void quickerFillDPTable(DPTable tbl, SMFScore melody) {
	int row, col, cc; //, lb;
	int prevNote;
	int pivot[] = new int[128];
	long Dneg[] = new long[128];
	long Dpos;
	
	for (col = 0; col < size(); col++) {
		if ( noteAt(col).isEndOfTrack() ) {
			tbl.d[0][col] = Integer.MAX_VALUE;
		} else {
			tbl.d[0][col] = 0;
		}
	}
	// for row > 0
	for (row = 1; row < melody.size(); row++) {
		for (col = 0 /*row */; col < size(); ) {
			
			if ( noteAt(col).isEndOfTrack() || col == 0 ) {
				// the channel has begun, or has been changed; initialize as the first column of the table.
				for (int note = 0; note < Dneg.length; note++) {
					pivot[note] = col+1;
					Dneg[note] = Integer.MAX_VALUE;
				}
				for (cc = col; cc < col + row + 1 && cc < size() ; cc++) {
					if (noteAt(cc).isEndOfTrack())
						col = cc;
					tbl.d[row][cc] = Integer.MAX_VALUE;
				}
				col = cc;
				continue;
			}
			for (int note = 0; note < Dneg.length; note++)
				if ( Dneg[note] != Integer.MAX_VALUE ) 
					Dneg[note] = Dneg[note] + (noteAt(col).noteOn - noteAt(col-1).noteOn);
			Dpos = Integer.MAX_VALUE;
			//
			prevNote = noteAt(col).number - ( melody.noteAt(row).number - melody.noteAt(row - 1).number );
			if ( prevNote < 0 || prevNote >= 128 ) {
				tbl.d[row][col] = Integer.MAX_VALUE;
			} else {
				for (cc = pivot[prevNote]; cc < col; cc++) {
					if ( melody.noteAt(row).noteOn - melody.noteAt(row-1).noteOn < noteAt(col).noteOn - noteAt(cc).noteOn ) {
						pivot[prevNote] = cc + 1;
						long tmp = ((long) tbl.d[row-1][cc]) + dist(noteAt(cc), noteAt(col), melody.noteAt(row-1), melody.noteAt(row));
						if ( tmp < Dneg[prevNote] ) {
							Dneg[prevNote] = tmp;
						}
					} else {
						Dpos = Math.min(Dpos, ((long) tbl.d[row-1][cc]) + dist(noteAt(cc), noteAt(col), melody.noteAt(row-1), melody.noteAt(row)) );
					}
				}
				tbl.d[row][col] = (int) Math.min( Dneg[prevNote], Dpos );
			}
			col++;
		}
	}
	return; 
}



public void fillByRowDPTable(DPTable tbl, SMFScore melody) {
	//
	int row, col, lb;
	long min;
	
	for (row = 1; row < melody.size(); row++) {
		for (col = row, lb = col-1; col < size(); col++) {
			if ( noteAt(col).noteOn < noteAt(col-1).noteOn ) {
				// the channel has been changed; initialize as the first column of the table. 
				lb = col; 
			}
			min = Integer.MAX_VALUE;
			for ( int cc = lb; cc < col; cc++) {
				min = Math.min( min, ((long) tbl.d[row-1][cc]) + dist(noteAt(cc), noteAt(col), melody.noteAt(row-1), melody.noteAt(row)) );
				//System.out.println(min);
			}
			tbl.d[row][col] = (int) min; // Math.min( D_minus[r][0], D_plus ); 
		}
	}
	return; 
}

public void fillByColumnDPTable(DPTable tbl, SMFScore melody) {
	//		
	int row, col, lb;
	long min;
	
	//totalTime -= System.currentTimeMillis();
	for (col = 1, lb = 0; col < size(); col++) {
		for (row = 1; row < melody.size() && (row <= col); row++) {
			//if (noteAt(c).channel != noteAt(c-1).channel) {
			if ( noteAt(col).isEndOfTrack() ) {
				// the channel has been changed; initialize as the first column of the table. 
				lb = col; 
			}
			min = Integer.MAX_VALUE;
			for ( int cc = lb ; cc < col; cc++) {
				min = Math.min( min, ((long) tbl.d[row-1][cc]) + dist(noteAt(cc), noteAt(col), melody.noteAt(row-1), melody.noteAt(row)) );
				//System.out.println(min);
			}
			tbl.d[row][col] = (int) min; // Math.min( D_minus[r][0], D_plus ); 
			}
		}
	return; 
	}


public Vector approximateSearchFor(SMFScore melody, int err) {
	long swatch;
	DPTable tbl = new DPTable(this, melody);
	swatch = System.currentTimeMillis();
	quickerFillDPTable(tbl, melody);
	swatch = System.currentTimeMillis() - swatch;
	System.out.println("[" + swatch + "]");
	//System.out.println(tbl);
	return traceBackTable(tbl, melody, err);
}


	public Vector traceBackTable(DPTable tbl, SMFScore mel, int err) {
		int col, row;
		int occurrence, cc;
		long best;
		Vector occurrences = new Vector(); // the empty Vector
		
		if ( err == -1 ) { // find the left-most occurrence with the minimum distance
			occurrence = mel.size() - 1;
			for ( col = 0 + mel.size(); col < this.size(); col++) {
				if ( tbl.d[mel.size() - 1][col] < tbl.d[mel.size() - 1][occurrence] ) {
					occurrence = col;
				}
			}
		} else { // find the left-most occurrence within the specified distance
			occurrence = this.size();
			for ( col = 0 + mel.size(); col < this.size(); col++) {
				if ( tbl.d[mel.size() - 1][col] <= err ) {
					occurrence = col;
					break;
				}
			}
		}
		
		if ( occurrence == this.size() /* exceeded the size of text */) {
			//occindex[0] = -1;
			//return Integer.MAX_VALUE;
			return occurrences; // returns the empty Vector "occurrences"
		}
		//
		occurrences.addElement(new int[mel.size()]);
		((int[])occurrences.lastElement())[mel.size() - 1] = occurrence;
		//System.out.println(occurrence);
		best = tbl.d[mel.size() - 1][occurrence];
		for ( row = mel.size() - 1, col = occurrence; row > 0; row--) {
			for ( cc = col - 1; ! (cc < 0); cc--) {
				//System.out.print(cc+", ");
				if ( ((long)tbl.d[row-1][cc]) == best - dist(noteAt(cc), noteAt(col), mel.noteAt(row-1), mel.noteAt(row)) ) {
					break;
				}
			}
			if ( cc < 0 )
			//	return new int[0];
				System.out.println(tbl);
			best -= dist(noteAt(cc), noteAt(col), mel.noteAt(row-1), mel.noteAt(row));
			col = cc;
			((int[])occurrences.lastElement())[row-1] = col;
		}
		//
		return occurrences; 
	}
	
	
	public String toString() {
		StringBuffer buf = new StringBuffer();
		buf.append("SMFScore(");
		buf.append("Format "+format+", TQPN=" + division +", ");
		for(int i = 0; i < score.size(); i++) {
			if (! (i < 24) ) {
				buf.append("...");
				break;
			}
			buf.append( /* "["+i+"] " + */ ((MusicalNote) score.elementAt(i)) );
		}
		buf.append(" ) ");
		return buf.toString();
	}
	
    public static void main (String args[]) throws Exception {
        // insert code here...
		SMFScore smfscore;
		int melodyErr;
        System.out.println("Hello World!");
		
		long t2 = 0;
		// Get the file name list
		Vector fileNames = new Vector();
		File file = new File(args[args.length-1]);
		String files[], fileName;
		if ( file.isDirectory() ) {
			files = file.list();
			for (int i = 0; i < files.length; i++) {
				if ( (! files[i].startsWith(".")) &&  files[i].endsWith(".mid") ) {
					fileNames.add(file.getPath() + File.separatorChar + files[i]);
				}
			}
		} else {
			fileNames.add(file.getPath());
		}
		
		SMFScore melody = new SMFScore(args[0]);
		melodyErr = Integer.parseInt(args[1]);
		System.out.println(melody);
		//int edist;
		Vector occurrences;
		int occurrence[];
		//int occurrence[] = new int[melody.size()];
		
		long stopwatch = System.currentTimeMillis();
		java.text.DecimalFormat fmt = new java.text.DecimalFormat();
		// perform search
		//SMFScore.totalTime = 0;
		for (Iterator i = fileNames.iterator(); i.hasNext(); ) {
			fileName = (String) i.next();
			t2 -= System.currentTimeMillis();			
			FileInputStream istream = new FileInputStream( fileName );
			try {
				smfscore = new SMFScore(new BufferedInputStream(istream));
			} catch (IOException e) {
				System.out.println(fileName+": ");
				throw e;
			}
			istream.close();
			t2 += System.currentTimeMillis();
			//System.out.println("For parsing the score: " + (-stopwatch) );

			if (smfscore.size() == 0) {
				System.out.println(fileName + ": unknown file format!");
				System.out.println();
				continue;
			} else {
				System.out.println(fileName + " (" + smfscore.size() + ") ");
			}
			//
			occurrences = smfscore.approximateSearchFor(melody, melodyErr );
			//
			//SMFScore.totalTime += System.currentTimeMillis();
			if ( occurrences.isEmpty() ) { 
				continue;
			}
			occurrence = (int[]) occurrences.lastElement();

			System.out.println();
			System.out.println(fileName);
			System.out.println("TPQN = " + smfscore.division + ", the total number of notes = " + smfscore.size() /* + ", millisecs to find: " + (-stopwatch) */ );
			System.out.print("Ch. "+ smfscore.noteAt(occurrence[0]).channel + " from " + occurrence[0] + "th note: ");
			int k, j;
			for ( k = 0, j = occurrence[k]; j <= occurrence[occurrence.length - 1]; j++) {
				if ( smfscore.noteAt(j).channel != smfscore.noteAt(occurrence[0]).channel )
					continue;
				if ( j == occurrence[k] ) {
					System.out.print("*");
					k++;
				}
				System.out.print(""+smfscore.noteAt(j)+", ");
			}

			System.out.println();
			System.out.println();
			//System.out.println();
		}
		//
		//System.out.println("\r" + "Total time: " + SMFScore.totalTime );
		System.out.println("Total input time: " + t2 );
		System.out.println("Execution time: " + (System.currentTimeMillis() - stopwatch) );
		//
		return;
    }
}
