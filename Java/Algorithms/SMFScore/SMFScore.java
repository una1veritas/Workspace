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
	public static long totalTime;
	
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
								 (((MusicalNote) score.lastElement()).noteOn == deltaTotal 
								  && ((MusicalNote) score.lastElement()).number <= oneByte) ) {
								((MusicalNote) score.lastElement()).setValues(low4bits + 1, deltaTotal, oneByte);
							} else
								 */
							score.add( new MusicalNote(low4bits + 1, deltaTotal, oneByte) );
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
	
	
	public int[] approximateSearchFor(SMFScore melody, int err) {
		//
		long lap = System.currentTimeMillis();
		//
		DPTable tbl = new DPTable(this, melody);
				
		long D_plus, delta;
		int r, c, cc;
		int pv[] = new int[melody.size()];
		//long D_minus[] = new long[melody.size()];
		MusicalNote nt;
		
		for (r = 0; r < melody.size(); r++) {
			//D_minus[r] = Integer.MAX_VALUE; // initialization @ the head of a track
			pv[r] = r - 1;
		}
		for (c = 1; c < size(); c++) {
			for (r = 1; r < melody.size() && (r <= c); r++) {
				//if (noteAt(c).channel != noteAt(c-1).channel) {
				if ( noteAt(c).noteOn < noteAt(c-1).noteOn ) {
					// initialize as the first column of the table.
					//D_minus[r] = Integer.MAX_VALUE;
					pv[r] = r - 1 + c;
				} else {
					//D_minus[r] = Math.min( Integer.MAX_VALUE, D_minus[r] + (noteAt(c).noteOn - noteAt(c-1).noteOn) );
				}
				D_plus = Integer.MAX_VALUE;
				for (cc = pv[r]; cc < c; cc++) {
					delta = (melody.noteAt(r).noteOn - melody.noteAt(r-1).noteOn) - (noteAt(c).noteOn - noteAt(cc).noteOn) * defaultDivision / division;
					if ( delta < 0 && (-delta + tbl.d[r-1][cc] > err) ) {
						pv[r] = cc + 1;
					}
					//	D_minus[r] = Math.min( D_minus[r], ((long) tbl.d[r-1][cc]) + dist(division, noteAt(cc), noteAt(c), melody.division, melody.noteAt(r-1), melody.noteAt(r)) );
					//} else {
						D_plus = Math.min( D_plus, 
										   ((long) tbl.d[r-1][cc]) + dist(noteAt(cc), noteAt(c), melody.noteAt(r-1), melody.noteAt(r)) );
					//}
				}
				tbl.d[r][c] = (int) D_plus; /* Math.min( D_minus[r], D_plus ); */
			}
		}
		
		//System.out.print(tbl);
		//System.out.println();
		
		int occurrence;
		int occindex[]; 
		long best;
		
		if ( err == -1 ) {
			occurrence = melody.size() - 1;
			for (c = melody.size(); c < size(); c++) {
				if ( tbl.d[melody.size() - 1][c] < tbl.d[melody.size() - 1][occurrence] ) {
					occurrence = c;
				}
			}
		} else {
			occurrence = size();
			for (c = melody.size(); c < size(); c++) {
				if ( tbl.d[melody.size() - 1][c] <= err ) {
					occurrence = c;
					break;
				}
			}
		}
		
		if ( occurrence == size() ) {
			return new int[0];
		}
		occindex = new int[melody.size()];
		occindex[melody.size() - 1] = occurrence;
		best = tbl.d[melody.size() - 1][occurrence];
		for (r = melody.size() - 1, c = occurrence; r > 0; r--) {
			for (cc = c - 1; ! (cc < 0); cc--) {
				if ( tbl.d[r-1][cc] == best - dist(noteAt(cc), noteAt(c), melody.noteAt(r-1), melody.noteAt(r)) ) {
					break;
				}
			}
			best -= dist(noteAt(cc), noteAt(c), melody.noteAt(r-1), melody.noteAt(r));
			c = cc;
			occindex[r-1] = c;
		}
		//
		totalTime += System.currentTimeMillis() - lap;
		//
		return occindex;
	}
	
	/*
	public int[] BUGGYapproximateSearchFor(SMFScore melody, int err) {
		//
		long lap = System.currentTimeMillis();
		//
		DPTable tbl = new DPTable(this, melody);
		
		long D_plus, delta;
		int r, c, cc;
		int pv[] = new int[melody.size()];
		long D_minus[] = new long[melody.size()];
		MusicalNote nt;
		
		for (r = 0; r < melody.size(); r++) {
			D_minus[r] = Integer.MAX_VALUE; // initialization @ the head of a track
			pv[r] = r - 1;
		}
		for (c = 1; c < size(); c++) {
			for (r = 1; r < melody.size() && (r <= c); r++) {
				//if (noteAt(c).channel != noteAt(c-1).channel) {
				if ( noteAt(c).noteOn < noteAt(c-1).noteOn ) {
					// initialize as the first column of the table.
					D_minus[r] = Integer.MAX_VALUE;
					pv[r] = r - 1 + c;
				} else {
					D_minus[r] = Math.min( Integer.MAX_VALUE, D_minus[r] + (noteAt(c).noteOn - noteAt(c-1).noteOn) );
				}
				D_plus = Integer.MAX_VALUE;
				for (cc = pv[r]; cc < c; cc++) {
					delta = (melody.noteAt(r).noteOn - melody.noteAt(r-1).noteOn) * division - (noteAt(c).noteOn - noteAt(cc).noteOn) * melody.division;
					if ( delta < 0 ) {
						pv[r] = cc + 1;
						D_minus[r] = Math.min( D_minus[r], 
											   ((long) tbl.d[r-1][cc]) + dist(noteAt(cc), noteAt(c), melody.noteAt(r-1), melody.noteAt(r)) );
					} else {
						D_plus = Math.min( D_plus, 
										   ((long) tbl.d[r-1][cc]) + dist(noteAt(cc), noteAt(c), melody.noteAt(r-1), melody.noteAt(r)) );
					}
				}
				tbl.d[r][c] = (int) Math.min( D_minus[r], D_plus );
				}
			}
		
		System.out.print(tbl);
		System.out.println();
		
		int occurrence;
		int occindex[]; 
		long best;
		
		if ( err == -1 ) {
			occurrence = melody.size() - 1;
			for (c = melody.size(); c < size(); c++) {
				if ( tbl.d[melody.size() - 1][c] < tbl.d[melody.size() - 1][occurrence] ) {
					occurrence = c;
				}
			}
		} else {
			occurrence = size();
			for (c = melody.size(); c < size(); c++) {
				if ( tbl.d[melody.size() - 1][c] <= err ) {
					occurrence = c;
					break;
				}
			}
		}
		
		if ( occurrence == size() ) {
			return new int[0];
		}
		occindex = new int[melody.size()];
		occindex[melody.size() - 1] = occurrence;
		best = tbl.d[melody.size() - 1][occurrence];
		for (r = melody.size() - 1, c = occurrence; r > 0; r--) {
			for (cc = c - 1; ! (cc < 0); cc--) {
				if ( tbl.d[r-1][cc] == best - dist(noteAt(cc), noteAt(c), melody.noteAt(r-1), melody.noteAt(r)) ) {
					break;
				}
			}
			best -= dist(noteAt(cc), noteAt(c), melody.noteAt(r-1), melody.noteAt(r));
			c = cc;
			occindex[r-1] = c;
		}
		//
		totalTime += System.currentTimeMillis() - lap;
		//
		return occindex;
	}
	*/

	
	boolean melodylike(int occurrence[], SMFScore melody ) {
		int i, j, p;
		j = occurrence[0];
		for ( ; j > 0 && (noteAt(j).noteOn == noteAt(occurrence[0]).noteOn) ; j--) {
			if ( noteAt(j).number > noteAt(occurrence[0]).number )
				return false;
		}
		for (i = 0; i+1 < occurrence.length; i++) {
			j = occurrence[i];
			for ( ; j < occurrence[i+1] ; j++) {
				if ( noteAt(j).number > Math.max(noteAt(occurrence[i]).number, noteAt(occurrence[i+1]).number) 
					 /* || noteAt(j).number < Math.min(noteAt(occurrence[i]).number, noteAt(occurrence[i+1]).number) */ )
					return false;
			}
		}
		if (! (j < melody.size() * 1.4) )
			return false;
		return true;
	}
	
	public int[] naiveSearchFor(SMFScore melody) {
		int occindex[] = new int[melody.size()];
		int i, j, p, c;
		
		for ( i = 0; i + 1 < size(); i++) {
			if ( noteAt(i).channel == 0 )
				continue;
			occindex[0] = i;
			for (p = i, c = i + 1, j = 0; j + 1 < melody.size() && c < size(); ) {
				if ( (noteAt(c).number - noteAt(p).number) == (melody.noteAt(j+1).number - melody.noteAt(j).number) ) {
					occindex[j+1] = p;
					p = c; c = c + 1; j++;
					continue;
				}
				if ( noteAt(p).noteOn == noteAt(c).noteOn && noteAt(c).number > noteAt(p).number )
					break;
				c++;
			}
			if (! (j + 1 < melody.size()) ) {
				return occindex;
			}
		}
		
		return new int[0];
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
		
		long stopwatch = System.currentTimeMillis();
		java.text.DecimalFormat fmt = new java.text.DecimalFormat();
		// perform search
		SMFScore.totalTime = 0;
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
			//System.out.println(fileName);
			System.out.print(".");
			//System.out.println(smfscore);
			
			if (smfscore.size() == 0) {
				System.out.println(fileName + ": unknown file format!");
				System.out.println();
				continue;
			}
			
			//System.out.println("For parsing the score: " + (-stopwatch) );
			
				//try {
				//System.out.println("Result: " + smfscore.naiveSearchFor(new SMFScore("0 64 0 64 0 64 0 64")) );
				int occurrence[] = smfscore.approximateSearchFor(melody, melodyErr );
				
				if ( occurrence.length == 0 ) {
					//System.out.println();
					continue;
				}
				if ( true /* || smfscore.melodylike(occurrence, melody) */ ) {
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
				}
				System.out.println();
				System.out.println();
			/* } catch (ArrayIndexOutOfBoundsException e) {
				System.out.println();
				System.out.println("!!! ArrayIndexOutOfBounds !!!" + fileName);
				System.out.println();
			} */
			//System.out.println();
		}
		//
		System.out.println("\r" + "Total time: " + SMFScore.totalTime );
		System.out.println("Total input time: " + t2 );
		System.out.println("Execution time: " + (System.currentTimeMillis() - stopwatch) );
		//
		return;
    }
}
