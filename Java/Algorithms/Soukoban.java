class Soukoban {

    int map[][];
    int columns, rows;

    Soukoban () {
	columns = 20;
	rows = 20;
	map = new int[columns][rows];
	String resource[] =
	    {"WWWWWWWWWWWFFFFFFFFF",
	     "WFFFFFFFFFWWWWWWWWWW",
	     "WFFFFFFFFFFFFFFFFFFW",
	     "WFFFFFFFFFFFFFFFFFFW",
	     "WFFFFFFFFFFFFFFFFFFW",
	     "WFFFFFFFFFFFFFFFFFFW",
	     "WFFFFFFFFFFFFFFFFFFW",
	     "WFFFFFFFFFFFFFFFFFFW",
	     "WFFFFFFFFFFFFFFFFFFW",
	     "WFFFFFFFFFFFFFFFFFFW",
	     "WFFFFFFHFFFFFFFFFFFW",
	     "WFFFFFFFBFFFFFFFFFFW",
	     "WWWWWFFFFFFFFFFFFFFW",
	     "WFFFWFFFFFFFFFFFFFFW",
	     "WFFFWFFFFFFFFFFFFFFW",
	     "WFFFFFFFFFFFFFFFFFFW",
	     "WFFFFFFFFFFFFFFFFFFW",
	     "WFFFFFFFFFFFFFFFFFFW",
	     "WFFFFFFFFFFFFFFFFFFW",
	     "WWWWWWWWWWWWWWWWWWWW" };

	for (int r = 0; r < rows; r++) {
	    for (int c = 0; c < columns; c++) {
		map[c][r] = resource[r].charAt(c);
	    }
	}
    }

    Soukoban(String mapstr) {
	int c, r, i;

	for (columns = 0; mapstr.charAt(columns) != '\n'; columns++);
	for (rows = 0, c = 0; c < mapstr.length(); c++) {
	    if (mapstr.charAt(c) == '\n')
		rows++;
	}
	System.out.println("Rows: " + rows + "Columns: " + columns);

	map = new int[columns][rows];
	for (i = 0, r = 0; r < rows; r++) {
	    for (c = 0; c < columns; c++) {
		map[c][r] = mapstr.charAt(i);
		i++;
		if (mapstr.charAt(i) == '\n')
		    i++;
	    }
	}

    }
    
    void printMap() {
	for (int r = 0; r < rows; r++) {
	    for (int c = 0; c < columns; c++) {
		if ((char)map[c][r] == 'F')
		    System.out.print(' ');
		else
		    System.out.print((char)map[c][r]);
	    }
	    System.out.println();
	}
    }

    public static void main(String args[]) {
	Soukoban w;
	String m = 
	    "WWWWW\nWFWFW\nWFWFW\nWFMFW\nWFFWW\nWWWWW\n";

	w = new Soukoban(m);
	w.printMap();
    }
}

