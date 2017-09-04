
class Occurrence implements Comparable {
    public int identifier, position;
    
    public Occurrence(int id, int pos) {
        identifier = id;
        position = pos;
    }
    
    public boolean equals(Object obj) {
        if (! (obj instanceof Occurrence)) {
            return false;
        }
        return (identifier == ((Occurrence) obj).identifier) &&
                (position == ((Occurrence) obj).position);
    }
    
    
    public int hashCode() {
	    long bits = (long) position;
	    bits ^= ((long) identifier) * 31;
	    return (((int) bits) ^ ((int) (bits >> 32)));
    }
    
    
    public int compareTo(Object obj) {
        if (! (obj instanceof Occurrence)) {
            return -1;
        }
        if (identifier == ((Occurrence) obj).identifier) {
            return position - ((Occurrence) obj).position;
        } else {
            return identifier - ((Occurrence) obj).identifier;
        }
    }
    
    public synchronized String toString() {
        return "("+identifier+", "+position+") ";
    }
}
