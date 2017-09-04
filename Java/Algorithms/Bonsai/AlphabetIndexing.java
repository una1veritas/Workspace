import java.lang.*;
import java.util.*;

public class AlphabetIndexing extends Hashtable {
    Vector domain, range;
    int lastUpdated;

    public AlphabetIndexing(String rng) {
        super();
        if (! (rng.length() > 1) ) {
            rng = "12";
        }
        domain = new Vector();
        range = new Vector();
        for (int i = 0; i < rng.length(); i++) {
            if (!range.contains(new Character(rng.charAt(i)))) {
                range.addElement(new Character(rng.charAt(i)));
            }
        }
        lastUpdated = 0;
    }
    
    public synchronized Object clone() {
        AlphabetIndexing indexing;
        indexing = (AlphabetIndexing) super.clone();
        indexing.domain = (Vector) domain.clone();
        indexing.range = (Vector) range.clone();
        lastUpdated = indexing.lastUpdated;
        return (Object) indexing;
    }

    public AlphabetIndexing randomize() {
        Random rnd = new Random();
        Enumeration enm;
        for (enm = domain.elements(); enm.hasMoreElements(); ) {
            put(enm.nextElement(),range.elementAt(Math.abs(rnd.nextInt()) % range.size()));
        }
        lastUpdated = Math.abs(rnd.nextInt()) % domain.size();
        return this;
    }
    
    public AlphabetIndexing addToDomain(String str) {
        Character c;
        int pos;
        for (int i = 0; i < str.length(); i++) {
            c = new Character(str.charAt(i));
            if (!containsKey(c)) {
                put(c, range.elementAt(0));
                for (pos = 0; pos < domain.size(); pos++) {
                    if ( ((Character)domain.elementAt(pos)).charValue() > c.charValue()) {
                        domain.insertElementAt(c, pos);
                        break;
                    }
                }
                if (pos == domain.size()) {
                    domain.addElement(c);
                }
            }
        }
        return this;
    }
    
    public String translate(String str) {
        StringBuffer transedBuff = new StringBuffer(str.length());
        for (int i = 0; i < str.length(); i++) {
            transedBuff.append(get(new Character(str.charAt(i))));
        }
        return new String(transedBuff);
    }
    
    public void translate(char carray[]) {
        for (int i = 0; i < carray.length; i++) {
            carray[i] = ((Character)get(new Character(carray[i]))).charValue();
        }
    }
    
    public String domainString() {
        StringBuffer buf = new StringBuffer();
        for (Enumeration e = domain.elements(); e.hasMoreElements(); ) {
            buf.append((Character)e.nextElement());
        }
        return new String(buf);
    }
        
    public class Neighbor {
        AlphabetIndexing indexing;
        int domainIndex, rangeIndex;
        int neighborsCount;

        Neighbor(AlphabetIndexing ix) {
            indexing = ix;
            domainIndex = (indexing.lastUpdated+1) % indexing.domain.size();
            rangeIndex = 0;
            neighborsCount = 0;
        }

        public boolean hasNext() {
            if (neighborsCount > indexing.domain.size() * (indexing.range.size() -1)) {
                System.out.println("Neighbor enumeration error!!");
                return false;
            }
            return (indexing.lastUpdated != domainIndex);
        }

        private void findNextDisplacement() {
            Character orgc;
            if (! (rangeIndex < indexing.range.size())) {
                domainIndex = (domainIndex + 1) % indexing.domain.size();
                rangeIndex = 0;
            }
            while (true) {
                orgc = (Character) indexing.get(indexing.domain.elementAt(domainIndex));
                if (indexing.range.elementAt(rangeIndex).equals(orgc)) {
                    if (rangeIndex + 1 < indexing.range.size()) {
                        rangeIndex++;
                        return;
                    } else {
                        domainIndex = (domainIndex + 1) % indexing.domain.size();
                        rangeIndex = 0;
                        // this part should be passed at most once.
                    }
                } else {
                    return;
                }
            }
        }

        public AlphabetIndexing next() {
            Character chr;
            AlphabetIndexing neighbor;
            neighborsCount++;

            findNextDisplacement();
            neighbor = (AlphabetIndexing) indexing.clone();
            neighbor.put(neighbor.domain.elementAt(domainIndex),
                            neighbor.range.elementAt(rangeIndex));
            neighbor.lastUpdated = domainIndex;
            rangeIndex++;
            return neighbor;
        }

    }
}

