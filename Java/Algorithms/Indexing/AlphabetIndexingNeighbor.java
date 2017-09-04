import java.util.*;
import AlphabetIndexing;

public class AlphabetIndexingNeighbor {
    AlphabetIndexing indexing;
    int domainIndex, rangeIndex;
    int neighborsCount;

    AlphabetIndexingNeighbor(AlphabetIndexing ix) {
        indexing = ix;
        domainIndex = (indexing.lastUpdated+1) % indexing.domain.size();
        rangeIndex = 0;
        neighborsCount = 0;
    		//{{INIT_CONTROLS
		//}}
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

	//{{DECLARE_CONTROLS
	//}}
}
