
class HeapTree {
    int[] heap;
    int size;

    private static int smallestLimitSize = 10;

    HeapTree(int limitSize) {
        heap = new int[Math.max(limitSize, smallestLimitSize)];
        size = 0;
    		//{{INIT_CONTROLS
		//}}
}
    
    public void append(int obj) {
        int i;
        
        if (! (size + 1 < heap.length)) {
            System.out.println("Error: heap over full.\n");
            return;
        }
        heap[size] = obj;
        i = size;
        size++;
        i = upToRoot(i);
        downToLeaf(i);
    }
    
    private int downToLeaf(int i) {
        int t;
        int obj;
        
        while (2*i+1 < size) {
            t = 2*i+1;
            if (2*i+2 < size) {
                if (heap[2*i+1] > heap[2*i+2]) {
                    t = 2*i+2;
                }
            }
            if (heap[t] < heap[i]) {
                obj = heap[t];
                heap[t] = heap[i];
                heap[i] = obj;
                i = t;
            } else {
                return i;
            }
        }
        return i;
    }
    
    private int upToRoot(int i) {
        int t;
        int obj;
        
        while (i > 0) {
            t = (i - 1) / 2;
            if (heap[i] < heap[t]) {
                obj = heap[t];
                heap[t] = heap[i];
                heap[i] = obj;
                i = t;
            } else {
                return i;
            }
        }
        return i;
    }
    
    synchronized public String toString() {
        StringBuffer buf = new StringBuffer();
        
        for (int i = 0; i < size; i++) {
            buf.append(heap[i]);
            buf.append(", ");
        }
        return buf.toString();
    }
    
    public static void main(String[] args) {
        HeapTree hp = new HeapTree(10);
        
        hp.append(11);
        hp.append(10);
        hp.append(4);
        hp.append(22);
        hp.append(2);
        hp.append(7);
        hp.append(13);
        hp.append(1);
        hp.append(5);
        
        System.out.println(hp);
    }
	//{{DECLARE_CONTROLS
	//}}
}