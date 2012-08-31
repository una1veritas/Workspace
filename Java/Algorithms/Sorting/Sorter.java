import java.io.*;

class Sorter {

    public static void quickSort(int array[]) {
        quickSort(array, 0, array.length);
    }

    public static void quickSort(int array[], int start, int end) {
        int pivot, smaller, larger, temp;
        
        if (end - start <= 1)
            return;
        pivot = (array[start] + array[end-1] + array[start + ((end - start)/2)])/3;
        smaller = start;
        larger = end;
        while ( smaller < larger ) {
            if (array[smaller] <= pivot) {
                smaller++;
            } else {
                temp = array[smaller];
                array[smaller] = array[larger-1];
                array[larger-1] = temp;
                larger--;
            }
        }
        if (smaller == end || smaller == start)
            return;
        quickSort(array,start,smaller);
        quickSort(array,smaller,end);
        return;
    }

    public static void mergeSort(int array[]) {
        int copy[] = new int[array.length];
        int width, group, subseq1, subseq2, limit1, limit2, copyIndex;
        
        for (width = 1; width < array.length; width = width * 2) {
            for (group = 0; group * 2 * width < array.length; group++) {
                subseq1 = group * 2 * width;
                subseq2 = subseq1 + width;
                copyIndex = subseq1;
                limit1 = Math.min(subseq1+width, array.length);
                limit2 = Math.min(subseq2+width, array.length);
                while (subseq1 < limit1 && subseq2 < limit2) {
                    if (array[subseq1] < array[subseq2]) {
                        copy[copyIndex] = array[subseq1];
                        subseq1++;
                    } else {
                        copy[copyIndex] = array[subseq2];
                        subseq2++;
                    }
                    copyIndex++;
                }
                for ( ; subseq1 < limit1; subseq1++, copyIndex++) 
                    copy[copyIndex] = array[subseq1];
                for ( ; subseq2 < limit2; subseq2++, copyIndex++) 
                    copy[copyIndex] = array[subseq2];
            }
            for (copyIndex = 0; copyIndex < array.length; copyIndex++) {
                array[copyIndex] = copy[copyIndex];
            }
        }
    }
    
    private static void showContents(int array[]) {
        int i;
        System.out.println("Input size: "+array.length);
        for (i = 0; i < array.length-1; i++) {
            System.out.print(array[i]+", ");
        }
        System.out.println(array[i]);
    }

    public static void main(String argv[]) {
        int array[] = {8, 18, 2, 10, 7, 9, 5, 1, 2, 4, 11, 6, 1, 9};
        Sorter.showContents(array);
        Sorter.quickSort(array);
        System.out.println("\nSorted.");
        Sorter.showContents(array);
		try {
		    (new BufferedReader(new InputStreamReader(System.in))).readLine();
		} catch(IOException e) { }
    }

}