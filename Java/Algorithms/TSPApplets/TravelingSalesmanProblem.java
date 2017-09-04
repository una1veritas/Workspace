
import java.math.*;
import java.awt.*;
import java.util.Vector;

public class TravelingSalesmanProblem extends Object {
    // as a set of Points
    protected Vector cities;
    protected int size;
    // Axis-Parallel Bounding Box
    protected Rectangle boundRect;
    // k-opt support
    
    public TravelingSalesmanProblem(int sz, Rectangle rect) {
        int x,y;
        
        size = sz;
        cities = new Vector(sz);
        boundRect = new Rectangle(rect);
        for (int i = 0; i < sz; i++) {
            // Generate axis values between 0 to rect-size - 1 at random
            x = (int) (Math.random() * boundRect.width);
            y = (int) (Math.random() * boundRect.height);
            cities.addElement(new Point(x,y)); //at i
        }
    
	}
    
    public TravelingSalesmanProblem(int sz) {
        
        this(sz, new Rectangle(0, 0, 240, 240));
    }
    
    public Point cityAt(int i) {
        return (Point) cities.elementAt(i);
    }
    
    public int size() {
        return size;
    }
    
    public Rectangle boundRect() {
        return boundRect;
    }
    
    public double tourLength(int [] tour) {
        int i, dx, dy;
        double sum;
        Point p1, p2;
        if (tour.length != size) {
            return -1;
        }
        for (sum = 0, i = 0; i < size; i++) {
            p1 = (Point) cities.elementAt(tour[i]);
            p2 = (Point) cities.elementAt(tour[(i+1)%size]);
            dx = Math.abs(p1.x - p2.x);
            dy = Math.abs(p1.y - p2.y);
            sum = sum + Math.sqrt(dx*dx+dy*dy);
        }
        return sum;
    }
    
    public boolean findBetterNeighbor(int [] tour) {
        int f, t, fstart, fshift, tstart, tshift;
        int [] newTour = new int[tour.length];
        double currentLen, newLen;
        
        if (tour.length != size) {
            return false;
        }
        currentLen = tourLength(tour);
        fstart = (int) (Math.random() * size);
        for (fshift = 0; fshift < size; fshift++) {
            f = (fstart + fshift) % size;
            tstart = (int) (Math.random() * (size - f)) ;
            for (tshift = 0; tshift < size - f; tshift++) {
                t = f + ((tstart + tshift) % (size - f)) + 1;
                for (int i = 0; i < f; i++) {
                    newTour[i] = tour[i];
                }
                for (int i = t; i < size; i++) {
                    newTour[i] = tour[i];
                }
                for (int i = 0; f + i < t; i++) {
                    newTour[f+i] = tour[t-i-1];
                }
                //System.out.print("("+f+", "+t+"), ");
                newLen = tourLength(newTour); 
                if ( newLen < currentLen) {
                    //System.out.println("Swapped from "+f+" to "+t+" with improvement "+(oldLen - newLen));
                    for (int i = 0; i < tour.length; i++) {
                        tour[i] = newTour[i];
                    }
                    currentLen = newLen;
                    return true;
                }
            }
        }
        return false;
    }
    
    public boolean tryModifiedNeighbor(int [] tour, double temp) {
        double currentLen, newLen;
        int [] newTour = new int[tour.length];
        final double boltzConst = 0.04;
        int fstart, tstart, f, t;
        
        if (tour.length != size) {
            return false;
        }
        
		currentLen = tourLength(tour);
        fstart = (int) (Math.random() * size);
        f = fstart % size;
        tstart = (int) (Math.random() * (size - f)) ;
        t = f + (tstart % (size - f)) + 1;
        for (int i = 0; i < f; i++) {
            newTour[i] = tour[i];
        }
        for (int i = t; i < size; i++) {
            newTour[i] = tour[i];
        }
        for (int i = 0; f + i < t; i++) {
            newTour[f+i] = tour[t-i-1];
        }
        newLen = tourLength(newTour); 
		
		if (newLen < currentLen ||
		    Math.exp(((double) currentLen - newLen)/(temp*boltzConst)) > Math.random()) {
		    //if (newLen > currentLen) 
		    //    System.out.println(currentLen - newLen);
		    for (int i = 0; i < tour.length; i++) {
		        tour[i] = newTour[i];
		    }
		    return true;
        }
        return false;
    }
    
    public int [] randomSolution() {
        int i, j, skip, remained;
        int sol[] = new int[size];
        boolean check[] = new boolean[size];
        
        for (i = 0; i < size; i++) {
            check[i] = true;
        }
        for (i = 0, j = 0, remained = size; remained > 0; remained--) {
            skip = (int) (Math.random() * remained) + 1;
            while (skip > 0) {
                i = (i+1) % size;
                if (check[i]) {
                    skip--;
                }
            }
            sol[j] = i;
            check[i] = false;
            j++;
        }
        return sol;
    }
    
    //
    public String toString() {
	    return getClass().getName() + "{" + boundRect + ": " + cities.toString() + "}";
    }
    
    
}