//
//  NumberPlace.java
//  NumberPlace
//
//  Created by ?? ?? on 05/07/31.
//  Copyright (c) 2005 __MyCompanyName__. All rights reserved.
//
import java.util.*;

public class NumberPlace {

    public static void main (String args[]) {
        // insert code here...
        //System.out.println("Hello World!");
		Permutation p;
		
		for (p = new Permutation(Integer.parseInt(args[0])); ; p.next()) {
			System.out.println(p.toString());
			if (! p.hasNext())
				break;
		}
    }
}
