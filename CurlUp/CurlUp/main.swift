//
//  main.swift
//  CurlUp
//
//  Created by Sin Shimozono on 9/21/14.
//  Copyright (c) 2014 Kyushu Inst. of Tech. All rights reserved.
//

import Foundation


extension String {
    
    subscript (i: Int) -> Character {
        return self[advance(self.startIndex, i)]
    }
    
    subscript (from: Int, before: Int) -> String {
        var range = Range<String.Index>(start: advance(self.startIndex, min(countElements(self), from)),
            end: advance(self.startIndex, min(countElements(self), before) ) )
            return self.substringWithRange(range)
    }
    
    // string[range] -> substring form start pos on the left to end pos on the right
    subscript(range: Range<Int>) -> String {
        return self[range.startIndex, range.endIndex]
    }
    
}


class KMPMachine {
    var mypattern : String
    var border : [Int]

    init(pattern : String) {
        var i, j: Int
        mypattern = pattern
        var periodic : [ Int: Character ] = [ Int: Character]()
        border = [Int](count: countElements(mypattern), repeatedValue: -1)
        for i = 1, j = 0, border[0] = 0; i < countElements(mypattern) ;  {
            
            println(periodic)

            var match = (mypattern[i] == mypattern[j])
            if pattern[i] == "*" {
                periodic[i] = mypattern[j]
//                println(periodic)
                match = true
            } else if pattern[j] == "*" {
                if let tpm = periodic[j] {
                    if periodic[j] == pattern[i] {
                        match = true
                    }
                }
            }

            if match {
                print("\(pattern[0,i+1]) : \(pattern[i-j,i+1]) ;")
                j++;
                border[i] = j;
                i++;
                println("(i,j) = (\(i),\(j)) ")
            } else if j > 0 {
                print("\(pattern[0,i+1]) : \(pattern[i-j,i+1]) ;")
                j = border[j-1];
                println("(i,j) = (\(i),\(j)) ")
            } else /* j == 0 */ {
                print("\(pattern[0,i+1]) : \(pattern[i-j,i+1]) ;")
                border[i] = 0
                i++;
                println("(i,j) = (\(i),\(j)) ")
                periodic.removeAll(keepCapacity: true)
            }
            println(border)
            println()
            
        }
//        println("\n\(periodic).")
    }

    var pattern : String {
        get { return mypattern }
    }
}

println("Hello, World!")

var kmp : KMPMachine = KMPMachine(pattern: "10010*1011*110*10") //"110*100**011001*11001***10")

for var i : Int = 0; i < countElements(kmp.pattern) ; i++ {
    print(kmp.pattern[i])
    print(" ")
}
println()
for var i : Int = 0; i < countElements(kmp.pattern) ; i++ {
    print(kmp.border[i])
    print(" ")
}
println()
