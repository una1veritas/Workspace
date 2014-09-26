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

class IntPair : Hashable, Printable {
    var first, second : Int
    
    init(first: Int, second: Int) {
        self.first = first
        self.second = second
    }
    
    var hashValue : Int {
        get { return (first<<15)^second }
    }

    var description: String {
        get { return "(\(first),\(second))"}
    }
}
func == (lhs: IntPair, rhs: IntPair) -> Bool {
    return (lhs.first == rhs.first) && (lhs.second == rhs.second)
}


class KMPMachine {
    var mypattern : String
    var border : [Int]

    init(pattern : String) {
        var i, j: Int
        mypattern = pattern
        var periodic = [ IntPair:Character]()
        border = [Int](count: countElements(mypattern), repeatedValue: -1)
        for i = 1, j = 0, border[0] = 0; i < countElements(mypattern) ;  {
            
            //println(periodic)

            println(pattern[0,i+1])
            for var iter = 0; iter < i - j; iter++ { print(" ") }
            println(pattern[0,j+1])
            print("(i,j,period) = (\(i),\(j); \(i-j))")

            var match = (mypattern[i] == mypattern[j])
            if !match && (pattern[i] == "*") {
                let pair = IntPair(first: i, second: j)
                if let temp_result = periodic[pair] {
                    match = periodic[pair] == pattern[j]
                } else {
                    periodic[pair] = pattern[j]
                    match = true
                }
            } else if !match && (pattern[j] == "*") {
                let pair = IntPair(first: i, second: j)
                if let temp_result = periodic[pair] {
                    match = periodic[pair] == pattern[i]
                } else {
                    periodic[pair] = pattern[i]
                    match = true
                }
            }


            if match {
                j++;
                border[i] = j;
                i++;
                println(" -o-> (\(i),\(j); \(i-j)) ")
            } else if j > 0 {
                j = border[j-1];
                println(" -x-> (\(i),\(j); \(i-j)) ")
            } else /* j == 0 */ {
                border[i] = 0
                i++;
                println(" -x-> (\(i),\(j); \(i-j)) ")
            }
            
            print("B[] = "); println(border)
            println(periodic)
            println()
            
        }
//        println("\n\(periodic).")
    }

    var pattern : String {
        get { return mypattern }
    }
}

println("Hello, World!")

var kmp : KMPMachine = KMPMachine(pattern: "abaa*baaabaaaaababaababaabaas") //"110*100**011001*11001***10")

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
