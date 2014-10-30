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
    
    func length() -> Int {
        return countElements(self)
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


/* program main part. */
    println("Hello, World!")

let base0 : Int = -1
    var ell = [Character]("_0111011011")
    println("ell = \(ell).")
    println()
    
    /* algorithm naive */
    
    let n = countElements(ell)-1
    var d = n, p = 1  /* d = n means there's no loop. */

    for pp in 1...n /* test period pp */ {
        var t = [Character](count: pp+1, repeatedValue: "_") /* we ignore the 0th element */
        t[1...pp] = ell[n-pp+1...n]
        println("status: t = \(t), pp = \(pp),")
        var k : Int
        for k = pp+1 ; k < n; ++k {
            println("k = \(k).")
            //print("* = ell[\(n-k+1)]? "); println("*" == ell[n-k+1] )
            if "*" != ell[n-k+1] {
                println("t[pp-(k % pp)] = t[\(pp-(k % pp))] = \(t[pp-(k % pp)]).")
                if t[pp-(k % pp)] == "*" {
                    t[pp - (k % pp)] = ell[n - k]
                } else if t[pp-(k % pp)] != ell[n-k] {
                    break
                }
                println("passed.")
            }
        }
        if n - k - 1 + pp < d + p - 1 {
            (d, p) = (n-k, pp)
            print("(d, p) = (\(d), \(p)), t = \(t[1...pp]); ")
            println("\(ell[1...d]) -> \(ell[d+1 ... d+p])")
        }
        println()
    }
    print("(d,p) = (\(d), \(p))")
    println()
