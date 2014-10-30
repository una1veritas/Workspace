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


func length1(s : [Character]) -> Int {
    return countElements(s)-1
}

/* program main part. */
    println("Hello, World!")

    var ell = [Character]("_01110111011")
    println("ell = \(ell), length \(countElements(ell))")
    println()
    
    /* algorithm naive */
    
    let n = countElements(ell)-1
    var d = n, p = 1  /* d = n means there's no loop. */

    for pp in 1...n /* test period pp */ {
        var t = [Character](count: pp+1, repeatedValue: "_") /* we ignore the 0th element */
        t[1...pp] = ell[n-pp+1...n]
        println("period table: t[] = \(t[1...length1(t)]), pp = \(pp),")
        var k : Int = pp
        for var kp = pp + 1; kp <= n; ++kp {
            print("k' = \(kp): ")
            //print("* = ell[\(n-kp+1)]? "); println("*" == ell[n-k'+1] )
            if "*" != ell[n-kp+1] {
                println("t[pp-((k'+pp-1) % pp)] = t[\(pp-((kp+pp-1) % pp))] = \(t[pp-((kp+pp-1) % pp)]).")
                if t[pp-((kp+pp-1) % pp)] == "*" {
                    t[pp - ((kp+pp-1) % pp)] = ell[n - kp+1]
                } else if t[pp-((kp+pp-1) % pp)] != ell[n-kp+1] {
                    println("compare failed.")
                    break
                }
                println("compare passed.")
            } else {
                println("compare don't cared.")
            }
            k = kp
        }
        if n - k + pp < d + p - 1{
            (d, p) = (n-k+1, pp)
            print("(d, p) = (\(d), \(p)), t = \(t[1...pp]); ")
            if d-1 > 0 {
                print("\(ell[1...d-1]) -> ")
            }
            print("([\(ell[d])] -> ")
            if p > 1 {
                print("\(ell[d+1...d+p]))");
            } else {
                print(")")
            }
            println()
            println("temporary the minimum.")
        } else {
            println("skipped since not the minimum.")
        }
        println()
    }
    print("(d,p) = (\(d), \(p))")
    println()
