//
//  KMP.swift
//  CurlUp
//
//  Created by Sin Shimozono on 10/29/14.
//  Copyright (c) 2014 Kyushu Inst. of Tech. All rights reserved.
//

import Foundation

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
            
            let period = i-j
            var match = (mypattern[i] == mypattern[j])
            /*
            if !match && (pattern[i] == "*") {
            let pair = IntPair(first: period, second: i)
            if let temp_result = periodic[pair] {
            match = periodic[pair] == pattern[j]
            } else {
            periodic[pair] = pattern[j]
            match = true
            }
            } else if !match && (pattern[j] == "*") {
            let pair = IntPair(first: period, second: j)
            if let temp_result = periodic[pair] {
            match = periodic[pair] == pattern[i]
            } else {
            periodic[pair] = pattern[i]
            match = true
            }
            }
            */
            
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
