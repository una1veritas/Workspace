//
//  main.swift
//  SubstitutionDecoder
//
//  Created by Sin Shimozono on 2014/08/13.
//  Copyright (c) 2014年 Sin Shimozono. All rights reserved.
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
    
    func componentsSeparatedBy(sep : String) -> [String] {
        var comps : [String] = []
        var suffix = self
        for ; !suffix.isEmpty ; {
            if let lfr = suffix.rangeOfString(sep) {
                comps.append(suffix[Range<String.Index>(start: suffix.startIndex, end: lfr.startIndex)])
                suffix = suffix[Range<String.Index>(start: lfr.endIndex, end: suffix.endIndex)]
            } else {
                comps.append(suffix)
                break
            }
        }
        return comps
    }
    
    var length : Int {
    get { return countElements(self) }
    }
}

class Permutation : Printable {
    var domain : [Character]
    var range : [Character]
    var map : [Int]
    var hash : [Character : Int]
    
    init(alphabet: String, range: String) {
        domain = []
        self.range = []
        map = []
        hash = [:]
        for c in alphabet {
            if find(domain, c) == nil {
                domain.append(c)
            }
        }
        for var i = 0; i < domain.count ; ++i {
            hash[domain[i]] = i
        }
        for var i = range.startIndex; i != range.endIndex ; ++i {
            self.range.append(range[i])
        }
        for var i = 0; i < domain.count; i++ {
            map.append(i)
        }
    }
    
    var description: String {
        var str : String = ""
            for var i = 0; i < domain.count; ++i {
                str += range[map[i]]
            }
            return str
    }
    
    func next() -> Bool {
        var pivot : Int
        var i : Int
        for pivot = map.count-1 ; pivot > 0; --pivot {
            if map[pivot - 1] < map[pivot] {
                break
            }
        }
        if pivot == 0 { return false }
        mapsort(pivot, before: map.count, order: { $0 < $1 } )
        for i = pivot ; map[i] < map[pivot-1] ; ++i {}
        let t = map[i]
        map[i] = map[pivot-1]
        map[pivot-1] = t
        return true
    }
/*
    func asort(from: Int, before: Int) {
        var part = map[from...before-1]
        part.sort { $0 < $1 }
        map[from...before-1] = part
    }
  */
    func mapsort(from: Int, before: Int, order: (Int, Int) -> (Bool) ) {
        var part = map[from...before-1]
        map[from...before-1] = part.sorted(order)
    }
    
    func loop() -> Bool {
        for var i = 0 ; i < map.count; i++ {
            if map[i] == i {
                return true
            }
        }
        return false
    }

    
    func loopSkip() -> Bool {
        // skip the heighest loop point
        for var i = 0 ; i+1 < map.count; i++ {
            if String(domain[map[i]]).uppercaseString == String(range[i]).uppercaseString  {
                mapsort(i+1, before: map.count, { $0 > $1 } )
                return true
            }
        }
        return false
    }

    func translate(msg: String) -> String {
        var str: String = ""
        for var i = msg.startIndex; i != msg.endIndex; ++i {
            if let val = hash[msg[i]] {
                str += range[map[val]]
            } else {
                str += msg[i]
            }
        }
        return str
    }
}


println("Hello, World!")

var perm : Permutation
var message : String
var alphabet : [String]
if Process.arguments.count > 1 {
    message = Process.arguments[2]
    alphabet = Process.arguments[1].componentsSeparatedBy("/")
} else {
    message = "tpfccdlfdttepcaccplircdtdklpcfrp?qeiqlhpqlipqeodfgpwafopwprtiizxndkiqpkiikrirrifcapncdxkdciqcafmdvkfpcadf."
    alphabet = "tpfcdqhaeiklr/WANTODCIHERKS".componentsSeparatedBy("/")
    // tpfcdaeiklrhqogwzxvnm/WANTOIHERKSBCMDYXPULG: WANTTOKNOWWHATITTAKESTOWORKATNSA?CHECKBACKEACHMONDAYINMAYASWEEXPLORECAREERSESSENTIALTOPROTECTINGOURNATION.
}
println(alphabet)
if  alphabet[0].length != alphabet[1].length {
    println("alphabet/range error.")
    exit(1)
}

perm = Permutation(alphabet: alphabet[0], range: alphabet[1])

//"tpfccdlfdttepcaccplircdtdklpcfrp?qeiqlhpqlipqeodfgpwafopwprtiizxndkiqpkiikrirrifcapncdxkdciqcafmdvkfpcadf."

do {
    if perm.loopSkip() { continue }
    println("\(alphabet[0])/\(perm.translate(alphabet[0])): \(perm.translate(message))")
} while perm.next()
println("\(alphabet[0])/\(perm.translate(alphabet[0])): \(message)")

println("done.")
