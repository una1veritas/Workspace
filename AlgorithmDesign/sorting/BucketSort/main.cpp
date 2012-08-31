#include <iostream>
#include <string>
#include <vector>

using namespace std;

unsigned long code(string s);

int main (int argc, char * const argv[]) {
    // insert code here...
    
    vector<string>  src(32);
    string tmp;
    while ( !std::cin.eof() ) {
        cin >> tmp;
        src.push_back(tmp);
    }
    
    for(int i = 0; i < src.size(); i++) 
        cout << i << ": " << src[i] << endl;
    cout << "Reading input has been done." << endl;
    cout.flush();
    
    vector<vector<string> > buckets(127);
    for(int i = 0; i < src.size(); i++) {
//        buckets[src[i][0]].push_back(src[i]);
        buckets[code(src[i]) % buckets.size()].push_back(src[i]);
    }
    cout << "All the data thrown into the Buckets." << endl;
    cout.flush();
    cout << "This is the result." << endl;
    
    for(int i = 0; i < buckets.size(); i++) {
        cout << i << ": ";
        // if buckects[i] is empty then the folloing for loop will simply be skipped.
        for(int e = 0; e < buckets[i].size(); e++) {
            cout << "'" << (buckets[i])[e] << "', ";
        }
        cout << endl;
    }
    return 0;
}

unsigned long code(string s) {
    return ((((unsigned long)s[0]<<14) ^ s[s.size()>>1+1])) ^ (s[s.size()-1]<<7);
//    return ((unsigned long)s[0]<<7) ^ s[s.size()-1];
//    return (unsigned long)s[0];
}