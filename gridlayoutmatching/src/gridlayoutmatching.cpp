//============================================================================
// Name        : gridlayoutmatching.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

#include <string>
#include <set>
#include <map>
#include <deque>
#include <vector>

#include "Point2D.h"
#include "Graph.h"
#include "GraphLayout.h"

#include "mytypes.h"

using namespace std;

uint readPoint2Ds(istream & stream, vector<Point2D> & ptlist, const int scale = 100);

int main(int argc, char * argv[]) {
	time_t watchstart, watchstop;
	timeval tvstart, tvstop;

	uint width = 767;
	uint hight = 767;
	uint nsize = 8;

	cerr << argc << endl;

	long x, y;
	string buf;
	vector<Point2D> points;
	  istringstream line;

	if (argc == 2) {
		nsize = atol(argv[1]);
		time(&watchstart);
		srand(watchstart);
		for (uint i = 0; i < nsize; i++) {
			x = rand() % width;
			y = rand() % hight;
			points.push_back(Point2D(x, y));
		}
	} else if (argc == 1 ) {
		cerr << "reading from stdin..." << endl;
		readPoint2Ds(cin, points, 200);
		cerr << "reading finished." << endl;
	}

	for(vector<Point2D>::iterator i = points.begin();
			i != points.end(); i++) {
		cerr << *i << ", ";
	}

	cerr << endl << endl;
	GraphLayout gl(points);
	cerr << gl << endl;

	cerr << "xsorted: ";
	for (uint i = 0; i < gl.graphSize(); i++) {
		cerr << gl.node(gl.xsorted[i]) << ", ";
	}
	cerr << endl;
	cerr << "ysorted: ";
	for (unsigned int i = 0; i < gl.graphSize(); i++) {
		cerr << gl.node(gl.ysorted[i]) << ", ";
	}
	cerr << endl << endl;

	uint right = gl.rightmost();
	uint top = gl.top();
	cerr << "Now computing..." << endl;
	map<Point2D, triple<uint,int,int> > table;
	map<Point2D, triple<uint,int,int> >::iterator it;
	table.clear();
	gl.bestGridAlignGap();
	time(&watchstart);
	gettimeofday(&tvstart, NULL);
	long talcost = gl.bestGridAlignGap(table);
	gettimeofday(&tvstop, NULL);
	time(&watchstop);
	cerr << endl;

	cerr << "(BottomLeft, TopRight) = " << "(" << gl.graphLeft() << ", " << gl.graphBottom() << ") --- ("
			<< gl.graphRight() << ", " << gl.graphTop() << ") " << endl << endl;
	deque<Point2D> seq;
	Point2D p;
	for(p = Point2D(right, top); table.find(p) != table.end();  ) {
		seq.push_front(p);
//		cout << p << endl;
		p = Point2D(table[p].second,table[p].third);
	}
	//seq.push_front(p);
	//cout << p;
	cerr << endl << endl;
//	cout << "added" << '\t' << "x" << '\t' << "y" << '\t' << "prev" << '\t' <<  "x" << '\t' << "y" << '\t' << "prev x" << '\t' << "prev y" <<  '\t' << "transfer" << endl;
	vector<pair<int,Point2D> > layouted(seq.size());
	Point2D topright;
	for(uint i = 0; i < seq.size(); i++) {
		topright = seq[i];
		int rgtindex = topright.x;
		int topindex = topright.y;
		cerr << "topright [p_" << rgtindex << ", p_" << topindex << "] = "
				<< " (" << gl.node(rgtindex).x << ", " << gl.node(topindex).y << ") ";
		if ( table[topright].second == -1 && table[topright].second == -1 ) {
			// this point is specified by one point and the first point aligned onto a grid.
			layouted[rgtindex] = pair<int,Point2D>(i,gl.node(rgtindex));
			cerr << endl;
			continue;
		}
		int rprevindex = table[topright].second;
		int tprevindex = table[topright].third;
		Point2D gap;
		if ( rgtindex != rprevindex ) {
			cerr << " both/right " << rprevindex << gl.node(rprevindex) << " ~~> " << rgtindex << gl.node(rgtindex);
			gap = gl.node(rgtindex)-gl.node(rprevindex);
			cerr << " == " << gl.node(rprevindex)+GraphLayout::gridTransfer(gap, gap.x, gap.y);
			layouted[rgtindex] = pair<int,Point2D>(i,layouted[rprevindex].second+GraphLayout::gridTransfer(gap, gap.x, gap.y));
		} else if ( topindex != tprevindex ) {
			cerr << " top " << tprevindex << gl.node(tprevindex) << " ~~> " <<topindex << gl.node(topindex);
			gap = gl.node(topindex)-gl.node(tprevindex);
			cerr << " == " << gl.node(tprevindex)+GraphLayout::gridTransfer(gap, gap.x, gap.y);
			layouted[topindex] = pair<int,Point2D>(i,layouted[tprevindex].second+GraphLayout::gridTransfer(gap, gap.x, gap.y));
		}
		cerr << endl;
	}
	cerr << endl;

	cout << "Pid\tOrder\tgrid.x\tgrid.y\torg.x\torg.y" << endl;
	for(uint i = 0; i < layouted.size(); i++) {
		cout << i << "\t" << layouted[i].first << "\t"
				<< layouted[i].second.x << "\t" << layouted[i].second.y << "\t"
				<< gl.node(i).x << "\t" << gl.node(i).y
				<< endl;
	}

	if ( tvstop.tv_sec - tvstart.tv_sec > 0 )
		cout << endl << "This calculation has took "<< (tvstop.tv_sec - tvstart.tv_sec)*1000.0 + (tvstop.tv_usec - tvstart.tv_usec)/1000.0 << " mu sec." << endl;
	else
		cout << endl << "This calculation has took "<< /* (tvstop.tv_sec - tvstart.tv_sec)*1000 + */ (tvstop.tv_usec - tvstart.tv_usec)/1000.0 << " mu sec." << endl;
	cout << "The best total grid align cost is " << talcost << "." << endl << endl;
 	//
	return 0;
}

uint readPoint2Ds(istream & stream, vector<Point2D> & ptlist, const int scale) {
	string buf;
	istringstream line;
	long x, y;
	double dx, dy;

	ptlist.clear();
	if ( !stream.good() ) return 0;
	while ( !stream.eof() ) {
		stream >> dx;
		stream >> dy;
		x = (long)(dx*scale);
		y = (long)(dy*scale);
		ptlist.push_back(Point2D(x,y));
	}
	return ptlist.size();
}
