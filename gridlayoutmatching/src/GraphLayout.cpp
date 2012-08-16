/*
 * GridLayout.cpp
 *
 *  Created on: 2012/06/20
 *      Author: sin
 */

#include <cmath>
//using namespace std;

//#include "Graph.h"
#include "mytypes.h"

#include "GraphLayout.h"

uint GraphLayout::bestGridAlignGap() {
	map<Point2D, triple<uint,int,int> > table;
			// more than one points
			uint rgtindex = rightmost();
			uint topindex = top();
			if (table.find(Point2D(rgtindex, topindex)) != table.end()) {
				return table[Point2D(rgtindex, topindex)].first;
			}
			if (subgraphSize() == 1) {
				table[Point2D(rgtindex, topindex)] = triple<uint,int,int>(0,-1,-1);
				return 0;
			}
			//
			uint rtprev = rightmostPrevious();
			uint tpprev = topPrevious();
			Point2D rgtdist = node(rgtindex) - node(rtprev);
			Point2D topdist = node(topindex) - node(tpprev);
			int rightmargin = rgtdist.x;
			int topmargin = topdist.y;
			//
			Point2D rgridtrans = GraphLayout::gridTransfer(rgtdist, rightmargin, topmargin);
			Point2D tgridtrans = GraphLayout::gridTransfer(topdist, rightmargin, topmargin);
			//
			uint rcost, tcost;
			removeRightmost();
			rcost = (rgridtrans - rgtdist).norm1() + bestGridAlignGap(table);
			restoreNode(rgtindex);
			removeTop();
			tcost = (tgridtrans - topdist).norm1() + bestGridAlignGap(table);
			restoreNode(topindex);
			if (rcost <= tcost) {
				if ( rgtindex == topindex ) {
					table[Point2D(rgtindex, topindex)] = triple<uint,int,int>(rcost, rtprev, tpprev);
				} else {
					table[Point2D(rgtindex, topindex)] = triple<uint,int,int>(rcost, rtprev, topindex);
				}
				return rcost;
			} else {
				if ( rgtindex == topindex ) {
					table[Point2D(rgtindex, topindex)] = triple<uint,int,int>(tcost,rtprev,tpprev);
				} else {
					table[Point2D(rgtindex, topindex)] = triple<uint,int,int>(tcost,rgtindex,tpprev);
				}
				return tcost;
			}

	return 0;
}

Point2D GraphLayout::gridTransfer(Point2D & pt, long leftmargin, long bottommargin, const long gridgap) {
	long x0 = gridgap * (int) ceil((float) pt.x / gridgap);
	long x1 = gridgap * (int) floor((float) pt.x / gridgap);
	long y0 = gridgap * (int) ceil((float) pt.y / gridgap);
	long y1 = gridgap * (int) floor((float) pt.y / gridgap);
	if (abs(x0 - pt.x) > abs(x1 - pt.x)
			&& abs(leftmargin) > abs(x1 - pt.x)) {
		x0 = x1;
	}
	if (abs(y0 - pt.y) > abs(y1 - pt.y)
			&& abs(bottommargin) > abs(y1 - pt.y)) {
		y0 = y1;
	}
	return Point2D(x0, y0);
}
