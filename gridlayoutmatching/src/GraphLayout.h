/*
 * GraphLayout.h
 *
 *  Created on: 2012/05/04
 *      Author: sin
 */

#ifndef GRAPHLAYOUT_H_
#define GRAPHLAYOUT_H_

#include <vector>
#include <map>
#include <algorithm>
#include <cmath>

using namespace std;

#include "Graph.h"
#include "mytypes.h"


class GraphLayout {
	Graph graph;
public:
	vector<int> xsorted, ysorted;

public:
	static const long gridWidth = 20;
	static const Point2D nullPoint;

private:
	vector<int> toxsorted, toysorted;
	vector<int>::iterator xsortedlast, ysortedlast;
	vector<boolean> subgraphnodes;
	uint subgraphsize;

	void init_subgraphnodes() {
		for(uint i = 0; i < graph.nodes.size(); i++) {
			subgraphnodes.push_back(true);
		}
		subgraphsize = graph.size();
	}

	void init_axis_sort_array() {
		for (uint i = 0; i < graph.size(); i++) {
			xsorted.push_back(i);
			ysorted.push_back(i);
		}
		axis_sort();
		toxsorted.resize(graph.size());
		toysorted.resize(graph.size());
		for(uint i = 0; i < graph.size(); i++) {
			toxsorted[xsorted[i]] = i;
			toysorted[ysorted[i]] = i;
		}
		xsortedlast = xsorted.end();
		ysortedlast = ysorted.end();
	}

public:
	GraphLayout(long xvals[], long yvals[], uint arraysize) : graph() {
		for(uint i = 0; i < arraysize; i++) {
			graph.addNode(Node(xvals[i], yvals[i]));
		}
		init_subgraphnodes();
		init_axis_sort_array();
	}

	GraphLayout(double xvals[], double yvals[], uint arraysize, double scale) : graph() {
		for(uint i = 0; i < arraysize; i++) {
			graph.addNode(Node(xvals[i]*scale, yvals[i]*scale));
		}
		init_subgraphnodes();
		init_axis_sort_array();
	}

	GraphLayout(vector<Point2D> points) : graph() {
		for(uint i = 0; i < points.size(); i++) {
			graph.addNode(points[i]);
		}
		init_subgraphnodes();
		init_axis_sort_array();
	}

	GraphLayout(Graph & g) :
			graph(g),
			subgraphnodes(g.size(), true) {
		subgraphsize = g.size();
		init_axis_sort_array();
	}

	uint graphSize() {
		return graph.size();
	}

	void axis_sort() {
		sort(xsorted.begin(), xsorted.end(), NodeLeft(graph.nodes));
		sort(ysorted.begin(), ysorted.end(), NodeBelow(graph.nodes));
	}

	uint graphBottom() {
		return ysorted.front();
	}

	uint graphLeft() {
		return xsorted.front();
	}

	uint graphRight() {
		return xsorted.back();
	}

	uint graphTop() {
		return ysorted.back();
	}

	Node & node(uint id) {
		return graph.nodes[id];
	}

	uint rightmost() {
		//	xsortedlast = xsorted.end();
		if (xsortedlast == xsorted.end())
			xsortedlast--;
		// skip if the ex-rightmost node is removed from subgraphnodes
		for (; xsortedlast != xsorted.begin(); xsortedlast--) {
			if (subgraphnodes.at(*xsortedlast) )
				break;
		}
		return *xsortedlast;
	}

	uint top() {
		//		ysortedlast = ysorted.end();
		if (ysortedlast == ysorted.end())
			ysortedlast--;
		for (; ysortedlast != ysorted.begin(); ysortedlast--) {
			if (subgraphnodes.at(*ysortedlast) )
				break;
		}
		return *ysortedlast;
	}

	uint rightmostPrevious() {
		rightmost();
		vector<int>::iterator i = xsortedlast;
		for (; i != xsorted.begin();) {
			i--;
			if ( subgraphnodes[*i] )
				break;
		}
		return *i;
	}

	uint topPrevious() {
		top();
		vector<int>::iterator i = ysortedlast;
		for (; i != ysorted.begin();) {
			i--;
			if ( subgraphnodes[*i] )
				break;
		}
		return *i;
	}

	void removeRightmost() {
		subgraphnodes[rightmost()] = false;
		subgraphsize--;
	}

	void removeTop() {
		subgraphnodes[top()] = false;
		subgraphsize--;
	}

	void restoreNode(uint id) {
		if (id < 0 || id >= graph.nodes.size())
			return;
		subgraphnodes[id] = true;
		subgraphsize++;
//		if ( node(*xsortedlast).pos.x < node(id).pos.x ) {
		if (NodeLeft(graph.nodes)(*xsortedlast, id)) {
			//	xsortedlast = xsorted.end();
			xsortedlast = xsorted.begin() + toxsorted[id];
		}
//		if ( node(*ysortedlast).pos.y < node(id).pos.y ) {
		if (NodeBelow(graph.nodes)(*ysortedlast, id)) {
			//	ysortedlast = ysorted.end();
			ysortedlast = ysorted.begin() + toysorted[id];
		}
	}

	const uint graphSize() const {
		return graph.size();
	}

	const unsigned int subgraphSize() const {
		return subgraphsize;
	}

	set<uint> nodesInBoundingBox(Point2D bl, Point2D tr) {
		set<uint> s;
		for (unsigned int i = 0; i < graph.size(); i++) {
			if (graph.nodes[i].inRect(bl, tr))
				s.insert(i);
		}
		return s;
	}

	uint bestGridAlignGap();

	uint bestGridAlignGap(map<Point2D, triple<uint,int,int> > & table) {
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
	}

/*
	void traceBestGridAlignGap(map<Point2D, pair<uint, uint> > & table) {

	}
*/
	//
	static Point2D gridTransfer(Point2D & pt, long leftmargin = 0, long bottommargin = 0, const long gridgap = gridWidth);

	// input & output ;
	friend ostream & operator <<(ostream & stream, const GraphLayout & obj) {
		stream << "GraphLayout ";
		for (unsigned int i = 0; i < obj.subgraphnodes.size(); i++) {
			if ( obj.subgraphnodes[i] )
			stream << i << obj.graph.nodes[i] << ", ";
		}
		return stream;
	}

};

#endif /* GRAPHLAYOUT_H_ */
