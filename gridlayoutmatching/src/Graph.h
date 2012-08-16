/*
 * Graph.h
 *
 *  Created on: 2012/04/26
 *      Author: sin
 */

#ifndef GRAPH_H_
#define GRAPH_H_

#include <time.h>

#include <vector>
#include <string>
#include <set>

#include "mytypes.h"
#include "Point2D.h"

using namespace std;

class Node : public Point2D {
	string label;

public:
	Node(const int px, const int py) : Point2D(px,py), label("") {
	}

	Node(string lab, const int px, const int py) : Point2D(px, py), label(lab) {
	}

	Node(const Point2D & p) : Point2D(p), label("") {}

	// input & output ;
	friend ostream & operator <<(ostream & stream, const Node & obj) {
		return obj.printOn(stream);
	}
};

class Edge {
	Node & start, & end;
};

class Graph {
public:
	vector<Node> nodes;
	set<Edge> edges;

public:
	// The empty graph.
	Graph() :
			nodes(), edges() {
	}

	const uint size() const {
		return nodes.size();
	}

	void addNode(const Node & p) {
		nodes.push_back(p);
	}

	void addNodes(const set<Node> & s) {
		for (set<Node>::iterator i = s.begin(); i != s.end(); i++) {
			addNode(*i);
		}
	}

	void remove(const unsigned int i) {
		nodes.erase(nodes.begin()+i);
	}

	/*
	const unsigned int rightmost() const {
		if ( nodes.size() == 0 )
			return 0;
		unsigned int index = 0;
		for(unsigned int i = 0; i < nodes.size() ; i++ ) {
			if ( nodes[index].x < nodes[i].x )
				index = i;
		}
		return index;
	}

	const unsigned int leftmost() const {
		if ( nodes.size() == 0 )
			return 0;
		unsigned int index = 0;
		for(unsigned int i = 0; i < nodes.size() ; i++ ) {
			if ( nodes[index].x > nodes[i].x )
				index = i;
		}
		return index;
	}
*//*
	const unsigned int top() const {
		if ( nodes.size() == 0 )
			return 0;
		unsigned int index = 0;
		for(unsigned int i = 0; i < nodes.size() ; i++ ) {
			if ( nodes[index].y < nodes[i].y )
				index = i;
		}
		return index;
	}

	const unsigned int bottom() const {
		if ( nodes.size() == 0 )
			return 0;
		unsigned int index = 0;
		for(unsigned int i = 0; i < nodes.size() ; i++ ) {
			if ( nodes[index].y > nodes[i].y )
				index = i;
		}
		return index;
	}
*//*
	Point2D bottomleft() {
		return Point2D(nodes[leftmost()].x, nodes[bottom()].y);
	}

	Point2D topright() {
		return Point2D(nodes[rightmost()].x, nodes[top()].y);
	}
*/
	// printing;
	ostream& printOn(ostream& stream) const {
		stream << "Graph ";
		for (unsigned int i = 0; i < nodes.size(); i++) {
			stream << nodes[i] << ", ";
		}
		return stream;
	}

	// input & output ;
	friend ostream & operator <<(ostream & stream, const Graph & obj) {
		return obj.printOn(stream);
	}

};

class NodeLeft {
	vector<Node> & nodes;
public:
	NodeLeft(vector<Node> & nodelist) : nodes(nodelist){	}

	bool operator()(const unsigned int & ip, const unsigned int & iq) const {
		return nodes[ip].x < nodes[iq].x ||
				(nodes[ip].x == nodes[iq].x && nodes[ip].y < nodes[iq].y);
	}
};

class NodeBelow {
	vector<Node> & nodes;
public:
	NodeBelow(vector<Node> & nodelist) : nodes(nodelist) {	}

	bool operator()(const unsigned int & ip, const unsigned int & iq) const {
		return nodes[ip].y < nodes[iq].y ||
				(nodes[ip].y == nodes[iq].y && nodes[ip].x < nodes[iq].x);
	}
};


#endif /* GRAPH_H_ */
