#ifndef header_Graph
#define header_Graph

#include "Generic.h"
#include "Set.h"

class Edge : public Generic {

 protected:
  const Generic * start, * end;

 public:
  Edge(const Generic & from, const Generic & to) {
    start = &from;
    end = &to;
  }

  const Boolean isEqualTo(const Generic & obj) const { 
    Edge * e;
    e = (Edge* ) &obj;
    return (const Boolean) (start == e->start && end == e->end);
  }

  const unsigned long hash() const {
    return (start->hash()^end->hash());
  }

  // printing;
  ostream& printOn(ostream& stream) const {
    stream << " (" <<  start << ", " << end << ")" ;
    return stream; 
  }
  
};

class Graph : public Generic {
 
 protected:
  Set * nodes;
  int node_limit;
  Set * edges;

 public:
  Graph(int sz) {
    int i, j;
    node_limit = sz;
    nodes = new Set(node_limit);
    edges = new Set(node_limit+1);
  }

  Graph(Set & nodeset) {
    int i;
    node_limit = nodeset.size();
    nodes = &nodeset;
    edges = new Set(node_limit+1);
  }

  ~Graph() {
    delete nodes;
    delete edges;
  } 

  const Generic & addNode(const Generic & node) {
    if (! nodes->includes(node) ) {
      nodes->add(node);
    }
    return node;
  }

  const Generic & addEdge(const Generic & from, const Generic & to) {
    if (! edges->includes(Edge(from,to)) ) {
      edges->add(*(new Edge(from,to)));
    }
    return from;
  }

  const Generic removeEdge(const Generic & from, const Generic & to) {
    if ( edges->includes(Edge(from,to)) ) {
      edges->remove(Edge(from,to));
    }
    return from;
  }

  int nodeSize(void) {
    return nodes->size();
  }

  int edgeSize(void) {
    return edges->size();
  }

  int hasNode(const Generic & node) {
    return nodes->includes(node);
  }

  int hasEdge(const Generic & from, const Generic & to) {
    return edges->includes(Edge(from,to));
  }

  // printing;
  ostream& printOn(ostream& stream) const {
    int i, j, k;
    Edge ** edge_list;
    edge_list = (Edge **) edges->elementArray();
    stream << "Graph( " << *nodes ;
    stream << ", (";
    for (i=0; i < edges->size(); i++) 
      stream << *(edge_list[i]) << ", ";
    stream << ") )";
    return stream; 
  }

};

#endif
