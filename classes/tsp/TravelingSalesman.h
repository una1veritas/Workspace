#ifndef header_TravelingSalesman
#define header_TravelingSalesman

#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "../Generic.h"
#include "../Point.h"

#include <vector.h>
#include <set.h>
#include <algo.h>


class xLess {
 public: 
  bool operator() (const pair<int,Point> & p, const pair<int,Point> & q) {
    return p.second.x < q.second.x;
  }
};

class yLess {
 public: 
  bool operator() (const pair<int,Point> & p, const pair<int,Point> & q) {
    return p.second.y < q.second.y;
  }
};


class TravelingSalesman : public Generic {
  
 protected:
  vector< pair<int, Point> > cities;
  
 public:
  TravelingSalesman(int n, Point lowleft, Point upright) {
    int i, x, y;
    cities.reserve(n);
    time_t currtime;
    time(&currtime);
    srand(currtime);
    for (i=0; cities.size() < n; i++) {
      x = lowleft.x + rand() % (upright.x - lowleft.x);
      y = lowleft.y + rand() % (upright.y - lowleft.y);
      // cerr << i << ": " << x << ", " << y << '\n';
      // cerr.flush();
      cities.push_back(pair<int,Point>(i,Point(x,y)));
    }
  }
  
  TravelingSalesman(istream &stream) {
    char tmp[256], *p;
    long n, x, y;

    cerr << "Going to read... "; 
    cerr.flush();
    for (stream >> tmp; strncmp(tmp,"DIMENSION",9) && (! stream.eof());
	 stream >> tmp);
    if ( stream.eof() ) {
      n = 10;
    } else {
      if (tmp[strlen(tmp)-1] == ':') {
	stream >> n;
      } else {
	stream >> tmp;
	stream >> n;
      }
    }
    cerr << n << " cities from input.\n";
    cerr.flush();
    cities.reserve(n);
    for (stream >> tmp; strcmp(tmp,"NODE_COORD_SECTION"); stream >> tmp);
    for (n=0, stream >> tmp; strcmp(tmp,"EOF"); stream >> tmp, n++) {
      stream >> x;
      stream >> y;
      cities.push_back(pair<int,Point>(n,Point(x,y)));
    }
  }

  ~TravelingSalesman() {
    cities.erase(cities.begin(), cities.end());
  }

 protected:
  TravelingSalesman(vector< pair<int,Point> >& supplied) {
    cities = supplied;
  }

 public:
  // accessing;
  const int size(void) const {
    return cities.size();
  }

  const Point city(int i) const {
    return cities[i].second;
  }

  const float tourLength(int * tour) const {
    int i;
    float sum = 0;
    for (i=0; i < cities.size() - 1; i++) 
      sum += city(tour[i]).distanceTo(city(tour[i+1]));
    return sum + city(tour[cities.size() - 1]).distanceTo(city(tour[0]));
  }
  

  // find the bounding box; 
  const pair<Point, Point> boundingBox() const {
    int i;
    Point ll, ur;
    ll = city(0); ur = city(0); 
    for (i=0; i < cities.size(); i++) {
      ll.x = (city(i).x > ll.x)? ll.x : city(i).x;
      ll.y = (city(i).y > ll.y)? ll.y : city(i).y;
      ur.x = (city(i).x > ur.x)? city(i).x : ur.x;
      ur.y = (city(i).y > ur.y)? city(i).y : ur.y;
    }
    // cerr << "Bounding box: " << ll.x << ", " << ll.y << " <-> " 
      // << ur.x << ", " << ur.y << '\n';
    return pair<Point,Point>(ll,ur);
  }

  // finding tour; 
  void divide_and_sort(int * tour, float factor = 3.4) {
    pair<Point,Point> bbox, rect;
    int i, j, swap;
    int divnum, ww, wh, tw, th, direction;
    float tmp;
    struct {
      float x, y; 
    } fll, fur;

    bbox = boundingBox();
    // cerr << "sorting the lists..."; 
    // sort the lists; 
    vector< pair<int,Point> > 
      lx(cities.begin(),cities.end()), 
      ly(cities.begin(),cities.end());
    sort(lx.begin(),lx.end(),xLess() );
    sort(ly.begin(),ly.end(),yLess() );

    // find the patition number; 
    tmp = (float) sqrt(cities.size()) / factor; 
    // cerr << "finding the partition. " << factor << "\n"; 
    // cerr.flush();
    divnum = ( tmp > ((float) (int) tmp) )? (int) (tmp+1) : (int) tmp;
    // cerr << "partition num.: " << divnum << "\n";

    // cerr << "making the tour. (divnum = " << divnum << "). \n"; 
    // cerr.flush();
    // make the tour;
    wh = (bbox.second.y - bbox.first.y) / (2*divnum) 
      + ( ((bbox.second.y - bbox.first.y) % (2*divnum) > 0)? 1 : 0 ); 
    tw = (bbox.second.x - bbox.first.x) / (2*divnum)
      + ( ((bbox.second.x - bbox.first.x) % (2*divnum) > 0)? 1 : 0) ;
    ww = (bbox.second.x - bbox.first.x) - tw;
    th = (bbox.second.y - bbox.first.y);

    rect.second.y = bbox.second.y ;
    rect.second.x = bbox.first.x + ww;
    rect.first.y = bbox.second.y - wh;
    rect.first.x = bbox.first.x;

    // cerr << "doing the rectangles..."; 
    // cerr.flush();
    for (i = 0; i < 2*divnum ; i++) {
      // R_i;
      // cerr << i << "th rect: " << rect.first << " - " << rect.second << "\n";
      // make right-to-left path in rect.first - rect.second;
      if ( i % 2 ) {
	j = 0; direction = 1;
      } else {
	j = cities.size() - 1; direction = -1;
      }
      for ( ; 0 <= j && j < cities.size() ; j = j+direction) {
	if ( city(lx[j].first).isInRect(rect.first,rect.second) ) {
	  *tour = lx[j].first;
	  tour++;
	}
      }
      rect.second.y = rect.first.y - 1; 
      rect.first.y = rect.first.y - wh;
    }
    //  cerr << "doing the exceptional rectangle.\n"; 
    //  cerr.flush();

    // R_2k+1;
    rect.second = bbox.second;
    rect.first.y = bbox.first.y;
    rect.first.x = bbox.first.x + ww + 1;
    // cerr << ++i << "th rect: " << rect.first << " - " 
      // << rect.second << "\n";
    // make bottom-to-top path in rectll - rect.second;
    for (j=0; j < cities.size() ; j++) {
      if ( city(ly[j].first).isInRect(rect.first, rect.second) ) {
	*tour = ly[j].first;
	tour++;
      }
    }
    return;
  }


  void karp_partitioning(int * tour) {
    vector< pair<int,Point> > 
      lx(cities.begin(),cities.end()), 
      ly(cities.begin(),cities.end());
    sort(lx.begin(),lx.end(),xLess() );
    sort(ly.begin(),ly.end(),yLess() );

    make_tour(*this, lx, ly, tour);
  }

  // karp partitioning alg. support; 
  void make_tour(const TravelingSalesman& tsp, 
		 vector< pair<int,Point> >& xsorted,
		 vector< pair<int,Point> >& ysorted, int *tour ) {
    int i, pivot, subtour1[tsp.size()], subtour2[tsp.size()];
    pair< Point, Point > bbox;
    vector<pair<int,Point> > part1, part2;

    if (tsp.size() <= 4) {
      for (i = 0; i < tsp.size(); i++) {
	*tour = tsp.cities[i].first; 
	tour++;
      }
      return;
    }
    bbox = tsp.boundingBox();
    if ( abs(bbox.second.x - bbox.first.x) >
	abs(bbox.second.y - bbox.first.y))  {
      for (i = 0; i < (tsp.size() / 2 + 1); ) {
	part1.push_back(xsorted[i]);
	if (i < (tsp.size() / 2 + 1))
	  i++;
      }
      pivot = i;
      for ( ; i < tsp.size(); i++) {
	part2.push_back(xsorted[i]);
      }
    } else {
      for (i = 0; i < (tsp.size() / 2 + 1); ) {
	part1.push_back(ysorted[i]);
	if (i < (tsp.size() / 2 + 1))
	  i++;
      }
      pivot = i;
      for ( ; i < tsp.size(); i++) {
	part2.push_back(ysorted[i]);
      }
    }
    make_tour(TravelingSalesman(part1), xsorted, ysorted, &subtour1[0]);
    make_tour(TravelingSalesman(part2), xsorted, ysorted, &subtour2[0]);
    // concatenate subtour1 with subtour2 into tour at tsp.city(pivot);
    return;
  }

  void nearest_neighbor(int * tour) {
    int curr, nearest, i, j;
    float distance; 
    char visited[cities.size()];
    int visitednum;

    for (i=0; i < cities.size() ; i++)
      visited[i] = 0;
    curr = 0; 
    visited[curr] = 1;
    visitednum = 1;
    *tour = curr;
    tour++;
    while ( visitednum < cities.size()) {
      nearest = -1; distance = 0;
      for (i = 0; i < cities.size() ; i++) {
	if (! visited[i] )
	  if ( (nearest == -1) ||
	      distance > city(curr).distanceTo(city(i)) ) {
	    nearest = i;
	    distance = city(curr).distanceTo(city(i));
	  }
      }
      *tour = nearest;
      tour++;
      visited[nearest] = 1;
      visitednum++;
      curr = nearest;
    }
    return;
  }


  void doubleSpanningTree(int * tour) {
    int closest[cities.size()];
    set< int,less<int> > nodes;
    set< pair<int,int>,less<pair<int,int> > > edges;
    float min;
    int i, j, next, current, returning;
    
    // find min. spanning tree;
    for (i=0; i < cities.size(); i++) 
      closest[i] = 0;

    nodes.insert(0);
    while ( true ) {
      if (! ( nodes.size() < cities.size() ) ) break;
      min = -1;
      for (i = 0; i < cities.size(); i++) {
	if ( nodes.count(i) == 0 )
	  if ( min == -1 ||
	      city(i).distanceTo(city(closest[i])) < min ) {
	    min = city(i).distanceTo(city(closest[i]));
	    next = i;
	  }
      }
      nodes.insert(next);
      edges.insert(pair<int,int>(next, closest[next]));
      edges.insert(pair<int,int>(closest[next],next));
      for (i = 0; i < cities.size() ; i++) {
	if ( nodes.count(i) == 0 ) {
	  if ( city(i).distanceTo(city(closest[i])) 
	      > city(i).distanceTo(city(next)) ) {
	    closest[i] = next;
	  }
	}
      }
    }
    //cerr << tree << "\n";
    //cerr.flush();
    current = 0;
    for (i=0; i < cities.size(); i++) 
      closest[i] = false;
    *tour = current;
    tour++;
    while ( edges.size() > 0 ) {
      min = -1;
      returning = true;
      for (i = 0; i < nodes.size() ; i++) {
	if (! edges.count(pair<int,int>(current,i)) )
	  continue;
	if ( returning && (edges.count(pair<int,int>(i, current))) ) {
	  returning = false;
	  next = i;
	  min = city(current).distanceTo(city(i));
	  continue;
	}
	if ( min == -1 ) {
	  next = i;
	  min = city(current).distanceTo(city(i));
	  continue;
	}
	if ( min > city(current).distanceTo(city(i)) ) {
	  if (returning) {
	    next = i;
	    min = city(current).distanceTo(city(i));
	  } else {
	    if ( edges.count(pair<int,int>(i,current)) ) {
	      next = i;
	      min = city(current).distanceTo(city(i));
	    }
	  }
	}
      }
      if ( !returning ) {
	*tour = next;
	tour++;
	// cout << current << " -> " << next << ", "; 
	// cout.flush();
      } else {
	// cout << "(" << current << " -> " << next << "), "; 
	// cout.flush();
      }
      edges.erase(pair<int,int>(current,next));
      current = next;
    }    
    //    cerr << "\n";
    /* delete closest; */
    return;
  }


  
 public:
  // printing;
  ostream& printOn(ostream& stream) const {
    vector< pair<int,Point> >::const_iterator i;
    stream << "totally " << cities.size() << " cities;\n";
    for (i = cities.begin(); i != cities.end() ; i++) {
      stream << (*i).first << ", " << (*i).second << '\n';
    }
    return stream; 
  }
  
};

#endif
