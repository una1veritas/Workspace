#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include <functional>
#include <list>

#include <pthread.h>

#define DEFINE_STREAM_PRINT(T) \
  template<typename TYPE>\
  std::ostream &operator <<(std::ostream &out, T<TYPE> &p) { \
    p.print(out); return out; \
  }

template<typename TYPE>
class Point {
  TYPE x, y;
public:
  Point() : x(0), y(0) {}
  Point(TYPE x, TYPE y) : x(x), y(y) {}
  Point(const Point &p) : x(p.x), y(p.y) {}
  ~Point() {}
#define GETTER(T,N,V) T get ## N () const { return V ; }
#define SETTER(T,N,V) void set ## N (T V) { this-> V = V; }
  GETTER(TYPE,X,x) GETTER(TYPE,Y,y)
  SETTER(TYPE,X,x) SETTER(TYPE,Y,y)
#undef GETTER
#undef SETTER
#define UV(A) \
  TYPE u##A = (std::abs(t.A) % grid) * (t.A > 0 ? -1 : 1); \
  TYPE v##A = (grid - std::abs(u##A)) * (t.A > 0 ? 1 : -1);
#define CHECK(X,Y) \
  tmp = Point(X,Y); \
  if(t.x+tmp.x <= rt2.x && t.y+tmp.y <= rt2.y) tmp.x+=grid, tmp.y+=grid; \
  if(t.x+tmp.x > rt2.x && t.y+tmp.y <= rt2.y) tmp.y+=grid; \
  if(t.x+tmp.x <= rt2.x && t.y+tmp.y > rt2.y) tmp.x+=grid; \
  if(tmp.x > 0 && tmp.y > 0) r = tmp;
  Point trans(Point q, Point rt, int grid) {
    Point t(*this - q), rt2(rt - q), r, tmp; UV(x) UV(y)
    r = Point(ux,uy); CHECK(vx,uy) CHECK(ux,vy) CHECK(vx,vy)
    //std::cerr << "TRANS " << *this << " / " << r << std::endl;
    return r;
  }
#undef UV
#undef CHECK
  TYPE sum() const { return x+y; }
  TYPE length() const { return abs(x)+abs(y); }
  bool operator ==(Point &p) { return (x == p.x) && (y == p.y); }
  Point &operator =(Point p) { x=p.x; y=p.y; return *this; }
  Point operator +(Point &p) { Point q(x + p.x, y + p.y); return q; }
  Point &operator +=(Point p) { x += p.x; y += p.y; return *this; }
  Point operator -(Point &p) { Point q(x - p.x, y - p.y); return q; }
  void print(std::ostream &out) { out << x << " " << y; }
  int checkPair(Point &q) {
    if(this == &q) return 1;
    if(x >= q.x && y <= q.y) return 2;
    return 0;
  }
};
DEFINE_STREAM_PRINT(Point)

template<typename TYPE>
class PointSet : public std::vector<Point<TYPE> > {
  typedef Point<TYPE> POINT_T;
  typedef typename std::vector<POINT_T>::iterator iterator;
public:
#define BEGIN std::vector<POINT_T>::begin()
#define END std::vector<POINT_T>::end()
#define SIZE std::vector<POINT_T>::size()
#define AT(I) std::vector<POINT_T>::at(I)
  PointSet() : std::vector<POINT_T>() {}
  ~PointSet() {}
  PointSet(std::istream &in) : std::vector<POINT_T>() {
    TYPE x, y; in >> x, in >> y;
    while(!in.eof()) push_back(POINT_T(x,y)), in >> x, in >> y;
  }
  PointSet(PointSet &set) : std::vector<POINT_T>(set.size()) {
    for(int i=0; i<set.size(); ++i) { std::vector<POINT_T>::at(i) = set[i]; }
  }
  PointSet(int size) : std::vector<POINT_T>(size) {}
  void print(std::ostream &out) {
    for(iterator i = BEGIN; i != END; i++) i->print(out), out << std::endl;
  }
  bool checkIndependent() {
    for(iterator i = BEGIN; i != END; i++)
      for(iterator j = BEGIN; j != END; j++)
        if(!(i==j) && (*i==*j)) return false;
    return true;
  }
  PointSet &operator=(PointSet &set) {
    for(int i=0; i<SIZE; ++i) AT(i) = set[i];
    return *this;
  }
#undef BEGIN
#undef END
#undef SIZE
#undef AT
};
DEFINE_STREAM_PRINT(PointSet)

template<typename TYPE>
class PointSequence : public std::vector<int> {
  typedef typename std::vector<int>::iterator iterator;
#define DEFINE_COMPARE(A,B) \
  class compare_##A { \
    PointSet<TYPE> &set; \
  public: \
    compare_##A (PointSet<TYPE> &set) : set(set) {} \
    bool operator ()(const TYPE &a, const TYPE &b) const { \
      return set[a].get##A() < set[b].get##A() || \
             ( set[a].get##A() == set[b].get##A() && \
               set[a].get##B() < set[b].get##B() ); \
    } \
  };
  DEFINE_COMPARE(X,Y)
  DEFINE_COMPARE(Y,X)
#undef DEFINE_COMPARE
#define BEGIN std::vector<int>::begin()
#define END std::vector<int>::end()
  PointSet<TYPE> &set;
public:
  ~PointSequence() {}
  PointSequence(PointSet<TYPE> &set, bool sortx)
  : std::vector<int>(set.size()), set(set) {
    int j=0;
    for(iterator i=BEGIN; i!=END; ++i) { *i = j++; }
    if(sortx) std::sort(BEGIN, END, compare_X(set));
    else std::sort(BEGIN, END, compare_Y(set));
  }
  Point<TYPE> &ref(int i) { return set.at(at(i)); }
  void print(std::ostream &out) {
    for(iterator i=BEGIN; i!=END; ++i) out << *i << " : " << set[*i] << std::endl;
  }
#undef BEGIN
#undef END
};
DEFINE_STREAM_PRINT(PointSequence)

template <typename TYPE>
class grid : public std::vector<TYPE> {
  typedef typename std::vector<TYPE>::iterator iterator;
  int grid_size;
public:
  grid() : std::vector<TYPE>(), grid_size(0) {}
  grid(int size) : std::vector<TYPE>(size*size), grid_size(size) {}
  TYPE &ref(int x, int y) { return std::vector<TYPE>::at(x + y * grid_size); }
  void print(std::ostream &out) {
    int j=0;
    for(iterator i=std::vector<TYPE>::begin(); i!=std::vector<TYPE>::end(); i++) {
      out << "  " << *i << std::endl;
      if(++j % grid_size == 0) out << std::endl;
    }
  }
};
DEFINE_STREAM_PRINT(grid)

template <typename TYPE>
class CalcDistanceArgments {
public:
  int grid_size;
  PointSequence<TYPE> &x, &y;
  grid< Point<TYPE> > &tt;
  grid<TYPE> &sum;
  grid<int> &dir;
  CalcDistanceArgments(PointSequence<TYPE> &x,
                       PointSequence<TYPE> &y,
                       grid< Point<TYPE> > &tt,
                       grid<int> &dir,
                       grid<TYPE> &sum,
                       int grid_size)
  : x(x), y(y), tt(tt), dir(dir), sum(sum), grid_size(grid_size) {}
};

template <typename TYPE>
class CalcRectangleDistance {
  typedef Point<TYPE> POINT_T;
  typedef PointSequence<TYPE> PSEQ_T;
  typedef typename std::list< Point<int> >::iterator itr;
  itr s, e;
  CalcDistanceArgments<TYPE> *dt;
#define DEFINE_PREV(A1,a1,A2,a2) \
  int prev##A1(int ix, int iy, PSEQ_T &x, PSEQ_T &y) { \
    int t = a2.ref(i##a2).get##A2(); \
    do { if(i##a1 <= 0) return -1; } while(t < a1.ref(--i##a1).get##A2()); \
    return i##a1; \
  }
  DEFINE_PREV(X,x,Y,y)
  DEFINE_PREV(Y,y,X,x)
#undef DEFINE_PREV
public:
  CalcRectangleDistance(itr s, itr e, void *data)
  : s(s), e(e), dt((CalcDistanceArgments<TYPE>*)data) {}
  ~CalcRectangleDistance() {}
  void run() {
    int i, j, px, py, gsize = dt->grid_size;
    TYPE r,t,alen,blen;
    POINT_T a,b,prea,preb,da,db;
    PointSequence<TYPE> &x = dt->x;
    PointSequence<TYPE> &y = dt->y;
    grid<POINT_T> &tt = dt->tt;
    grid<int> &dir = dt->dir;
    grid<TYPE> &sum = dt->sum;
    do {
#define CASE_TEMPLATE(ID,CODE) \
  case ID : { \
    px = prevX(i,j,x,y), py = prevY(i,j,x,y); \
    CODE \
  } break;
#define CALC_AXIS(A,IA,X,IX,IY,LOGIC) \
  X = (LOGIC) ? POINT_T() : tt.ref(IX,IY); \
  pre##X = (LOGIC) ? POINT_T() : ((dir.ref(IX,IY)!=2) ? x.ref(IX) : y.ref(IY)); \
  r = IX < 0 ? 0 : x.ref(IX).getX(); \
  t = IY < 0 ? 0 : y.ref(IY).getY(); \
  d##X = A.ref(IA).trans(pre##X, POINT_T(r,t), gsize); \
  X##len = d##X.length() + ((LOGIC) ? 0 : sum.ref(IX,IY));
#define SET_TABLE(T,S,D) tt.ref(i,j)=T, sum.ref(i,j)=S, dir.ref(i,j)=D;
      i = s->getX(), j = s->getY();
      switch(x.ref(i).checkPair(y.ref(j))) {
        CASE_TEMPLATE(1,
          CALC_AXIS(x,i,a,px,py,px<0||py<0)
          SET_TABLE(da,alen,1))
        CASE_TEMPLATE(2,
          CALC_AXIS(x,i,a,px,j,px<0)
          CALC_AXIS(y,j,b,i,py,py<0)
          if(alen < blen) SET_TABLE(da,alen,1)
          else SET_TABLE(db,blen,2))
      }
      //std::cerr << "calc : " << i << " " << j << std::endl;
#undef CASE_TEMPLATE
#undef CALC_AXIS
#undef SET_TABLE
    } while(s++ != e);
  }
};

template <typename TYPE, typename PREDICATE>
class RectangleTable {
  typedef std::list< Point<int> > RLST_T;
  typedef std::vector< std::list< Point<int> > > RDIC_T;
  typedef std::vector<int> RCNT_T;

  RDIC_T dictionary;
  RCNT_T sizetable;
public:
  RectangleTable(int size) : dictionary(size), sizetable(size) {}
  ~RectangleTable() {}
  void push(int ix, int iy, int count) {
    dictionary[count].push_front(Point<int>(ix,iy));
    sizetable[count] += 1;
  }
  int size(int count) {
    return sizetable(count);
  }
  void per_set(int count, void *data) {
    for(typename RLST_T::iterator i=dictionary[count].begin(); i!=dictionary[count].end(); ++i) {
      PREDICATE pred(i,i,data);
      pred.run();
    }
  }
  void per_set_parallel(int count, int delimite, void *data) {
    int j=0;
    void *ret = 0;
    std::list<PREDICATE> predl;
    typename RLST_T::iterator s, i;
    for(i=dictionary[count].begin(); i!=dictionary[count].end(); ++i) {
      if(j==0) s=i;
      if(j==delimite) {
        predl.push_front(PREDICATE(s,i,data));
        j=0;
      }
      ++j;
    }
    if(j > 0) {
      predl.push_front(PREDICATE(s,--i,data));
      j=0;
    }
    std::vector<pthread_t> threads(predl.size());
    int ii=0;
    for(typename std::list<PREDICATE>::iterator i=predl.begin(); i!= predl.end(); ++i) {
      if(pthread_create(&(threads[ii]), NULL, start_predicate, &(*i))) {
        //std::cerr << "Thread Error" << std::endl;
      }
      ++ii;
    }
    for(typename std::vector<pthread_t>::iterator i=threads.begin(); i!=threads.end(); ++i) {
      pthread_join(*i, NULL);
      //std::cerr << "Thread Joined" << std::endl;
    }
  }

  static void *start_predicate(void *pred) {
    PREDICATE *p = (PREDICATE*)pred;
    p->run();
    return NULL;
  }
};

template <typename TYPE>
class GridLayout {
  typedef Point<TYPE> POINT_T;
  typedef PointSet<TYPE> PSET_T;
  typedef PointSequence<TYPE> PSEQ_T;
  typedef RectangleTable<TYPE, CalcRectangleDistance<TYPE> > RDIC_T;
  int grid_size;
  int checkPointPair(POINT_T &p, POINT_T &q) {
    if(p == q) return 1;
    if(p.getX() >= q.getX() && p.getY() <= q.getY()) return 2;
    return 0;
  }
#define UNDIPLICATE_AXIS(A,a) \
  prev = a.ref(0).get##A(), bias = 0; \
  for(int i=1; i<set.size(); ++i) { \
    if(a.ref(i).get##A() + bias == prev) bias += grid_size; \
    a.ref(i).set##A(a.ref(i).get##A() + bias); \
    prev = a.ref(i).get##A(); \
  }
  void undiplicate(PSET_T &set, PSEQ_T &x, PSEQ_T &y) {
    TYPE prev, bias;
    UNDIPLICATE_AXIS(X,x)
    UNDIPLICATE_AXIS(Y,y)
  }
#undef UNDIPLICATE_AXIS
#define DEFINE_PREV(A1,a1,A2,a2) \
  int prev##A1(int ix, int iy, PSEQ_T &x, PSEQ_T &y) { \
    int t = a2.ref(i##a2).get##A2(); \
    do { if(i##a1 <= 0) return -1; } while(t < a1.ref(--i##a1).get##A2()); \
    return i##a1; \
  }
  DEFINE_PREV(X,x,Y,y)
  DEFINE_PREV(Y,y,X,x)
#undef DEFINE_PREV
#define CASE_TEMPLATE(ID,CODE) \
  case ID : { \
    px = prevX(i,j,x,y), py = prevY(i,j,x,y); \
    CODE \
  } break;
#define PCOUNT(V,T,X,Y) V = (T) ? 0 : cr.ref(X,Y)
#define SETCOUNT(V) cr.ref(i,j) = (V)+1; rd.push(i,j,V);
#define EACH_CELL(X,Y) for(int X=0; X<set.size(); ++X) for(int Y=0; Y<set.size(); ++Y)
  void listCountRects(PSET_T &set, PSEQ_T &x, PSEQ_T &y) {
    grid<int> cr(set.size());
    RDIC_T rd(set.size());
    int px, py, a, b;
    EACH_CELL(i,j) {
      switch(checkPointPair(x.ref(i), y.ref(j))) {
        CASE_TEMPLATE(1,
          PCOUNT(a, px<0 || py<0, px, py);
          SETCOUNT(a);)
        CASE_TEMPLATE(2,
          PCOUNT(a, px<0, px, j);
          SETCOUNT(a);)
      }
    }
    calcTranslateTable(set, x, y, rd);
  }
#undef CASE_TEMPLATE
#undef PCOUNT
#undef SETCOUNT
#undef EACH_CELL
#define GRID(T, N) grid<T> N(set.size());
  void calcTranslateTable(PSET_T &set, PSEQ_T &x, PSEQ_T &y, RDIC_T &rd) {
    GRID(POINT_T, tt) GRID(TYPE, sum) GRID(int, dir)
    CalcDistanceArgments<TYPE> args(x,y,tt,dir,sum,grid_size);
    for(int i=0; i<set.size(); ++i) {
      rd.per_set(i, (void*)&args);
      //rd.per_set_parallel(i, 10000000, (void*)&args);
    }
    applyLayout(set,x,y,tt,dir);
  }
#undef GRID
#define CASE_TEMPLATE(ID,CODE) \
  case ID : { \
    px = prevX(ix,iy,x,y), py = prevY(ix,iy,x,y); \
    CODE \
  } break;
#define PUSH(A) ap.push_front(&tmp[A[i##A]])
  void applyLayout(PSET_T &set, PSEQ_T &x, PSEQ_T &y, grid<POINT_T> &tt, grid<int> dir) {
    std::list<POINT_T*> ap;
    int ix = set.size()-1, iy = set.size()-1, px = 0, py = 0;
    POINT_T trans;
    PSET_T tmp(set);
    while(ix >= 0 && iy >= 0) {
      trans = tt.ref(ix, iy);
      if(!dir.ref(ix,iy)) return;
      switch(checkPointPair(x.ref(ix), y.ref(iy))) {
        CASE_TEMPLATE(1,
          (PUSH(x),ix=px,iy=py);)
        CASE_TEMPLATE(2,
          if(dir.ref(ix,iy)!=2) (PUSH(x),ix=px);
          else (PUSH(y),iy=py);)
      }
      //std::cerr << "APPLY : " << *ap.front() << " << " << trans << std::endl;
      for(typename std::list<POINT_T*>::iterator i=ap.begin(); i!=ap.end(); ++i) { *(*i) += trans; }
    }
    set = tmp;
  }
#undef CASE_TEMPLATE
#undef PUSH
public:
  GridLayout(int grid_size) : grid_size(grid_size) {}
  ~GridLayout() {}
  PSET_T &match(PSET_T &p) {
    PSEQ_T sortx(p, true), sorty(p, false);
    undiplicate(p, sortx, sorty);
    listCountRects(p, sortx, sorty);
    return p;
  }
  bool checkMatch(PSET_T &p) {
    for(int i=0; i<p.size(); ++i)
      if(p[i].getX() % grid_size != 0 || p[i].getY() % grid_size != 0)
        return false;
    return true;
  }
};

template<typename TYPE>
bool ApplyGridLayout(int grid, std::istream &in, std::ostream &out) {
  GridLayout<TYPE> gridlayout(grid);
  PointSet<TYPE> pointset(in);
  if(!pointset.checkIndependent()) return false;
  if(gridlayout.checkMatch(pointset)) return false;
  gridlayout.match(pointset);
  out << pointset;
  if(!pointset.checkIndependent()) {
    std::cerr << "NOT INDEPENDENT." << std::endl;
    return false;
  }
  if(!gridlayout.checkMatch(pointset)) {
    std::cerr << "NOT MATCHING." << std::endl;
    return false;
  }
  return true;
}

int main(int argc, char **argv) {
  if(argc != 2) {
    std::cout << "USAGE [layout.exe grid_size/int]" <<std::endl;
    return 0;
  }
  ApplyGridLayout<int>(atoi(argv[1]), std::cin, std::cout);
  return 0;
}

