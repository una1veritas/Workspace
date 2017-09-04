/********************************************************************

               aaaa
                 /\
             no /  \ yes           aaaa(yes[,],bbb(yes[,],no[,]))
               /    \
              1     bbb
                     /\
                 no /  \ yes
                   /    \
                  1      0


 ********************************************************************/

#include <ctype.h>
#include "math.h"
#include "Tree.h"

#define log2(n) (log(n)/log(2.))
#define min(a,b) ((a > b) ? b : a)
#define i_p_n(p,n) (( p + n ) ? ( -1.0 / (p + n) * ((p ? p * log2(p/(p+n)) : 0.0) + (n ? n * log2(n/(p+n)) : 0.0))) : 0.0 )

 Tree::Tree(String& labl, Tree& lchild, Tree& rchild) {
   
   label = new String(labl);
   isLeaf = false;
   left = &lchild;
   right = &rchild;
 }

 Tree::Tree(String& labl) {
   
   label = new String(labl);
   isLeaf = true;
 }

 Tree::Tree(char * * src) {
   int length;
   String labl;
   Tree* lc, * rc;
   
   for (length = 0;
	**src && isgraph(**src) &&
	**src != '('&& **src != ',' && **src != ')';
	++*src, ++length) {
     label[length] = **src;
   }
   label[length] = '\0';
   
   if (**src == '(') {
     ++*src; /* skip '(' */
     lc = new Tree(src);
     if (**src != ',') {
       /* syntax error! */
       return (Tree*) NULL;
     }
     ++*src; /* skip ',' */
     rc = ScanTree(src);
     if (**src != ')') {
       /* syntax error! */
       return (Tree*) NULL;
     }
     ++*src; /* skip ')' */
   }
   return new Tree(labl,lc,rc);
 }

const String& Tree::decide(const String& str) {

  if (isLeaf)
    return *label;
  if (label.contains(str)) {
    return right->decide(str);
  } else {
    return left->decide(str);
  }
}

ostream & printOn(ostream & stream) const {

  stream << label;
  if (isLeaf) {
    return stream;
  }
  stream << '(' << (*left) << ',' << (*right) << ')';
  return stream;
}


 Tree::Tree(Bag& pos, String& pname, Bag& neg, String& nname) {
   Tree* dtree;
   Bag * posm, * negm, * posu, * negu;
   String pttn;

   if (! pos.size())
     return Tree(nname);
   if (! neg.size())
     return Tree(pname);
   findBestPattern(pos,neg,7,pttn);
   if (pttn.length()) {
     posm = new Bag;
     negm = new Bag;
     posu = new Bag;
     negu = new bag;
     newpartition(pttn,pos,neg,posm,negm,posu,negu);
     dtree = new Tree(pttn, new Tree(posu,pname,negu,nname),
		      new Tree(posm,pname,negm,nname));
     delete(posm);
     delete(negm);
     delete(posu);
     delete(negu);
   } else {
     // Can not find patterns any more. ;
     dtree = new Tree(nname);
   }
   label = dtree.label;
   isLeaf = dtree.isLeaf;
   left = dtree.left;
   right = dtree.right;

   return dtree;
}


void Tree::findBestPattern(Bag& p, Bag& n, int pttnbnd, String& bestpttn) {
  int i;
  float infgain, bestinfgain;
  String pttn;
  unsigned long data;
  Array* array, * ex;
  int pttnstart, pttnlen, bestlen;
  float pmc, puc,nmc, nuc;
  Set* set;
  
  set = new Set(1024);
  bestpttn = "";

  array = new Array(p);
  for (data = 0; data < array->size() ; data++) {
    for (pttnstart = 0; pttnstart < (array[data])->length(); pttnstart++ ) {
      for (pttnlen = min((array[data])->length(),pttnbnd); pttnlen; pttnlen--){
	pttn = (array[data])->copyFrom_to(pttnstart,pttndtstart+pttnlen);

	if (! set->includes(pttn)) {
	  set->add(pttn);

	  for (pmc = puc = 0, ex = p; *ex; ex++)
	    if (strstr((char *) **ex, (char *) pttn))
	      pmc++;
	    else
	      puc++;
	  for (nmc = nuc = 0, ex = n; *ex; ex++)
	    if (strstr((char *) **ex, (char *) pttn))
	      nmc++;
	    else
	      nuc++;
	  if ((pmc || nmc) && (puc || nuc)) {
	    infgain = ((pmc+nmc)*i_p_n(pmc,nmc) + (puc+nuc)*i_p_n(puc,nuc))
	      /(pmc+puc+nmc+nuc) + 1.0;
	    if (infgain < bestinfgain ||
		(infgain == bestinfgain && pttnlen < bestlen)
		|| (!*bestpttn)) {
	      bestinfgain = infgain;
	      bestlen = pttnlen;
	      strcpy((char *) bestpttn, (char *) pttn);
	    }
	  }

	}

      }
    }
  }

  for (data = n; *data ; data++) {
    for (pttnstart = 0; pttnstart < strlen((char *) **data); pttnstart++ ) {
      strcpy((char *) pttn, (char *) **data+pttnstart);
      for (pttnlen = min(strlen((char *) pttn),pttnbnd); pttnlen; pttnlen--){
	pttn[pttnlen] = '\0';

	if (!isinSet(set,pttn)) {
	  addtoSet(set,pttn);

	  for (pmc = puc = 0, ex = p; *ex; ex++)
	    if (strstr((char *) **ex, (char *)pttn))
	      pmc++;
	    else
	      puc++;
	  for (nmc = nuc = 0, ex = n; *ex; ex++) 
	    if (strstr((char *) **ex, (char *) pttn))
	      nmc++;
	    else
	      nuc++;
	  if ((pmc || nmc) && (puc || nuc)) {
	    infgain = ((pmc+nmc)*i_p_n(pmc,nmc) + (puc+nuc)*i_p_n(puc,nuc))
	      /(pmc+puc+nmc+nuc) + 1.0;
	    if (infgain < bestinfgain ||
		(infgain == bestinfgain && pttnlen < bestlen)
		|| (!*bestpttn)) {
	      bestinfgain = infgain;
	      bestlen = pttnlen;
	      strcpy((char *) bestpttn, (char *) pttn);
	    }
	  }

	}

      }
    }
  }

  deleteSet(set);

}


void partition(Str255 str, Str255 * pos[], Str255 * neg[],
	       Str255 * posm[], Str255 * negm[],
	       Str255 * posu[], Str255 * negu[]) {

  for (; *pos; ++pos)
    if (strstr((char *) **pos, (char *) str)) {
      *posm = *pos;
      posm++;
    } else {
      *posu = *pos;
      posu++;
    }
  *posm = NULL;
  *posu = NULL;

  for (; *neg; neg++) 
    if (strstr((char *) **neg, (char *) str)) {
      *negm = *neg;
      negm++;
    } else {
      *negu = *neg;
      negu++;
    }
  *negm = NULL;
  *negu = NULL;}


void DisposeTree(Tree t) {
  if (!(t->isLeaf))  {
    DisposeTree(t->left);
    DisposeTree(t->right);
  }
  free(t);
}

int nodesof_Tree(Tree t) {
  if (t->isLeaf)
    return 0;
  else 
    return 1 + nodesof_Tree(t->left)+nodesof_Tree(t->right);
}
