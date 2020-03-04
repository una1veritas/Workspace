/*きっちり5段階で比較*/
#include <iostream> 
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cctype>
#include <cstring>

#include <vector>
#include <array>
#include <filesystem>

#include "dirlister_unix.h"
#include "libsmf/SMFEvent.h"
#include "libsmf/SMFStream.h"

std::vector<int> f;
int count = 0;

typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;



void loopcheck(char b[]){

  int i = 0;
  int j = 1;

  f.resize(strlen(b)+1);

  f[1] = 0;
  while(b[j] != '\0'){
    if(i == 0 || b[i] == b[j])
      f[++j] = ++i;
    else
      i = f[i];
  }
}


int search(const char a[], char b[]){
  int i = 0;
  int j = 0;
  int frag = 0; // 1 まっちんぐちゅう。

  while(b[j] != '\0'){
    //printf("1 = %c",a[i]);
    //printf(" 2 = %c\n",b[j]);
    if(a[i] == '\0'){
      //printf("not same \n");
      return 0;
    }
    
    if(a[i] != b[j] && frag == 0){
      i++;
    }
    else if(a[i] != b[j] && frag == 1){
      if(f[i] > 1) {
	/*printf("j=%d, i=%d, f[i]=%d\n", j, i, f[i]);*/
	j =f[j];
      } else
	j = 0;
      frag = 0;
    }
    else{
      i++;
      j++;
      frag = 1;
    }
    
  }
  
  //printf("same \n");
  //  allfrag = 1; 
  return 1;
  
}


std::array<std::vector<char>, 16> change(const char *filename) {
	std::fstream infile;
	std::vector<char> interval;
	std::array<std::vector<char>, 16> allinterval;

	infile.open(filename, (std::ios::in | std::ios::binary));
	if (!infile) {
		std::cerr << "失敗" << std::endl;
		return allinterval;
	}

	SMFStream smf(infile);
	count = count + 1;

	uint32 delta_total = 0, last_total;
	int last_number, latest_number;
	for (int i = 0; i <= 15; i++) {
		interval.clear();
		smf.reset();
		last_number = 0;
		latest_number = 0;
		while (smf.smfstream) {
			SMFEvent evt = smf.getNextEvent();
			if (evt.delta > 0)
				delta_total += evt.delta;
			if (evt.isMTRK()) {
				delta_total = 0;
				last_total = (uint32) -1;
			}

			if (evt.isNoteOn()) {
				/*	if ( last_total != delta_total ) {
				 std::cout << std::endl << std::dec << delta_total << " ";
				 last_total = delta_total;
				 }
				 */

				if (evt.channel() == i) {

					if (last_total != delta_total) {
						if (last_number > evt.number)
						  if(last_number - evt.number > 2)
							interval.push_back('-');
						  else
						    interval.push_back('b');

						else if (last_number == evt.number)
							interval.push_back('=');
						else
						  if(evt.number -last_number > 2)
							interval.push_back('+');
						  else
						    interval.push_back('#');
						
						latest_number = last_number;
						last_number = evt.number;
						last_total = delta_total;
					}

					else { /*和音の時*/
						if (interval.size() != 0) {
							if (last_number < evt.number) {
								interval.pop_back();
								if (latest_number > evt.number)
								  if(latest_number - evt.number > 2)
								    interval.push_back('-');
								  else
								    interval.push_back('b');
								else if (latest_number == evt.number)
									interval.push_back('=');
								else
								  if(evt.number -latest_number > 2)
								    interval.push_back('+');
								  else
								    interval.push_back('#');
							}
							last_number = evt.number;

						}

					}

				}

				//std::cout << evt << " ";
			}

		}
		allinterval[i] = interval;
		//printf("%d\n",i);

	}
	for(int i = 0; i < 16; ++i) {
		if (allinterval[i].size() > 0) {
			allinterval[i][0] = '.';
		}
	}
	return allinterval;
}


  
int main(int argc, char **argv) {

  std::array<std::vector<char>, 16> allinterval;
  dirlister lister("./");
  int counter = 0;
  if ( ! lister() ) {
    std::cerr << "error: opendir returned a NULL pointer." << std::endl;
    exit(1);
  }

  // char b[] = "+-#+=";
  char b[64] = "+==b#===b";
  printf("pattern length = %lu\n", strlen(b));
  loopcheck(b);
  std::cout << "failure: ";
  for(unsigned int i = 0; i < f.size(); ++i) {
	  std::cout << f[i] << ", ";
  }
  std::cout << std::endl;

  if ( argc > 2) {
	  strcpy(b,argv[2]);
  }
  std::cout << "search pattern: " << b << std::endl;

  while (lister.get_next_file(".*\\.mid") != NULL) {
    ++counter;
    printf("\n%s ",lister.entry_path().c_str());
    allinterval=change(lister.entry_path().c_str());
    printf("change finished.\n");
    //fflush(stdout);
  
  // std::vector<char> *allinterval= change(argv[1]); 

    
    /*printf("\n");
	printf("[");
	for(int i = 1; i <= 16; i++){
	  for(int j = 0; allinterval[i].size()  > j; ++j) {
	    printf("%c, ",allinterval[i][j]);
	  }
	printf("]\n");
	}*/

    int allfrag  = 0;
	
	for(int k = 0; k < 16; k++){
	  
	  if (allinterval[k].size() == 0) continue;
	  
	  std::string stdString(allinterval[k].begin(),allinterval[k].end());
	  const char *a = stdString.c_str();
	  
	  //printf("%d: %s\n",k,a);
	  allfrag = search(a,b);
	  if (allfrag)
	    break;
	}
	printf("総合結果 = ");
	if(allfrag == 1)
	  printf("一致 match %s",lister.entry_path().c_str());
	else
	  printf("不一致 no");

  
	std::cout << std::endl;
  }
  
  printf("%d\n",count);
  return 0;
  	 
}

