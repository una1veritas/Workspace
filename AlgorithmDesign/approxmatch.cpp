#include <iostream>
#include <string>



int main(int argc, char ** argv) {
  std::string buf, text, pattern;

  pattern = argv[1];

  while (! std::cin.eof() ) {
    std::cin >> buf;
    text += buf;
  }

  std::cout << "Pattern:\n" << '\"' << pattern << '\"' << std::endl << std::flush;
  std::cout << "Text:\n" << '\"' << text << '\"' << std::endl << std::flush;

  int i, j, tmp;
  int tbl[text.size()][pattern.size()];

  for (i = 0; i < text.size(); i++)
    tbl[i][0] = 0;
  for (j = 0; j < pattern.size(); j++) 
    tbl[0][j] = 0;

  for (i = 1; i < text.size(); i++) {
    for (j = 1; j < pattern.size(); j++) {
      
      tmp = tbl[i-1][j-1];
      if ( text[i-1] != pattern[j-1] )
	tmp += 1;
      if ( tbl[i-1][j] + 1 < tmp )
	tmp = tbl[i-1][j] + 1;
      if ( tbl[i][j-1] + 1 < tmp )
	tmp = tbl[i][j-1] + 1;
      tbl[i][j] = tmp;
    }
  }

  for (i = 0; i < text.size(); i++) {
    for (j = 0; j < pattern.size(); j++) {
      std::cout << tbl[i][j] << " ";
    }
    std::cout << std::endl << std::flush;
  }
  

  std::cout << std::endl << std::endl << "Result: " << tbl[text.size()-1][pattern.size()-1] << std::endl;

  return 0;
}
