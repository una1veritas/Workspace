#include <iostream>
#include <fstream>
#include <set>
#include <queue>
#include <string>
#include <vector>
#include <stack>
#include <time.h>

using namespace std;

const int SIGMA_SIZE = 100;
int NODE_NUM_AHO = 0;
int NODE_NUM_DAWG = 0;
int DAWGTOAC_NUM = 0;

struct Trie {
	Trie *edges[SIGMA_SIZE];
	Trie *fail;
	set<string> out;
	set<Trie*> ine;
	int nodenum;
	int ine_num;

	// parent = NULL for the root node
	Trie() {
		fail = NULL;
		ine_num = 0;
		nodenum = NODE_NUM_AHO;
		NODE_NUM_AHO++;
		for (int i = 0; i < SIGMA_SIZE; i++) {
			edges[i] = NULL;
		}
	}

};
Trie *troot = new Trie();


struct Node {
	Node *edges[SIGMA_SIZE];
	//edge_num[i]  0 = edgeなし 1 = primary-edgeあり 2 = secondary-edgeあり
	int edge_num[SIGMA_SIZE];
	Node *suff;
	Trie *dtoc;
	set<string> out;
	set<Node*> ine;
	//dawgにおけるノードの番号
	int nodenum;
	//ACのノードとdawgのノードの番号を一致させるための変数
	int dawgtoac;
	//trunkノードかbranchノードかの判定 1ならtrunk、0ならbranch 初期値は0
	int torb;
	//
	int mark;

	Node() {
		suff = NULL;
		dtoc = NULL;
		torb = 0;
		mark = 0;
		dawgtoac = 0;
		nodenum = NODE_NUM_DAWG;
		NODE_NUM_DAWG++;
		for (int i = 0; i < SIGMA_SIZE; i++) {
			edges[i] = NULL;
		}
		for (int i = 0; i < SIGMA_SIZE; i++) {
			edge_num[i] = 0;
		}

	}

};
Node *nroot = new Node();


Node *split(Node *parentnode, Node *childnode, char a) {
	Node *newchildnode = new Node();
	Node *currentnode = parentnode;
	int charnum = a - ' ';

	parentnode->edges[charnum] = newchildnode;
	parentnode->edge_num[charnum] = 1;

	//childのedgeをすべてnewchildのsecondary-edgeにコピーする
	for (int i = 0; i < SIGMA_SIZE; i++) {
		newchildnode->edges[i] = childnode->edges[i];
		newchildnode->edge_num[i] = 2;
	}


	//parentnode->hassecedgechar[childnode->nodenum] = 0;

	newchildnode->suff = childnode->suff;
	newchildnode->suff->ine.insert(newchildnode);
	childnode->suff->ine.erase(childnode);
	childnode->suff = newchildnode;
	newchildnode->ine.insert(childnode);

	//Node *currentnode = parentnode;

	int m = 0;
	while (currentnode != nroot) {
		currentnode = currentnode->suff;
		for (m = 0; m < SIGMA_SIZE; m++) {
			if ((currentnode->edges[m] == childnode) && (currentnode->edge_num[m] == 2)) {
				currentnode->edges[m] = newchildnode;
				break;
			}

		}


		if (m == (SIGMA_SIZE - 1))
			break;

	}

	return newchildnode;

}



Node *update(Node *activenode, char a) {


	int charnum = a - ' ';

	if (activenode->edges[charnum] != NULL) {

		if (activenode->edge_num[charnum] == 1)
			return activenode->edges[charnum];
		else
			return split(activenode, activenode->edges[charnum], a);

	}
	else {
		Node *newactivenode = new Node();
		activenode->edges[charnum] = newactivenode;
		activenode->edge_num[charnum] = 1;

		Node *currentnode = activenode;


		while ((currentnode != nroot) && (newactivenode->suff == NULL)) {
			currentnode = currentnode->suff;

			if ((currentnode->edges[charnum] != NULL) && (currentnode->edge_num[charnum] == 1)) {
				newactivenode->suff = currentnode->edges[charnum];
				newactivenode->suff->ine.insert(newactivenode);
			}
			else if ((currentnode->edges[charnum] != NULL) && (currentnode->edge_num[charnum] == 2)) {
				newactivenode->suff = split(currentnode, currentnode->edges[charnum], a);
				newactivenode->suff->ine.insert(newactivenode);

			}
			else {
				currentnode->edges[charnum] = newactivenode;
				currentnode->edge_num[charnum] = 2;
			}
		}

		if (newactivenode->suff == NULL) {
			newactivenode->suff = nroot;
			nroot->ine.insert(newactivenode);
		}



		return newactivenode;

	}
}


void Goto(Trie * root, string & str, int depth = 0) {
	Trie * node = root;
	int nextch;
	for(unsigned int depth = 0; depth < str.size(); depth++) {
		nextch = str[depth] - ' ';
		if ( node->edges[nextch] == NULL || node->edges[nextch] == root ) {
			node->edges[nextch] = new Trie();
		}
	}
	node->out.insert(str);
	return;
	/*
	if (depth == curString.size()) {
		node->out.insert(curString);

		return;
	}

	int next = curString[depth] - ' ';

	if (node->edges[next] == NULL || node->edges[next] == troot) {
		node->edges[next] = new Trie();
	}

	Goto(node->edges[next], curString, depth + 1);
	*/
}

/*
int Goto2(Trie *node, string &curString, int depth = 0, int depth2 = -1) {
	if (depth == curString.size()) {
		node->out.insert(curString);

		return depth2;
	}


	int next = curString[depth] - ' ';

	if (node->edges[next] == NULL || node->edges[next] == troot) {
		node->edges[next] = new Trie();
		if (depth2 == -1)
			depth2 = depth;
	}

	Goto2(node->edges[next], curString, depth + 1, depth2);
}
*/


vector<Trie *> getoutstates(string &string) {
	vector<Trie *> outstates;
	Node *activenode = nroot;
	Node *node;
	int stringsize = string.size();
	stack <Node*> st;

	for (int i = 0; (i < (stringsize)) && (activenode != NULL); i++) {

		activenode = activenode->edges[string[i] - ' '];
		
	}


	if (activenode != NULL) {
		queue <Node*> queue;
		queue.push(activenode);
		while (!queue.empty()) {
			node = queue.front();
			queue.pop();
			if (node->torb == 1) {
				outstates.push_back(node->dtoc);
				//node->dtoc->out.insert(string);
			}
			for (auto s : node->ine)
				queue.push(s);

		}
	}

	return outstates;
}



vector<Trie *> getfailstates(string &string, int depth) {
	vector< Trie * > failstates;
	vector<int> tmp;
	Node *activenode = nroot;
	Node *node;
	Node *enode;
	int num;
	int stringsize = string.size();
	stack<Node *> st;
	stack<int> st_num;


	for (int i = 0; (i < stringsize) && (activenode != NULL); i++) {
		activenode = activenode->edges[string[i] - ' '];
		//cout << string[i] << " " << activenode->nodenum << endl;
		if ((i+1 >= (depth)) && (activenode != NULL)) {
			st.push(activenode);
			st_num.push(i+1);
		}
		//activenode = activenode->edges[string[i] - ' '];
	}

	//cout << "st size " << st.size() << endl;

	queue<Node*> queue;

	while (!st.empty()) {
		node = st.top();
		st.pop();
		num = st_num.top();
		st_num.pop();
		queue.push(node);

		while (!queue.empty()) {

			enode = queue.front();
			queue.pop();
			if (enode->mark == 0) {
				enode->mark = 1;
				if (enode->torb == 1) {
					enode->dtoc->ine_num = num;
					failstates.push_back(enode->dtoc);
				}
				else
					for (/*auto*/ Node * s : enode->ine) {
						//cout << "now node num " << enode->nodenum << endl;
						//cout << "enode->ine num " << s->nodenum << endl;
						queue.push(s);
					}

			}
		}

	}

	//cout << "failstates size " << failstates.size() << endl;

	/*
	for (int i = 0; i < failstates.size(); i++) {
		cout << "failstate num " << failstates[i]->nodenum << endl;
	}
	*/

	return failstates;
}



int main(const int argc, const char * argv[]) {

	//キーワード（の集合）が入力として与えられる
	//キーワードを用いて、通常の方法でACマシンとDAWGを作る

	//
	//ACマシンの構成
	//

	// rootの初期化
	for (int i = 0; i < SIGMA_SIZE; i++) {
		troot->edges[i] = troot;
		troot->fail = troot;
	}

	//goto関数の構成
	char check = 'y';
	std::ifstream reading_file;
	//cout << "ファイルからの読み込みを行いますか？ y/n" << endl;
	//cin >> check;
	if (check == 'y') {
		//string ward_num;
		int ward_num_i;
		string ward_string;
		string file_name;

		if ( argc <= 1 ) {
			//cout << "ファイル名を入力してください" << endl;
			//cin >> file_name;

			//file_name = "word_2000.txt";
			//file_name = "test5.txt";
			//file_name = "word_list3.txt";
			file_name = "word_list2.txt";
			//file_name = "aho_check_in.txt";
		} else {
			file_name = argv[1];
		}
		//std::ifstream reading_file;
		string reading_line;

		reading_file.open(file_name, std::ios::in);
		if (reading_file.fail())
		{
			std::cerr << "reading file failed." << std::endl;
			return -1;
		}
		reading_file >> reading_line; //std::getline(reading_file, reading_line);
		//ward_num = reading_line;
		ward_num_i = atoi(reading_line.c_str());
		std::cout << "word num = " << ward_num_i << std::endl;

		for (int i = 0; i < ward_num_i; i++) {
			//std::getline(reading_file, reading_line);
			reading_file >> reading_line;
			std::cout << "line " << i << ": '" << reading_line << "'" << std::endl;

			Goto(troot, reading_line);
		}
	}
	else if (check == 'n') {

		// Number of strings
		int nrStrings;
		cin >> nrStrings;
		int j = 1;
		for (int i = 0; i < nrStrings; i++, j++) {
			string curString;

			cout << j;

			cin >> curString;

			Goto(troot, curString);
		}

	}


	// failure関数の構成
	queue<Trie*> q;

	// Must to this before, because of the fact that every edge out of the root is
	// not NULL
	for (int i = 0; i < SIGMA_SIZE; i++) {
		if (troot->edges[i] != NULL && troot->edges[i] != troot) {
			troot->edges[i]->fail = troot;
			q.push(troot->edges[i]);
		}
	}

	//Trie *tnode;

	//cout << "check point1" << endl;

	//int j = 1;
	while (!q.empty()) {
		Trie *curNode = q.front();
		q.pop();

		for (int i = 0; i < SIGMA_SIZE; i++) {
			Trie *next = curNode->edges[i];
			if (next != NULL && next != troot) {
				q.push(next);

				Trie *f = curNode->fail;
				for (; f->edges[i] == NULL; f = f->fail);

				next->fail = f->edges[i];

				/*
				if (next->nodenum == 4060) {
					cout << "4060 check" << endl;
					if (next->fail->nodenum == 4060)
						cout << "fail 4060 check" << endl;
				}

				if (curNode->nodenum == 4060) {
					cout << "curNode 4060 check" << endl;
					if (curNode->fail->nodenum == 4060)
						cout << "curNode fail 4060 check" << endl;
				}
				*/

				//f->edges[i]->ine.insert(next);

				for (auto s : next->fail->out) {
					next->out.insert(s);
				}
			}
		}
	}

	cout << "aho struct finish" << endl;


	//
	//dawgの構成
	//

	nroot->dawgtoac = DAWGTOAC_NUM;
	DAWGTOAC_NUM++;
	nroot->torb = 1;


	char check2 = 'y';
	//cout << "ファイルからの読み込みを行いますか？ y/n" << endl;
	//cin >> check2;
	if (check2 == 'y') {
		string ward_num;
		int ward_num_i;
		string ward_string;
		string file_name;

		Node *activenode;
		activenode = nroot;
		Node *tnode;

		Trie *trienode;
		Trie *tmptrie;

		//cout << "ファイル名を入力してください" << endl;
		//cin >> file_name;

		//file_name = "word_2000.txt";
		//file_name = "test5.txt";
		//file_name = "word_list3.txt";
		file_name = "word_list2.txt";
		//file_name = "aho_check_in.txt";

		std::ifstream reading_file2;
		string reading_line;

		reading_file2.open(file_name, std::ios::in);
		if (reading_file2.fail())
		{
			std::cerr << "failed reading file 2." << std::endl;
			return -1;
		}
		std::getline(reading_file2, reading_line);
		ward_num = reading_line;

		ward_num_i = atoi(ward_num.c_str());


		for (int i = 0; i < ward_num_i; i++) {

			//std::getline(reading_file2, reading_line);
			reading_file2 >> reading_line;
			std::cerr << "getline '" << reading_line << "'" << std::endl;

			//cout << reading_line << " " << i << endl;

			int stringsize = reading_line.size();

			for (int j = 0; j < stringsize; j++)
				activenode = update(activenode, reading_line[j]);


			activenode = nroot;
			trienode = troot;

			activenode = activenode->edges[reading_line[0] - ' '];
			trienode = trienode->edges[reading_line[0] - ' '];


			for (int j = 1; j < stringsize; j++) {
				if (activenode->dawgtoac == 0) {
					activenode->dawgtoac = DAWGTOAC_NUM;
					activenode->torb = 1;
					DAWGTOAC_NUM++;
					activenode->dtoc = trienode;
				}

				tnode = activenode->edges[reading_line[j] - ' '];
				tmptrie = trienode->edges[reading_line[j] - ' '];
				activenode = tnode;
				trienode = tmptrie;

				if ( trienode == NULL )
					std::cerr << "reached to NULL trie." << std::endl;
			}
			if (activenode->dawgtoac == 0) {
				activenode->dawgtoac = DAWGTOAC_NUM;
				activenode->torb = 1;
				DAWGTOAC_NUM++;
				activenode->dtoc = trienode;
			}

			activenode = nroot;
		}
	}
	else if (check2 == 'n') {

		cout << "入力するキーワードの数" << endl;
		int nrStrings;
		cin >> nrStrings;

		for (int i = 0; i < nrStrings; i++) {
			cout << "キーワードを入力してください   ";
			string curString;

			Node *activenode;
			activenode = nroot;
			Node *tnode;

			cin >> curString;
			int stringsize = curString.size();


			for (int j = 0; j < stringsize; j++)
				activenode = update(activenode, curString[j]);


			activenode = nroot;
			activenode = activenode->edges[curString[0] - ' '];

			for (int j = 1; j < stringsize; j++, DAWGTOAC_NUM++) {
				activenode->dawgtoac = DAWGTOAC_NUM;
				activenode->torb = 1;
				tnode = activenode->edges[curString[j] - ' '];
				activenode = tnode;
			}

			activenode = nroot;
		}

	}

	cout << "dawg struct finish" << endl;


	//（動作確認）

	// テキストからキーワードを検出
	string bigString;
	std::ifstream ifs("test4.txt");
	//std::ifstream ifs("aho_check_out.txt");
	if (ifs.fail())
	{
		std::cerr << "失敗" << std::endl;
		return -1;
	}
	getline(ifs, bigString);
	cout << bigString << endl;

	Trie *node = troot;
	int k = bigString.size();
	for (int i = 0; i < k; i++) {
		int cur = bigString[i] - ' ';

		for (; node->edges[cur] == NULL; node = node->fail);

		node = node->edges[cur];

		if (node->out.size() != 0) {
			cout << "At position " << i << " we found:\n";

			for (auto s : node->out) {
				cout << s << "\n";
			}
		}
	}


	//
	//キーワードを新たに与える（一語ずつ）
	//DAWGを動的に構成
	//DAWGをもとにACマシンを更新
	//


	while (1) { //dynamic start


		int key_num = 0;


		//cin >> curString2;
		cout << "キーワードをいくつ追加しますか？" << endl;
		cin >> key_num;

		//std::ifstream readline2("word_list3.txt");
		//std::ifstream readline2("test3.txt");
		//std::ifstream readline2("aho_check_in_plus.txt");

		clock_t start, end, total_start, total_end;
		double time1 = 0, time2 = 0, time3 = 0, time4 = 0, time5 = 0;

		total_start = clock();

		for (int i7 = 0; i7 < key_num; i7++) {


			//Trie *node2 = troot;
			//Trie *nodepoint2 = troot;
			Trie * tactivenode;
			Trie *newstates[50];
			//int depth = 0;

			string curString2;
			int size2 = 0;
			//int cur3 = 0;

			Node *activenode = nroot;
			Node *tnode = nroot;


			//キーワードの読み込み

			//getline(readline2, curString2);
			getline(reading_file, curString2);

			//cout << curString2 << " " << i7 << endl;

			size2 = curString2.size();

			/*

			//dawgを動的に追加

			for (int i = 0; i < size2; i++)
				activenode = update(activenode, curString2[i]);


			activenode = nroot;
			//Trie *trienode = troot;
			//Trie *tmptrie = troot;

			activenode = activenode->edges[curString2[0] - ' '];
			//trienode = trienode->edges[curString2[0] - ' '];



			for (int i = 1; i < size2; i++) {
				if (activenode->dawgtoac == 0) {
					//cout << "check node num" << activenode->nodenum << endl;
					activenode->dawgtoac = DAWGTOAC_NUM;
					activenode->torb = 1;
					DAWGTOAC_NUM++;
				}
				tnode = activenode->edges[curString2[i] - ' '];
				activenode = tnode;
			}
			if (activenode->dawgtoac == 0) {
				//cout << "check node num" << activenode->nodenum << endl;
				activenode->dawgtoac = DAWGTOAC_NUM;
				activenode->torb = 1;
				DAWGTOAC_NUM++;
			}

			*/

			//Algorithm1

			tactivenode = troot;
			int charnum = 0;
			int trie_depth = 10000;
			//Node *ncheck = nroot;

			//cout << "check 0" << endl;

			start = clock();

			for (int i = 0; i < size2; i++) {
				newstates[i] = tactivenode;
				charnum = curString2[i] - ' ';


				if ((tactivenode->edges[charnum] != NULL) && (tactivenode->edges[charnum] != troot)) {
					tactivenode = tactivenode->edges[charnum];
					//ncheck = ncheck->edges[charnum];
				}
				else {
					//cout << "else check" << endl;

					if (trie_depth == 10000) {
						trie_depth = i + 1;
						//cout << "trie_depth " << trie_depth << endl;
					}

					Trie *newstate = new Trie();

					tactivenode->edges[charnum] = newstate;
					//ncheck->edges[charnum]->dtoc = newstate;


					tactivenode = tactivenode->edges[charnum];
					//ncheck = ncheck->edges[charnum];


				}


			}
			newstates[size2] = tactivenode;
			tactivenode->out.insert(curString2);

			//cout << "check 1" << endl;

			end = clock();
			time1 += (end - start);

			start = clock();

			vector< Trie* > failstates;
			failstates = getfailstates(curString2, trie_depth);
			int fstatesize = failstates.size();


			for (int i = fstatesize - 1; i >= 0; i--) {
				failstates[i]->fail = newstates[failstates[i]->ine_num];
				//cout << "fail state num " << failstates[i]->nodenum << "to state " << newstates[failstates[i]->ine_num]->nodenum << endl;
				/*
				if (failstates[i]->nodenum == 23) {
					if (failstates[i]->fail == newstates[3])
						cout << "yes" << endl;
					else
						cout << "no" << endl;
				}
				*/
			}

			//cout << "check 2" << endl;

			end = clock();
			time2 += (end - start);

			tactivenode = troot;
			Trie *failurenode;

			start = clock();

			for (int i = 0; i < size2; i++) {

				if ((i + 1) >= trie_depth) {
					if (tactivenode == troot) {
						tactivenode->edges[curString2[i] - ' ']->fail = troot;
						tactivenode = tactivenode->edges[curString2[i] - ' '];
					}
					else {
						failurenode = tactivenode->fail;

						int j = 0;
						while (failurenode->edges[curString2[i] - ' '] == NULL) {

							if (j >= 2)
								break;

							failurenode = failurenode->fail;
							if (failurenode == troot)
								j++;
						}

						tactivenode = tactivenode->edges[curString2[i] - ' '];

						tactivenode->fail = failurenode->edges[curString2[i] - ' '];

						for (auto s : failurenode->edges[curString2[i] - ' ']->out)
							tactivenode->out.insert(s);

					}
					
				}
				else {
					tactivenode = tactivenode->edges[curString2[i] - ' '];
				}
			}

			//cout << "check 3" << endl;

			end = clock();
			time3 += (end - start);

			vector<Trie *> outstates;

			start = clock();

			//cout << "check 1" << endl;
			outstates = getoutstates(curString2);
			//cout << "check 2" << endl;

			int outstatesize = outstates.size();

			//end = clock();
			//time4 += (end - start);

			
			for (int i = 0; i < outstatesize; i++) {
				outstates[i]->out.insert(curString2);
			}
			
			
			end = clock();
			time4 += (end - start);

			activenode = nroot;

			start = clock();

			for (int i = 0; i < size2; i++)
				activenode = update(activenode, curString2[i]);

			//end = clock();
			//time5 += (end - start);


			activenode = nroot;
			Trie *trienode = troot;
			Trie *tmptrie;


			activenode = activenode->edges[curString2[0] - ' '];
			trienode = trienode->edges[curString2[0] - ' '];


			for (int i = 1; i < size2; i++) {
				if (activenode->dawgtoac == 0) {
					activenode->dawgtoac = DAWGTOAC_NUM;
					activenode->torb = 1;
					DAWGTOAC_NUM++;
					activenode->dtoc = trienode;
				}


				tnode = activenode->edges[curString2[i] - ' '];
				tmptrie = trienode->edges[curString2[i] - ' '];
				activenode = tnode;
				trienode = tmptrie;
			}


			if (activenode->dawgtoac == 0) {
				activenode->dawgtoac = DAWGTOAC_NUM;
				activenode->torb = 1;
				DAWGTOAC_NUM++;
				activenode->dtoc = trienode;
			}

			activenode = nroot;

			end = clock();
			time5 += (end - start);


		}

		total_end = clock();
		cout << "total time: " << (double)(total_end - total_start) / CLOCKS_PER_SEC << "sec.\n";
		cout << "time1: " << (double)time1 / CLOCKS_PER_SEC << "sec.\n";
		cout << "time2: " << (double)time2/ CLOCKS_PER_SEC << "sec.\n";
		cout << "time3: " << (double)time3/ CLOCKS_PER_SEC << "sec.\n";
		cout << "time4: " << (double)time4/ CLOCKS_PER_SEC << "sec.\n";
		cout << "time5: " << (double)time5/ CLOCKS_PER_SEC << "sec.\n";

		// Read big string, in which we search for elements
		string bigString2;
		//cin >> bigString;
		//cin.ignore();

		getline(ifs, bigString2);
		cout << bigString2 << endl;

		Trie *node3 = troot;
		int k2 = bigString2.size();
		cout << "string size " << k2 << endl;
		//cout << "root " << node3->nodenum << endl;

		for (int i = 0; i < k2; i++) {

			int cur = bigString2[i] - ' ';

			//ここまでok
			for (; node3->edges[cur] == NULL; node3 = node3->fail)
			{
				cout << "failure now... node number " << node3->fail->nodenum << endl;
			}


			node3 = node3->edges[cur];

			cout << "current node number " << node3->nodenum << endl;


			if (node3->out.size() != 0) {
				cout << "At position " << i << " we found:\n";

				for (auto s : node3->out) {
					cout << s << "\n";
				}
			}

		}



	} //dynamic finish

	return 0;
}
