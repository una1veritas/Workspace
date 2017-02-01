#include <iostream>
#include <fstream>
#include <set>
#include <queue>
#include <string>
#include <cstdlib>
#include <time.h>

using namespace std;


const int SIGMA_SIZE = 100;
int NODE_NUM = 0;



struct Trie {
	Trie *edges[SIGMA_SIZE];
	Trie *fail;
	set<string> out;
	set<Trie*> ine;
	int nodenum;

	Trie() {
		fail = NULL;
		nodenum = NODE_NUM;
		NODE_NUM++;
		for (int i = 0; i < SIGMA_SIZE; i++) {
			edges[i] = NULL;
		}

	}
};
Trie *root = new Trie();

void addString(Trie *node, string &curString, int depth = 0) {
	if (depth == curString.size()) {
		node->out.insert(curString);

		return;
	}

	//int next = curString[depth] - 'a';
	int next = curString[depth] - ' ';
	//cout << "curstring" << next << endl;

	if (node->edges[next] == NULL || node->edges[next] == root) {
		node->edges[next] = new Trie();
	}

	addString(node->edges[next], curString, depth + 1);
}

int addString2(Trie *node, string &curString, int depth = 0, int depth2 = -1) {

	//if (curString[depth] == '*')
		//cout << "*" << endl;

	if (depth == curString.size()) {
		node->out.insert(curString);

		return depth2;
	}

	//int next = curString[depth] - 'a';
	int next = curString[depth] - ' ';
	//cout << "curstring" << next << endl;
	

	if (node->edges[next] == NULL || node->edges[next] == root) {
		node->edges[next] = new Trie();
		if (depth2 == -1)
			depth2 = depth;
	}

	addString2(node->edges[next], curString, depth + 1, depth2);
}

int main() {
	// rootの初期化
	for (int i = 0; i < SIGMA_SIZE; i++) {
		root->edges[i] = root;
		root->fail = root;
	}


	Trie *tr[50];
	int queue_size = 0;

	//goto関数の構成
	char check;
	//cout << "ファイルからの読み込みを行いますか？ y/n" << endl;
	//cin >> check;
	check = 'y';
	std::ifstream reading_file;

	if (check == 'y') {
		string ward_num;
		int ward_num_i;
		string ward_string;
		string file_name;
		//cout << 1;
		//cin >> bigString;
		//cin.ignore();

		//cout << "ファイル名を入力してください" << endl;
		//cin >> file_name;

		file_name = "word_list2.txt";
		//file_name = "test5.txt";
		//file_name = "aho_check_in.txt";

		//std::ifstream reading_file;
		string reading_line;

		reading_file.open(file_name, std::ios::in);
		if (reading_file.fail())
		{
			std::cerr << "失敗" << std::endl;
			return -1;
		}
		std::getline(reading_file, reading_line);
		ward_num = reading_line;

		ward_num_i = atoi(ward_num.c_str());

		for (int i = 0; i < ward_num_i; i++) {

			std::getline(reading_file, reading_line);

			//cout << reading_line << " " << i << endl;

			addString(root, reading_line);
		}
	}
	else if (check == 'n') {

		// Number of strings
		int nrStrings;
		cin >> nrStrings;
		//cout << 1;
		int j = 1;
		for (int i = 0; i < nrStrings; i++, j++) {
			string curString;

			cout << j;

			cin >> curString;

			addString(root, curString);
		}

	}


	// failure関数の構成
	queue<Trie*> q;

	// Must to this before, because of the fact that every edge out of the root is
	// not NULL
	for (int i = 0; i < SIGMA_SIZE; i++) {
		if (root->edges[i] != NULL && root->edges[i] != root) {
			root->edges[i]->fail = root;
			root->ine.insert(root->edges[i]);
			q.push(root->edges[i]);
		}
	}

	//Trie *tnode;

	//cout << "check point1" << endl;

	int j = 1;
	while (!q.empty()) {
		Trie *curNode = q.front();
		q.pop();

		for (int i = 0; i < SIGMA_SIZE; i++) {
			Trie *next = curNode->edges[i];
			if (next != NULL && next != root) {
				q.push(next);

				Trie *f = curNode->fail;
				for (; f->edges[i] == NULL; f = f->fail);

				next->fail = f->edges[i];
				//tnode = f->edges[i];
				//auto n = next->fail->revf2;
				//f->edges[i]->revf2.insert(n);
				//f->edges[i]->revf2.insert(*next); ok
				//tnode->revf.insert(next->fail);
				//auto n = i;
				//f->edges[i]->revf.insert(i);

				//before
				//f->edges[i]->inedges[i] = next;
				//after

				//new before
				//f->edges[i]->inedges[next->nodenum] = next;
				//new after
				f->edges[i]->ine.insert(next);

				for (auto s : next->fail->out) {
					next->out.insert(s);
				}
			}
		}
	}

	cout << "struct finish" << endl;


	// テキストからキーワードを検出
	string bigString;
	//std::ifstream ifs("test_text.txt");
	//std::ifstream ifs("test2.txt");
	//std::ifstream ifs("test4.txt");
	std::ifstream ifs("aho_check_out.txt");
	if (ifs.fail())
	{
		std::cerr << "失敗" << std::endl;
		return -1;
	}
	getline(ifs, bigString);
	cout << bigString << endl;

	Trie *node = root;
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


	//動的な構成
	while (1) { //dynamic start

		//関数の追加
		//int nrStrings2;
		//cin >> nrStrings2;
		//int k = 1;
		/*
		int depth = 0;
		string curString2;
		Trie *node2 = root;
		Trie *nodepoint;
		Trie *nodepoint2 = root;
		int size2 = 0;
		int cur3 = 0;
		*/
		int key_num = 0;
		

		//cin >> curString2;
		cout << "キーワードをいくつ追加しますか？" << endl;
		cin >> key_num;

		//std::ifstream readline2("word_list2.txt");
		//std::ifstream readline2("test3.txt");
		std::ifstream readline2("aho_check_in_plus.txt");

		clock_t total_start, total_end, start, end;
		double total_time = 0, time1 = 0, time2 = 0, time3 = 0, time4 = 0;

		total_start = clock();

		//for (int i = 0; i < nrStrings2; i++) {
		for (int i7 = 0; i7 < key_num; i7++) {

			int depth = 0;
			string curString2;
			Trie *node2 = root;
			Trie *nodepoint;
			Trie *nodepoint2 = root;
			int size2 = 0;
			int cur3 = 0;

			/*
			cout << "キーワードを入力してください" << endl;
			cin >> curString2;
			*/

			//73480 69717 98736 est roop3で止まった？

			//getline(readline2, curString2);
			getline(reading_file, curString2);

			//
			//cout << curString2 << " " << i7 << endl;
			//

			size2 = curString2.size();

			//int cur3 = curString2[i] - ' ';

			start = clock();

			depth = addString2(root, curString2);

			end = clock();
			time1 += end - start;

			//cout << "depth " << depth << endl;

			//start = clock();

			if (depth == -1) {

				cout << "depth = -1" << endl;

				if (node2->out.count(curString2) > 0) {
					;
				}
				else {
					queue<Trie*> q4;
					Trie *rq;

					for (int k = 0; k < size2; k++) {
						cur3 = curString2[k] - ' ';
						node2 = node2->edges[cur3];
						//cout << "cur3 " << cur3 << " " << node2->nodenum << endl;
					}

					q4.push(node2);

					while (!q4.empty()) {
						rq = q4.front();
						q4.pop();
						rq->out.insert(curString2);
						for (auto s : rq->ine)
							q4.push(s);
					}

				}
			}
			else {

				start = clock();

				if (depth != 0) {
					//if (depth > 0) {
					for (int k = 0; k < depth; k++) {
						cur3 = curString2[k] - ' ';
						node2 = node2->edges[cur3];
						//cout << "cur3 " << cur3 << " " << node2->nodenum << endl;
					}
				}

				//cout << "depth " << depth << endl;

				nodepoint = node2;
				nodepoint2 = nodepoint;

				//cout << "check point1" << endl; //ok


				for (int k = depth; k < size2; k++) {
					//Trie *next2 = nodepoint->edges[curString2[k + 1]-' ']; before
					Trie *next2 = nodepoint->edges[curString2[k] - ' ']; //after
																		 //if (next != NULL && next != root) {
																		 //q.push(next);

					Trie *f = nodepoint->fail;
					//int h = curString2[k + 1]-' '; before
					int h = curString2[k] - ' '; //after
					//cout << "curString2 " << h << endl;

					int p = 0;
					for (; (f->edges[curString2[k] - ' '] == NULL) /*|| (p > 0)*/; f = f->fail);

					//cout << "chech point1.5" << endl;

					//if (depth == 0) {s
					if (k == 0) {
						next2->fail = root;
						//cout << "nodenum " << next2->nodenum << endl;
						//before
						//root->inedges[next2->nodenum] = next2; 
						//after
						root->ine.insert(next2);
					}
					else if (f->edges[curString2[k] - ' '] == root) {
						next2->fail = root;
						//cout << "nodenum " << next2->nodenum << endl;
						//before
						//root->inedges[next2->nodenum] = next2;
						//after
						root->ine.insert(next2);
					}
					else {
						//next2->fail = f->edges[curString2[k + 1]]; before
						next2->fail = f->edges[curString2[k] - ' ']; //after

																	 //before
																	 //f->edges[curString2[k] - ' ']->inedges[curString2[k] - ' '] = next2; 
																	 //after
						//cout << "nodenum " << next2->nodenum << endl;
						//before
						//f->edges[curString2[k] - ' ']->inedges[next2->nodenum] = next2;
						//after
						f->edges[curString2[k] - ' ']->ine.insert(next2);
					}


					nodepoint = next2;

					if (k == (size2 - 1))
						for (auto s : next2->fail->out)
							next2->out.insert(s);


				}

				end = clock();
				time2 += end - start;



				//既存の関数の更新
				
				// build the fail function
				queue<Trie*> q2;
				//queue<Trie> q2;
				queue<int> q2_num;
				int q2_n = 0;

				//q2.push(nodepoint);

				//before
				/*
				for (int i = 0; i < NODE_SIZE; i++) {
					if ((nodepoint2->inedges[i] != NULL))
						q2.push(nodepoint2->inedges[i]);

				}
				*/
				//after

				start = clock();

				for (auto s : nodepoint2->ine) {
					if (s != root->edges[curString2[0] - ' ']) {
						q2.push(s);
						q2_num.push(depth);
						//cout << "node->ine num" << s->nodenum << endl;
					}
				}

				//next2->out.insert(s);


				//cout << "check point2" << endl;

				//Trie *tr[50];
				Trie *temp = root;
				for (int i = 0; i < size2; i++) {
					tr[i] = temp;
					temp = temp->edges[curString2[i] - ' '];

				}
				//cout << "ok" << endl;

				/*
				curString2[size2 - 1] = '*';
				curString2[size2] = '\0';
				*/

				char curString3[50];

				for (int i = 0; i < size2; i++) {
					curString3[i] = curString2[i];
				}


				curString3[size2] = '*';
				curString3[size2+1] = '\0';

				Trie *r, *rnext;
				Trie *nodepoint3;
				//int i = depth;
				//cout << "size2 = " << size2 << endl;
				while (!q2.empty()) {
					//r = q2.pop;
					//before
					r = q2.front();
					//after
					//*r = q2.front();
					q2.pop();
					//int i = depth;
					int i = q2_num.front();
					q2_num.pop();
					//cout << "node number " << r->nodenum << " i " << i << endl;

					nodepoint3 = tr[i];

					//while (r->edges[curString2[i + 1]] != NULL) {
					//while ((r->edges[curString2[i] - ' '] != NULL) && (i < size2)) {
					//cout << "1a" << endl;
					while ((r->edges[curString3[i] - ' '] != NULL) &&
						   /*(r->edges[curString3[i] - ' '] != root) &&*/ (i < size2)) {
						//rnext = r->edges[curString2[i + 1]];
						//cout << "roop1" << endl;

						//cout << "1a" << endl;
						rnext = r->edges[curString3[i] - ' '];
						//cout << "a" << endl;
						//rnext->fail = nodepoint2.edges[curString2[i + 1]];

						//cout << "2a" << endl;
						rnext->fail = nodepoint3->edges[curString3[i] - ' '];
						//cout << "a" << endl;
						//cout << "roop2" << endl;

						r = rnext;
						//nodepoint = nodepoint2.edges[curString2[i + 1]];
						//cout << "3a" << endl;
						nodepoint3 = nodepoint3->edges[curString3[i] - ' '];
						//cout << "a" << endl;
						i = i + 1;

						//cout << "roop3" << endl;


					}
					//cout << "a" << endl;

					if (size2 <= i) {
						//cout << "1a" << endl;
						r->out.insert(curString2);
						//cout << "a" << endl;
					}

					//cout << "1a" << endl;
					for (auto s : r->ine) {
						q2.push(s);
						q2_num.push(i);
					}
					//cout << "a" << endl;
					//queue_size = q2.size();
					//cout << "queue size " << queue_size << endl;
					//cout << "roop4" << endl;
				}

				//cout << "check point3" << endl;

				end = clock();
				time3 += end - start;

			}

			//cout << "Finished constructing trie\n";

		}

		total_end = clock();
		cout << "total_time: " << (double)(total_end - total_start) / CLOCKS_PER_SEC << "sec.\n";
		cout << "time1: " << time1 / CLOCKS_PER_SEC << "sec.\n";
		cout << "time2: " << time2 / CLOCKS_PER_SEC << "sec.\n";
		cout << "time3: " << time3 / CLOCKS_PER_SEC << "sec.\n";


		// Read big string, in which we search for elements
		string bigString2;
		//cin >> bigString;
		//cin.ignore();

		/*
		std::ifstream ifs("test2.txt");
		if (ifs.fail())
		{
		std::cerr << "失敗" << std::endl;
		return -1;
		}
		*/
		getline(ifs, bigString2);
		cout << bigString2 << endl;

		Trie *node3 = root;
		int k2 = bigString2.size();
		cout << "string size " << k2 << endl;
		cout << "root " << node3->nodenum << endl;

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