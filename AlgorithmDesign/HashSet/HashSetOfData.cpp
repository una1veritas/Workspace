#include <stdio.h>

struct Data {
	long value;
	
	Data(long d) {
		value = d;
	}
	
	int equals(Data * p) {
		return (value == p->value);
	}
	
	unsigned long hashCode() {
		return value;
	}
	
	void printContents() {
		printf(" %ld", value);
	}
};

struct LinkedList {
	Data * data;
	LinkedList * next;
	
	LinkedList() {
		next = NULL;
	}
	
	LinkedList * findInNextOrNull(Data * d) {
		LinkedList *last = this;
		while (last->next != NULL) {
			if (last->next->data->equals(d))
				break;
			last = last->next;
		}
		return last;
	}
	
	void add(Data * d) {
		LinkedList *last; 
		last = findInNextOrNull(d);
		if (last->next == NULL) {
			last->next = new LinkedList();
			last->next->data = d;
		}
		return;
	}
	
	void printContents() {
		LinkedList *last = this;
		while (last->next != NULL) {
			last = last->next;
			printf("key ");
			last->data->printContents();
			printf(", ");
		}
	}
};

struct HashSetOfData {
	LinkedList * table;
	long tally;
	
	HashSetOfData() {
		tally = 17;
		table = new LinkedList[17]();
	}
	
	void add(long v) {
		Data * tmp = new Data(v);
		table[tmp->hashCode() % tally].add(tmp); 
	}
	
	void printContents() {
		for (int i= 0; i < tally; i++) {
			printf("at table[%d]: ", i);
			table[i].printContents();
			printf("\r\n");
		}
	}
};

int main(int argc, char * argv[]) {
	HashSetOfData * set = new HashSetOfData();
	
	set->add((long)1000);
	set->add((long)13);
	set->add((long)34871);
	set->add((long)1000);
	set->add((long)21);
	set->printContents();

	return 0;
}

/*
 *
sin$ ./a.out 
at table[0]: 
at table[1]: 
at table[2]: 
at table[3]: 
at table[4]: key 34871, key 21, 
at table[5]: 
at table[6]: 
at table[7]: 
at table[8]: 
at table[9]: 
at table[10]: 
at table[11]: 
at table[12]: 
at table[13]: key 13, 
at table[14]: key 1000, 
at table[15]: 
at table[16]: 
sin$ 
 */
