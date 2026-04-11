/*
 ============================================================================
 Name        : search_h_example.c
 Author      : Sin Shimozono
 Version     :
 Copyright   : GPL
 Description : Hello World in C, Ansi-style
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <search.h>

typedef struct namedef {
	char name[16];
	int def;
} namedef;

// Comparison function for integers
int compare_namedef(const void *a, const void *b) {
    namedef * val_a = (namedef *)a;
    namedef * val_b = (namedef *)b;
	return strcmp(val_a->name, val_b->name);
}

// Callback function to print tree nodes (used with twalk)
void print_node(const void * node, VISIT visit, int level) {
    namedef * ptr = *(namedef **) node;

    // Only print during preorder traversal to avoid duplicates
    if (visit == preorder || visit == leaf) {
        printf("%s, %d ", ptr->name, ptr->def);
    }
}

namedef * new_namedef(const char *name, const int def) {
	namedef * new_val = (namedef *) malloc(sizeof(namedef));
	if (new_val == NULL) {
		fprintf(stderr, "Memory allocation failed\n");
		return NULL;
	}
	strncpy(new_val->name, name, 15);
	new_val->name[15] = '\0'; // Ensure null termination
	new_val->def = def;
	return new_val;
}

void free_node(void *node) {
	namedef * ptr = (namedef *) node;
	if (ptr != NULL) {
		free(ptr);
	}
}

int main() {
    void *root = NULL;  // Root of the BST
    namedef values[] = {
    		{"bne", 50},
			{"beq", 30},
			{"lda", 70},
			{"clra", 20},
			{"pshs", 40},
			{"leax", 60},
			{"stu", 80},
			{"stu", 10},
			{"adc", 25},
			{"addr", 35},
    };
    int num_values = sizeof(values) / sizeof(values[0]);

    printf("=== Binary Search Tree Operations ===\n\n");

    // 1. INSERT operations
    printf("1. Inserting values: ");
    for (int i = 0; i < num_values; i++) {
        printf("%s, %d ", values[i].name, values[i].def);
        // tsearch inserts the value if not found, returns pointer to node
        namedef * new_node = new_namedef(values[i].name, values[i].def);
        if (new_node != NULL) {
            tsearch(new_node, &root, compare_namedef);
            printf("inserted. %s\n", values[i].name);
        } else {
			fprintf(stderr, "Failed to create node for %s\n", values[i].name);
			continue;
		}
    }
    printf("\n");

    // 2. SEARCH operations
    printf("\n2. Searching for values:\n");
    const char * search_vals[] = {"leax", "beq", "adc"};
    for (int i = 0; i < 3; i++) {
    	namedef node_to_find;
    	strncpy(node_to_find.name, search_vals[i], 15);
    	node_to_find.name[15] = '\0'; // Ensure null termination
    	node_to_find.def = 0; // Value is not used for searching, only name is compared

        void * result = tfind(&node_to_find, &root, compare_namedef);
        if (result != NULL) {
        	namedef * found_node = *(namedef **) result;
            printf("   Found: %s, %d\n", found_node->name, found_node->def);
        } else {
            printf("   Not found: %s\n", search_vals[i]);
        }
    }

    // 3. TRAVERSE the tree (in-order using twalk)
    printf("\n3. In-order traversal of tree:\n   ");
    twalk(root, print_node);
    printf("\n");

    // 4. DELETE operations
    printf("\n4. Deleting values: 30, 20\n");
    int delete_val1 = 30;
    int delete_val2 = 20;

    // tdelete removes the node and returns pointer to parent, or NULL if not found
    tdelete(&delete_val1, &root, compare_namedef);
    tdelete(&delete_val2, &root, compare_namedef);

    printf("   After deletion, tree contains:\n   ");
    twalk(root, print_node);
    printf("\n");

    // 5. COUNT nodes using twalk
    //int node_count = 0;
    printf("\n5. Tree statistics:\n");
    printf("   Number of nodes: %d\n", num_values - 2);

    // Clean up - delete all remaining nodes
    printf("\n6. Cleaning up...\n");
    // Note: search.h doesn't provide automatic cleanup, you need to traverse
    // and free nodes manually or use a custom deletion function

    return 0;
}

