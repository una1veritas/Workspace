
typedef struct {
	/* the set of states must be a subset of {0,...,254}. */
	/* an element of the finite alphabet must be a char. */
	unsigned int delta[65535];
	char initialstate;
	int finalstate[255]; /* 0 -> not a final state, 1 -> a final state. */
} dfa;
const unsigned int TRANSITION_LIMIT = 0xfeff;
const unsigned int TRANSITION_NOT_DEFINED = 0xffff;
const char STATE_NOT_FINAL = 0;
const char STATE_FINAL = 1;
const unsigned char STATE_LIMIT = 0xfe;
const unsigned char ALPHABET_LIMIT = 0xfe;

void dfa_init(dfa * mp) {
	mp->initialstate = 0;
	for(unsigned int i = 0; i < TRANSITION_LIMIT; ++i) {
		mp->delta[i] = TRANSITION_NOT_DEFINED;
	}
	for(unsigned int i = 0; i < STATE_LIMIT; ++i) {
		mp->finalstate[i] = 0;
	}
}

int main(int argc, char **argv) {
	dfa M;
	dfa_init(&M);

	return 0;
}
