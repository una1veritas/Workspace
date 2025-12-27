
SRC_DIR=../src
SRCS=$(SRC_DIR)/modem_xfer.c $(SRC_DIR)/ymodem.c $(SRC_DIR)/ymodem_send.c
HDRS=$(SRC_DIR)/modem_xfer.h $(SRC_DIR)/modem_xfer_debug.h
#RZ=/Users/takemura/workspace/github/lrzsz-0.12.20/src/lrz
RZ=rz
PIPE=/tmp/modem_test

all: modem_test

modem_test: modem_test.c $(SRCS) $(HDRS)
	cc -I$(SRC_DIR) -DDEBUG -o modem_test modem_test.c $(SRCS)

test:: all
	pkill -a modem_test || true
	pkill -a rz || true
	for r in 654321 123456 341278 923781; do \
          rm -f foo.txt bar.txt baz.dat; \
          ./modem_test --random-seed $${r} & \
          sz --ymodem data/foo.txt data/bar.txt data/baz.dat < $(PIPE)-tx > $(PIPE)-rx; \
          make check_test_result || exit 1; \
          $(RZ) --ymodem --overwrite < /tmp/modem_test-tx > /tmp/modem_test-rx & \
          ./modem_test --random-seed $${r} data/foo.txt data/bar.txt data/baz.dat; \
          make check_test_result || exit 1; \
        done

check_test_result::
	err_count=0; \
	for i in foo.txt bar.txt baz.dat; do \
	    if diff $${i} data/$${i}; then \
	         echo ok $${i} ; \
	    else \
	         echo $${i} ERROR; \
	         shasum $${i} data/$${i}; \
	         hexdump -C data/$${i} > /tmp/modem_test_a; \
	         hexdump -C $${i} > /tmp/modem_test_b; \
	         diff -u /tmp/modem_test_a /tmp/modem_test_b; \
	        err_count=$$(( err_count + 1 )); \
	    fi \
	done; \
	echo ;\
	if [ $${err_count} == 0 ]; then \
	    echo OK;\
	else \
	    echo "some ERROR(s) were detected. Please check the result"; \
	    exit 1; \
	fi; \
	echo

clean::
	rm -f modem_test
