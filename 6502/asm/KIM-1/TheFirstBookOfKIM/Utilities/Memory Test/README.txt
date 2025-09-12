MEMORY TEST
Jim Butterfield

Testing RAM isn't just a question of storing a value and
then checking it.  It's important to test for interference
between locations.  Such tests often involve writing to one
location and then checking all other locations to see they
haven't been disturbed; this can be time consuming.

This program checks memory thoroughly and runs exceptionally
fast.  It is adapted from an algorithm by Knaizuk and Hartmann
published in 'IEEE. Transactions on Computers', April 1977.

The program first puts value FF in every location under test.
Then it puts 00 in every third location, after which it tests
all locations for correctness.  The test is repeated twice more
with the positions of the 00's changed each time.  Finally,
the whole thing is repeated with the FF and 00 values 
interchanged.

To run:  Set the addresses of the first and last memory pages
you wish to test into locations 0000 and 0001 respectively.
Start the program at address 0002; it will halt with a memory
address on the display.  If no faults were round, the address
will be one location past the last address tested.  If a fault
is found, its address will be displayed.

Example:  To test 0100 to 02FF (pages 01 and 02) in KIM:
Set 0000 to 01, 0001 to 02, start program at 0002.  If memory
is good, see 0300 (=02FF + 1).  Now if you try testing
0100 to 2000 (0000=01,0001=20) the program will halt at
the first bad location - this will be 0400 if you haven't
added memory.
