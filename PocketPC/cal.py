

header = [0x04, 0x50, 0x49, 0x43, 0x41, 0x4C, 0x43, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x42, 0x41, 0x53, 0x0D, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24]

sum = 0
xsum = 0
for x in range(2,len(header)-1):
    sum += x
    xsum ^= x

print('sum = %x' % sum)
print('xsum = %x' % xsum
)

