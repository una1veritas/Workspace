rhyme = [
    "Humpty Dumpty sat on a wall,",
    "Humpty Dumpty had a great fall;",
    "All the king's horses, and all the king's men",
    "Couldn't put Humpty together again."
]

with open('nursery.txt', 'w') as file:
    for l in rhyme:
        file.write(l)
        file.write('\n')

print("Finished.")
