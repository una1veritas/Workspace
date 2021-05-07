import hashlib

headerstr = b'Algorithm Design 2021'
hasher = hashlib.sha256()
hasher.update(headerstr + b'202C1104')
print(hasher.hexdigest())
print(hasher.digest_size)
print(hasher.block_size)
