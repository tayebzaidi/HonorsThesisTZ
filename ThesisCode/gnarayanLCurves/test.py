import json

data = ([0,1,2,3,4], "a", 3,4)
dest = '/home/levicivita/'
with open(dest + 'test.p', mode='w') as f:
    json.dump(data, f)

