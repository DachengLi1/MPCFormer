class struct():
    def __init__(self):
        self.a = 0

def add_one(obj):
    obj.a += 1

obj = struct()
print(obj.a)
add_one(obj)
print(obj.a)


def add_list(list):
    list.append(1)

list = []
add_list(list)
print(list)

def add_tuple(tup):
    tup = (1,2)

tup = ()
add_tuple(tup)
print(tup)
