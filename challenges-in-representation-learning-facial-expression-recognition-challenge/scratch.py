
def gen():
    a, b = 0, 1
    for i in range(19):
        yield a
        a, b = b, a + b

    for i in range(30):
        yield 42

if __name__ == '__main__':
    for i in gen():
        print(i)
        input()
