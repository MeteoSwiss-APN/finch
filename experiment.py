args = [1, 2]
def foo(x):
    return x+1
funcs = [lambda : foo(a) for a in args]
for f in funcs:
    print(f())