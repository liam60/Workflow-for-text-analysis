
class A:

    name = 'A'

    @staticmethod
    def foo():
        pass

    def __init__(self):
        print(self.name)
        # self.name = 'b'

        print(self.__dict__)

print(A.__dict__)
a = A()
print(a.name)
print(A.name)

print(A.foo == a.foo)
print(A.name == a.name)


class B(A):
    name = 'BBB'


print(A.name)
print(B.name)

b = B()

print(b.name)
print(getattr(b, 'name'))
print(b.__dict__)


def foo(a, **kwargs):
    print(type(kwargs))
    print(kwargs)

foo(a=1)