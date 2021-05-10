---
layout: post
title: "python method"
date: 2021-04-23 18:00:00 +0000
categories: [python, code]
---

#### Python method 정리

##### python : 3.7.5

---

```python

class Foo:
    x = 1  # public
    _x = 2  # protected
    __x = 3  # private

    def __init__(self):
        self.y = 1  # public
        self._y = 2  # protected
        self.__y = 3  # private

    def instance_method(self, a):
        '''
        instance method
        '''
        print(f"Foo : instance_method(self, a) = {a}")
        return a

    @classmethod
    def class_method(cls, a):
        '''
        class method
        '''
        print(f"Foo : class_method(cls, a) = {a}")
        return a

    @staticmethod
    def static_method(a):
        '''
        static method
        '''
        print(f"Foo : static_method(a) = {a}")
        return a


foo = Foo
print(foo.x, foo._x)
# print(foo.__x)  # AttributeError: type object 'Foo' has no attribute '__x'
# print(foo.y, foo._y, foo.__y) # AttributeError: type object 'Foo' has no attribute 'y'

print(foo.instance_method)
print(foo.class_method)
print(foo.static_method)

print(Foo.instance_method)
print(Foo.class_method)
print(Foo.static_method)

foo = Foo()
print(foo.x, foo._x)
print(foo.y, foo._y)

print(foo.class_method(2))
print(foo.static_method(2))
print(foo.instance_method(2))

# print(Foo.instance_method(2)) #TypeError: instance_method() missing 1 required positional argument: 'a'
print(Foo.class_method(2))
print(Foo.static_method(2))


```
