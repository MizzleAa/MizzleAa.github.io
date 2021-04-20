---
layout: post
title: "python magic method"
date: 2021-04-18 10:20:00 +0000
categories: [python]
---

### Python Magic Method 정리

#### version : 3.7.5

---

```python
class Foo:
    #Magic Method 정리

    def __new__(cls):
        '''
        1. 개채 생성
        새로운 인스턴스를 만들때 제일처음으로 실행되는 메소드 개채를 반환해야함
        '''
        print(f"1 : __new__(cls)")
        return super(Foo, cls).__new__(cls)

    def __init__(self):
        '''
        2. 개채 초기화
        인스턴스가 __new__로 생성되고 나서 호출되는 메소드, 인자를 받아서 내부에 지정해 줄 수 있음
        '''
        self.val = 10
        print(f"2 : __init__(self) = self.val : {self.val}")
        super().__init__()

    def __del__(self):
        '''
        last. 개채 소멸
        객체가 소멸시 수행해야 할 일을 지정, 내부 레퍼런스 카운터가 0이되면 자동 소멸
        '''
        print(f"last : __del__(self)")

    def __str__(self):
        '''
        클래스 객채를 표현하는 수단 : __repr__과 동시선언되어 있다면 우선권이 있음
        __str__ : 객체의 비공식적인(informal) 문자열을 출력할 때 사용
        사용자가 보기 쉬운 형태로 사용
        '''
        print(f"in : __str__")
        val = f"__str__() : Foo Class"

        return val

    def __repr__(self):
        '''
        클래스 객채를 표현하는 수단
        __repr__ 은 공식적인(official) 문자열을 출력할 때 사용
        시스템이 해당 객체를 인식할 수 있도록 사용
        '''
        print(f"in : __repr__")
        val = f"{self.val}"

        return val

    def __bytes__(self):
        '''
        bytes형식으로 반환
        '''
        print(f"in : __bytes__")
        val = f"__bytes__() : Foo Class"
        return str.encode(val)

    def __format__(self, format):
        '''
        format 형식으로 반환
        '''
        print(f"in : __format__")
        val = "__format__() : {}".format(format)
        return val

    def __getattr__(self, name):
        '''
        객체의 없는 속성을 참조하려 할때 호출,
        일반적으로 찾는 속성이 있다면 호출되지 않음
        '''
        print(f"in : __getattr__ Not Found : {name}")
        #return super().__getattr__(name)

    def __setattr__(self, name, value):
        '''
        객체의 속성을 변경할때 호출
        '''
        print(f"in : Set Attribute : {name}:{value}")
        super().__setattr__(name, value)

    def __delattr__(self, name):
        '''
        객체의 속성을 del키워드로 지울 때 호출
        '''
        print(f"in : __delattr__ : {name}")
        return super().__delattr__(name)

    def __dir__(self):
        '''
        객체의 가지고있는 속성을 보여줌
        '''
        print(f"in : __dir__ ")
        return super().__dir__()



foo = Foo()
print(foo)
print(f"__str__ 형식 반환 : {str(foo)}")
print(f"__repr__ 형식 반환 : {repr(foo)}")
print(f"__byte__ 형식 반환 : {bytes(foo)}")
print(f"__format__ 형식 반환 : {format(foo, 'Test')}")

foo.val = 20
print(f"__setattr__ 형식 반환 : {foo.val}")
print(f"__getattr__ 형식 반환 : {getattr(foo,'val')}")

print(f"__delattr__ 형식 반환 : {delattr(foo, 'val')}")
print(f"지워진 후 상태 확인 : {foo.val}")

print(dir(foo))
```
