---
layout: post
title:  "python built-in function"
date:   2021-04-18 10:25:00 +0000
categories: [python]
---
### Python built-in function 정리
#### version : 3.7.5

```python

################################################
# docs 
# = 숫자의 절댓값을 돌려줍니다. 인자는 정수, 실수 또는 __abs__()를 구현하는 객체입니다. 인자가 복소수면 그 크기가 반환됨
# abs(x)
# = 절대값
# x값이 실수형일 경우 음수값이 양수로 변환

def fun():
    x = -10
    foo = abs(x)
    print(f"x = {x}")
    print(f"foo = {foo}")

################################################
# all(iterable)
# = iterable 의 모든 요소가 참이면 (또는 iterable 이 비어있으면) True 를 돌려줌
# iterable = [1,2,None] 비어있는 값이 있을경우 false 반환

def fun():
    def all(iterable):
        for element in iterable:
            if not element:
                return False
        return True

    #iterable = [1,2,None]
    iterable = [1,2,3]
    foo = all(iterable)
    print(f"iterable = {iterable}")
    print(f"foo={foo}")

################################################
# any(iterable)
# = iterable 의 요소 중 어느 하나라도 참이면 True 를 돌려줍니다. iterable이 비어 있으면 False 를 돌려줌
# iterable = [1,2,None] 

def fun():
    def any(iterable):
        for element in iterable:
            if element:
                return True
        return False

    iterable = [1,2,None]
    #iterable = [1,2,3]
    foo = any(iterable)
    print(f"iterable = {iterable}")
    print(f"foo={foo}")

################################################
# ascii(object)
# repr() 처럼, 객체의 인쇄 가능한 표현을 포함하는 문자열을 반환하지만, 
# '\x' 나 '\u' 또는 '\U' 이스케이프를 사용하여 repr() 이 돌려주는 문자열에 포함된 비 ASCII 문자를 이스케이프 
# 이것은 파이썬 2의 repr() 이 돌려주는 것과 비슷한 문자열 생성
# '\x01', '\u0001', '\U00000001' 자리수

def fun():
    object = "\x01"
    object = "\u0001"
    object = "\U00000001"
    
    foo = ascii(object)
    print(f"obejct = {object}")
    print(f"foo={foo}")

################################################
# bin(x)
# 정수를 《0b》 가 앞에 붙은 이진 문자열로 변환 
# x 가 파이썬 int 객체가 아니라면, 정수를 돌려주는 __index__() 메서드를 정의

def fun():
    b1 = bin(3)
    b2 = bin(-10)
    b3 = format(14, '#b')
    b4 = format(14, 'b')
    print(f"bin(3) = {bin(3)}")
    print(f"bin(-10) = {bin(-10)}")
    print(f"format(14, '#b') = {format(14, '#b')}")
    print(f"format(14, 'b') = {format(14, 'b')}")

################################################
# bool([x])
# 논리값, 즉 True 또는 False 중 하나를 돌려줌
def fun():
    bar = bool(True)
    print(f"bool(True)={bar}")

################################################
# breakpoint 
# 이 함수는 호출 지점에서 디버거로 진입하게 만듬
# 특히 sys.breakpointhook() 을 호출하고 args 와 kws 를 그대로 전달

# c - continue로 다음에 설정 된 중단점으로 바로 이동
# n - 다음 줄로 이동 함
# r - 현재 함수가 return 될 때까지 계속 실행
# l - 현재 라인을 포함하여 위 아래로 11줄의 코드를 출력함

def fun():
    import sys
    import pdb

    #pdb.set_trace()
    breakpoint()

    def foo():
        var = 10
        print(var)

    foo()

################################################
# bytearray(size)
# 새로운 바이트 배열을 돌려줌
# bytearray 클래스는 0 <= x < 256 범위에 있는 정수의 가변 시퀀스

def fun():
    foo = bytearray(10)
    print(f"bytearray(10) = {foo}")
    val = "ABCD".encode()
    foo = bytearray(val)
    print(f"bytearray({val}) = {foo}")
    print(f"ABCD bytearray({foo}) = {foo[0]},{foo[1]},{foo[2]},{foo[3]}")

################################################
# bytes
# 새로운 《바이트열》 객체를 돌려줌
# 객체는 0 <= x < 256 범위에 있는 정수의 불변 시퀀스
def fun():
    foo = bytes(10)
    print(f"bytes(10) = {foo}")
    val = "ABCD".encode()
    foo = bytes(val)
    print(f"bytes({val}) = {foo}")
    print(f"ABCD bytes({foo}) = {foo[0]},{foo[1]},{foo[2]},{foo[3]}")

################################################
# callable(object)
# object 인자가 콜러블인 것처럼 보이면 True를, 그렇지 않으면 False 를 돌려줌
# 이것이 True를 돌려줘도 여전히 호출이 실패할 가능성이 있지만, False일 때 object 를 호출하면 반드시 실패
# __call__ 을 선언시 사용 가능
def fun():
    val = 1
    foo = callable(val)
    print(f"callable(val) = {foo}")
    class fooClass:
        def __init__(self):
            pass
        #def __call__(self):
        #    print('callable')
    print(f"callable(fooClass) = {callable(fooClass)}")
    foo_class = fooClass()

    print(f"callable(fooClass) = {callable(foo_class)}")

################################################
# chr(j)
# 유니코드 코드 포인트가 정수 i 인 문자를 나타내는 문자열을 돌려줌
# 예를 들어, chr(97) 은 문자열 'a' 를 돌려주고, chr(8364) 는 문자열 '€' 를 돌려줌
# ord() 의 반대

def fun():
    val = 97
    foo = chr(val)
    print(f"chr({val}) = {foo}")

################################################
# @classmethod
# 메서드를 클래스 메서드로 변환
# 인스턴스 메서드가 인스턴스를 받는 것처럼, 클래스 메서드는 클래스를 묵시적인 첫 번째 인자로 받음
# 클래스 메서드를 선언하려면 이 관용구를 사용
# cls = 클레스 호출인자를 불러옴

def fun():
    class Foo:
        num = 10
        @classmethod
        def tmp(cls):
            return 20

        @classmethod
        def function(cls, val):
            return val + cls.num * cls.tmp()

    foo = Foo()
    val = foo.function(10)
    print(f"foo.function(10) = {val}")
    val = Foo.function(10)
    print(f"Foo.function(10) = {val}")

################################################
# compile(source, filename, mode)
# source 를 코드 또는 AST 객체로 컴파일
# 코드 객체는 exec() 또는 eval() 로 실행
# source 는 일반 문자열, 바이트열 또는 AST 객체
# 문자열을 수식으로 반환
def fun():
    #val = "pow(2,16)"
    a = 10
    b = 2
    x = 3

    val = f"{a}*{x}+{b}"
    foo = compile(val,"<string>","eval")
    res = eval(foo)
    print(f"compile(val,'<string>','eval') = {foo}")
    print(f"eval(foo) = {res}")

################################################
# delattr(object, name)
# 이것은 setattr() 의 친척뻘
# 인자는 객체와 문자열
# 문자열은 객체의 어트리뷰트 중 하나의 이름
# 이 함수는 객체가 허용하는 경우 명명된 어트리뷰트를 삭제
def fun():
    class Foo:
        def __init__(self, x):
            self.x = x

    foo = Foo(10)
    print(f"foo.x = {foo.x}")
    delattr(foo,'x')
    # del foo.x
    # 이미 선언되어있던
    print(f"delattr(foo.x) = {delattr(foo,'x')}") #error

################################################
# dict
# 새 딕셔너리를 만듭니다. dict 객체는 딕셔너리 클래스
def fun():
    foo = dict()
    foo.update( {'val1':1,'val2':2} )
    
    print(f"foo['val1'] = {foo['val1']}")
    print(f"foo.get('val1') = {foo.get('val1')}")

    print(f"foo.keys() = {foo.keys()}")
    print(f"foo.values() = {foo.values()}")

    print(f"del foo['val1'] = {foo}")

################################################
# dir([object])
# 인자가 없으면, 현재 지역 스코프에 있는 이름들의 리스트를 돌려줌
# 인자가 있으면, 해당 객체에 유효한 어트리뷰트들의 리스트를 돌려주려고 시도
def fun():
    class Foo:
        def __init__(self):
            pass
        def test(self):
            pass

    foo = dir(Foo)
    print(f"dir(Foo) = {foo}")

################################################
# divmod(a,b)
# 두 개의 (복소수가 아닌) 숫자를 인자로 취하고 정수 나누기를 사용할 때의 몫과 나머지로 구성된 한 쌍의 숫자를 돌려줌
def fun():
    a = 5
    b = 3
    foo = divmod(a,b)

    print(f"divmod(a,b) = {divmod(a,b)}")

################################################
# enumerate
# 열거 객체를 돌려줌 
# iterable 은 시퀀스, 이터레이터 또는 이터레이션을 지원하는 다른 객체
def fun():
    seasons = ['Spring', 'Summer', 'Fall', 'Winter']
    foo = list(enumerate(seasons))
    print(f"list(enumerate(seasons)) = {foo}")
    foo = list(enumerate(seasons, start=3 ))
    print(f"list(enumerate(seasons, start=3 )) = {foo}")

    def test(sequence, start=0):
        n = start
        for elem in sequence:
            yield n, elem
            n += 1
            print(n, elem)
    foo = list(test(seasons,5))
    print(f"list(test(sequence, start=5)) = {foo}")

################################################
# eval(expression)
# 인자는 문자열 및 선택적 globals 및 locals
# globals 는 딕셔너리
# 제공되는 경우, locals 는 모든 매핑 객체
def fun():
    val = 2
    foo = eval('val*2+1')
    print(f"eval('val*2+1') = {foo}")


################################################
# exec(str,dict)
# 모든 경우에, 선택적 부분을 생략하면, 현재 스코프에서 코드가 실행
# 
def fun():
    val = {"val":5, "foo":0}
    exec("foo = val + 4",val)
    print(f"exec('foo = val + 4',val) = {val['foo']}")

################################################
# filter(function, iterable)
# function 이 참을 돌려주는 iterable 의 요소들로 이터레이터를 구축
#  iterable 은 시퀀스, 이터레이션을 지원하는 컨테이너 또는 이터레이터 

def fun():
    val = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    def is_even(n):
        return True if n % 2 == 0 else False
    #foo = filter(is_even, val)
    foo = filter(lambda x : x%2==0, val)
    print(f"filter(lambda x : x%2==0, val) = {list(foo)}")

################################################
# float
# 실수형

def fun():
    # sign = + / -
    # infinity = inf
    # nan = nan
    # numeric_value = floatnuber | infinity | nan
    # numeric_string = numeric_value
    val = float('+1.23')
    print(f"float('+1.23') = {val}")
    val = float('-123.456 \n')
    print(f"float('-123.456 \\n') = {val}")
    val = float('1e-003')
    print(f"float('1e-003') = {val}")
    val = float('+1e6')
    print(f"float('+1e6') = {val}")
    val = float('-infinity')
    print(f"float('-infinity') = {val}")
      
################################################
# format
# format_spec 의 제어에 따라, value 를 《포맷된》 표현으로 변환
# format_spec 의 해석은 value 인자의 형에 의존하지만, 대부분의 내장형에 의해 사용되는 표준 포매팅 문법
def fun():
    a = 10
    b = 20
    val = "foo = {0} x {1} = {2}".format(a,b,a*b)
    print(f"'foo = {0} x {1} = {2}'.format(a,b,a*b) = {val}")

################################################
# frozenset([iterable])
# 새 frozenset 객체를 돌려주는데, 선택적으로 iterable 에서 가져온 요소를 포함
# set과 동일한데 set보다 우성

def fun():
    val = frozenset() 
    print(f"frozenset() = {val}")
    val = frozenset((0, 1, 2, 3, 1, 2, 3))
    print(f"frozenset((0, 1, 2, 3)) = {val}")
    val = frozenset([0, 1, 2, 3, 1, 2, 3])
    print(f"frozenset([0, 1, 2, 3]) = {val}")
    val = frozenset(range(4))
    print(f"frozenset(range(4)) = {val}")

################################################
# getattr(object, name[, default])
# 주어진 이름의 object 어트리뷰트를 돌려줌
# 명명된 어트리뷰트가 없으면, default 가 제공되는 경우 그 값이 반환
# 그렇지 않으면 AttributeError 가 발생

def fun():
    class Foo:
        val = 10
        def __init__(self):
            pass

    foo = Foo()
    val = getattr(foo, 'val') # foo.val
    print(f"getattr(foo, 'val') = {val}")

################################################
# globals()
# 현재 전역 심볼 테이블을 나타내는 딕셔너리를 돌려줌
# 이것은 항상 현재 모듈의 딕셔너리
# 함수 또는 메서드 내에서, 이 모듈은 그것들을 호출하는 모듈이 아니라, 그것들이 정의된 모듈

def fun():
    #def fun를 없에고 수행 x = 10
    x = 20 
    print(f"x = {x}")
    def foo():
        global x
        x = 10
    print(f"global x = {x}")    
        
################################################
# hasattr(object, name)
# 인자는 객체와 문자열
# 문자열이 객체의 속성 중 하나의 이름이면 결과는 True
# 그렇지 않으면 False
def fun():
    class Foo:
        val = 10
        def __init__(self):
            pass
    foo = Foo()
    val = hasattr(foo,'val') #true
    #val = hasattr(foo,'aa') #false
    
    print(f"hasattr(foo,'val') = {val}")

################################################
# hash(object)
# 객체의 해시값을 돌려줍
# 해시값은 정수
# 딕셔너리 조회 중에 딕셔너리 키를 빨리 비교하는 데 사용
# 같다고 비교되는 숫자 값은 같은 해시값을 갖습
# key = 8106883684374178537 : 재실행할 시 계속 달라짐
def fun():
    val = "test"
    print(f"hash(val) = {hash(val)}")

################################################
# help([object])
# 내장 도움말 시스템을 호출
# 
def fun():
    
    val = 10
    class Foo:
        """
        주석란은 help에 나타납니다.
        """
        def __init__(self):
            """
                self.others = 20
            """
            self.others = 20
    help(Foo)
    #help(str)

################################################
# hex(x)
# 정수를 《0x》 접두사가 붙은 소문자 16진수 문자열로 변환
#
def fun():
    print(f"hex(16) = {hex(16)}")

################################################
# id(object)
# 객체의 《아이덴티티》를 돌려준
# 이것은 객체의 수명 동안 유일하고 바뀌지 않음이 보장되는 정수
# This is the address of the object in memory.
def fun():
    val = 0
    print(f"id({val}) = {id(val)}")
    val = val + 1
    print(f"id({val}) = {id(val)}")
    val = val + 1
    print(f"id({val}) = {id(val)}")
    
################################################
# input([prompt])
# prompt 인자가 있으면, 끝에 개행 문자를 붙이지 않고 표준 출력
# 그런 다음 함수는 입력에서 한 줄을 읽고, 문자열로 변환해서 (줄 끝의 줄 바꿈 문자를 제거한다) 돌려줌
# scanf
def fun():
    val = input("scanf : ")
    print(val)

################################################
# int([x])
# 숫자 나 문자열 x 로 부터 만들어진 정수 객체를 돌려줌
def fun():
    val = int(10)
    print(f"int(10) = {val}")


################################################
# isinstance(object,classinfo)
# object 인자가 classinfo 인자 또는 그것의 (직접, 간접 혹은 가상) 서브 클래스의 인스턴스면 True를 돌려줍

def fun():
    val = isinstance(1,int)
    print(f"isinstance(1,int) = {val}")

    val = isinstance(1.2,int)
    print(f"isinstance(1.2,int) = {val}")
    
    class Foo:
        def __init__(self):
            pass
    
    foo = Foo()
    val = isinstance(foo,Foo)
    print(f"isinstance(foo,Foo) = {val}")

################################################
# issubclass(class,classinfo)

################################################
# iter(object[, sentinel])

################################################
# len(s)

################################################
# list([iterable])

################################################
# locals()

###############################################
# map(function, iterable)

################################################
# max(iterable, *[, key, default])

################################################
# min(iterable, *[, key, default])

################################################
# next(iterator, [, default])

################################################
# object

################################################
# oct(x)

################################################
# open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None)

################################################
# ord(c)

################################################
# pow(base, exp[, mod])

################################################
# print(*objects, sep=' ', end='\n', file=sys.stdout, flush=False)

################################################
# property(fget=None, fset=None, fdel=None, doc=None)

################################################
# range(start, stop[, step])

################################################
# repr(object)

################################################
# reversed(seq)

################################################
# round(number[, ndigits])

################################################
# set([iterable])

################################################
# setattr(object, name, value)

################################################
# slice(start, stop[, step])

################################################
# sorted(iterable, *, key=None, reverse=False)

################################################
# @staticmethod

################################################
# str(object=b'', encoding='utf-8', errors='strict')

################################################
# sum(iterable, /, start=0)

################################################
# super([type[, object-or-type]])

################################################
# tuple([iterable])

################################################
# type(object)

################################################
# type(name, bases, dict, **kwds)

################################################
# vars([object])

################################################
# zip(*iterables)

# fun() #실행함수

```