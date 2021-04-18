---
layout: post
title:  "python abstract"
date:   2021-04-18 10:20:00 +0000
categories: [python]
---
Python abstract 정리
{% highlight python %}

class Vector(metaclass=ABCMeta):
    '''
        추상 메소드 정의
    '''
    def __init__(self):
        pass


    @abstractmethod
    def up(self):
        '''
            추상 클레스의 메서드를 빈 메서드 형태로 변환
            추상 클래스는 인스턴스를 만들 수 없음
        '''
        pass

    @abstractmethod
    def down(self):
        pass

    @abstractmethod
    def left(self):
        pass

    @abstractmethod
    def right(self):
        pass

class Keyboard(Vector):
    
    def __init__(self, step=1):
        '''
            2차원 공간좌표를 dict형태로 표현(예시를 위한 데이터)
            dict {x : 0, y : 0}
        '''
        self.__point = {'x':0,'y':0}
        self.__step = step
        
    def up(self):
        self.__point['y'] += self.__step

    def down(self):
        self.__point['y'] -= self.__step

    def right(self):
        self.__point['x'] += self.__step

    def left(self):
        self.__point['x'] -= self.__step
    
    def __position(self, point):
        '''
            private
        '''
        self.__point = point

    def __add__(self, objects):
        point = {
            'x':self.__point['x']+objects.__point['x'],
            'y':self.__point['y']+objects.__point['y']
        }
        obj = Keyboard()
        obj.__position(point)
        return obj

    def position(self):
        '''
            2차원 공간좌표 반환
        '''
        return self.__point


a = Keyboard()
b = Keyboard()

a.up()
a.left()

b.down()
b.right()

c = a+b

print(f"a.up -> a.left = {a.position()}")
print(f"b.down -> b.right = {b.position()}")
print(f"a + b = {c.position()}")

{% endhighlight %}


