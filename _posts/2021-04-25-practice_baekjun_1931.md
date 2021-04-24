---
layout: post
title: "practice baekjun 1931 - 회의실 배정"
date: 2021-04-25 00:40:00 +0000
categories: [practice, greedy, baekjun]
---
# Practice Greedy 

---

## 1. 문제 
### "회의실 배정"
- 색인 : [1931](https://www.acmicpc.net/problem/1931)
- 풀이 시간 : 1시간(시간 오버)
- 문제 :
  - 한 개의 회의실이 있는데 이를 사용하고자 하는 N개의 회의에 대하여 회의실 사용표를 만들려고 한다. 
  - 각 회의 I에 대해 시작시간과 끝나는 시간이 주어져 있고, 각 회의가 겹치지 않게 하면서 회의실을 사용할 수 있는 회의의 최대 개수를 찾아보자. 
  - 단, 회의는 한번 시작하면 중간에 중단될 수 없으며 한 회의가 끝나는 것과 동시에 다음 회의가 시작될 수 있다. 
  - 회의의 시작시간과 끝나는 시간이 같을 수도 있다. 
  - 이 경우에는 시작하자마자 끝나는 것으로 생각하면 된다.

- 입력 조건 : 
  - 첫째 줄에 회의의 수 N(1 ≤ N ≤ 100,000)이 주어진다.
  - 둘째 줄부터 N+1 줄까지 각 회의의 정보가 주어지는데 이것은 공백을 사이에 두고 회의의 시작시간과 끝나는 시간이 주어진다.
  - 시작 시간과 끝나는 시간은 2^31-1보다 작거나 같은 자연수 또는 0이다.

- 출력 조건 : 
  - 첫째 줄에 최대 사용할 수 있는 회의의 최대 개수를 출력한다.

- 입력 예시 :
  ```
  11
  1 4
  3 5
  0 6
  5 7
  3 8
  5 9
  6 10
  8 11
  8 12
  2 13
  12 14
  ```  

- 출력 예시 :
  ```
  4
  ```

## 2. 풀이
~~~python
'''
# 알고리즘 사고적 생각 정리
N = 11

start_list = [1,3,0,5,3,5,6,8,8,2,12]
end_list = [4,5,6,7,8,9,10,11,12,13,14]
list = [
    [start,end],
    [start,end],
    [start,end],
]
list.sort()

count = 0
s = start_list[0]
e = end_list[0]
print(s,e)
count += 1
#없으면 +1
t = start_list.index(e)

s1 = start_list[t]
e1 = end_list[t]

count += 1
t = start_list.index(e1)

s2 = start_list[t]
e2 = end_list[t]

count += 1
t = start_list.index(e2)

s3 = start_list[t]
e3 = end_list[t]

count += 1
'''

def set_input():
    size = int(input())
    value_list = []
    for _ in range(int(size)):
        value_list.append(
            list(map(int, input().split()))
        )
    value_list.sort(
        key = lambda v : (v[1],v[0])
    )
    return value_list, size

def main():
    value_list, size = set_input()

    count = 0
    finish = -1
    for i in range(size):
        if value_list[i][0] >= finish:
            count += 1
            finish = value_list[i][1]

    print(count)

if __name__ == '__main__':
    main()

~~~