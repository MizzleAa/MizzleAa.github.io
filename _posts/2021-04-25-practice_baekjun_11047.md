---
layout: post
title: "practice baekjun 11047 - 동전 0"
date: 2021-04-25 00:42:00 +0000
categories: [practice, greedy, baekjun]
---
# Practice Greedy 

---
## 1. 문제 
### "동전 0"
- 색인 : [11047](https://www.acmicpc.net/problem/11047)
- 풀이 시간 : 12분
- 문제 :
  - 준규가 가지고 있는 동전은 총 N종류이고, 각각의 동전을 매우 많이 가지고 있다.
  - 동전을 적절히 사용해서 그 가치의 합을 K로 만들려고 한다. 
  - 이때 필요한 동전 개수의 최솟값을 구하는 프로그램을 작성하시오.

- 입력 조건 : 
  - 첫째 줄에 N과 K가 주어진다. (1 ≤ N ≤ 10, 1 ≤ K ≤ 100,000,000)
  - 둘째 줄부터 N개의 줄에 동전의 가치 Ai가 오름차순으로 주어진다.(1 ≤ Ai ≤ 1,000,000, A1 = 1, i ≥ 2인 경우에 Ai는 Ai-1의 배수)

- 출력 조건 : 
  - 첫째 줄에 K원을 만드는데 필요한 동전 개수의 최솟값을 출력한다

- 입력 예시 :
  ```
  10 4200
  1
  5
  10
  50
  100
  500
  1000
  5000
  10000
  50000
  ```  

- 출력 예시 :
  ```
  6
  ```

## 2. 풀이
~~~python
# source code
'''
# 알고리즘 사고적 생각 정리
out += value//coin
value %= coin
'''
def set_input():
    size, value = map(int, input().split())
    
    coin_list = []
    for _ in size:
        coin = input()
        coin_list.append(int(coin))

    return value, coin_list

def main():   
    value, coin_list = set_input()
    coin_list.reverse()

    out = 0
    for coin in coin_list:
        out += value//coin
        value %= coin

    print(out)

if __name__ == '__main__':
    main()
    
~~~