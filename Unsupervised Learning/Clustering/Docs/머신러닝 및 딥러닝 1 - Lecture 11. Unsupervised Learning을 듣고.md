 ## 데이터사이언스를 위한 머신러닝 및 딥러닝 1 - Lecture 11. Unsupervised Learning을 듣고.

 Class info: [링크](https://gsds.snu.ac.kr/course/m3239-004600/)  
 Youtube: [링크](https://www.youtube.com/watch?v=eTqt4oRfXrA)


## Unsupervised Larning
- 비지도 학습
- Labeling(라벨링)이 되어 있지 않은 데이터에 대해 흥미있는 무언가를 찾아내는 것.
    - 흥미있는 무언가?
        - 흥미있는 무언가의 '차원'에 대해서 생각해보자면, 전체 데이터셋의 차원보다 훨씬 작을 확률이 높다.
        - 예를 들어, MNIST 데이터셋에서 전체 데이터는 256 x 28 x 28 사이즈로 주어지지만, 그 중에 의미있는 데이터는 10가지 (0 ~ 9 손글씨) 의 데이터 그룹이다.
- 사실, 라벨링이 되어 있는 데이터는 꽤나 비싸다.
- 크롤링 등을 이용한 광범위한 빅데이터 수집에서 데이터가 라벨링 되어 있지 않을 경우가 많은데, 그 수많은 데이터 속에 무언가 있다는 생각이 들었다면, 비지도 학습으로 데이터를 분석해 보자.
- 라벨링은 즉, 명확한 정답/목표가 있다는 것이고, 이 라벨링이 없다는 것은 분석가의 주관에 모델의 성과가 달려있다는 뜻이기도 하다.

## Clustering
- 군집화
### Recommendation System (추천 시스템)
- Collaborative filtering
- 협업 필터링
- 컨텐츠에 대한 User들의 평가가 데이터베이스에 쌓이면, 자연스럽게 사용자들을 Clustering 할 수 있음.
- **해당 클러스터의 사용자들이 좋아하는 영상들 중에서 특정 사용자가 보지 않은 영상을 추천한다.**

### What do we need?
> Pairwise Distance Matrix(or Metric)

### Types
- Hierarchical (bottom-up)
    - 어떻게 뭉쳐갈 것인가?
        - 가장 작은 단위에서부터 군집을 하나하나 만들어가자.
        - agglomerative

- Partitional (top-down)
    - 어떻게 나눌 것인가?
        - 나눈 후에 그 결과를 보고 더 나은 분할법을 찾자.

## Agglomerative Clustering
> 두 번째 군집화에서부터 우리는, 두 클러스터간의 거리를 어떻게 구할 것인가 하는 문제에 직면한다.

#### 가장 가까운 두 데이터 포인트 사이의 거리 (Closest pair)를 사용했을 때의 문제점

<p style="text-align:center">
<img src="./assets/closest-pair-problem.png" />
</p>


- 서로 다른 군집이 sparse 한 connection으로도 이어질 수 있다.


## K-Means Algorithm

### Objective Function

$$
Minimize: 
\sum_{i=1}^{N}\sum_{k=1}^{K}r_{i,k}||x_i - \mu_k||^2
$$

### Algorithm
$x_i$: data point

$\mu_k$ : centroid

1. centroid 초기화

2. 수렴할 때까지 반복
   
   1)  모든 데이터 포인트에 대해 어떤 클러스터에 속할지 결정해줌. 
   2) 모든 클러스터를 평균으로 업데이트.

   
$$ 
c^i := \argmin_{j}||x^i - \mu_j||^2
$$

   
$$
\mu_j := \frac{\sum_{i=1}^{m}\mathbb{1}\{c^i = j\}x^i}{\sum_{i=1}^m\mathbb{1}\{c^i = j\}}
$$

### Convergence?
- 이 알고리즘은 끝나는가?
- 그렇다.
- 1번과 2번 과정 모두 목적함수를 '감소'시키거나, 적어도 '유지'시킨다. 
    - 따라서, local optimal에 도달할 것을 보장할 수 있다.
    - 다만, 끝없이 미세하게 작아질 수는 있다. (발산하듯이)
- 모든 데이터 포인트 N개에 대해서 K개의 클러스터에 배정하는 것은 유한하다.
- 이 시간복잡도는 지수적이지만, 대부분 모델을 돌려보면 꽤나 빨리 끝난다 < 어김없는 블랙박스

- 초기값에 영향을 많이 받는다.
  - 여러번 돌려봐야 한다.

### Time Complexity?
- N개의 데이터 포인트를 K개의 클러스터에 대해 할당 : `O(KN)`
- centroid 재조정 : `O(N)`
