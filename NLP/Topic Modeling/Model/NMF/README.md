# NMF(Non-negative Matrix Factorization)

- 음수 미포함 행렬 분해
- 음의 원소가 없는 행렬을 두 개의 음의 원소가 없는 행렬로 나눈다.

- 행은 topic, 열은 문서가 된다.
- 가정:
  - 각각의 document는 topic의 선형 조합이고,
  - 각각의 topic은 term들의 선형 조합이다.
- 차원 축소와 feature 추출이 목표이다.
- 원본 행렬이 feature matrix와 coefficient matrix로 분해된다.
- 작은 데이터셋과 짧은 문장에 적합하다.
