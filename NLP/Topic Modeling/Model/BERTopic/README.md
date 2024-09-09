# BERTopic

[원문](https://arxiv.org/pdf/1810.04805)

- 구글의 BERT(Bidirectional Encoder Representations from Transformers) 임베딩을 사용하고, Class 기반의 TF-IDF 변형을 사용한다.

> class-based variation of TF-IDF를 통해 coherent topic representation을 추출, 언어모델의 힘을 빌려 문서를 임베딩하고 클러스터링한다.

문장은 Uni-direction이다. 따라서, Self-Attention 의 경우 한쪽 방향을 마스킹한채로 학습시킨다. 이는 문장 단위의 작업에서 우수한 성적을 보인다.

그러나, Term, 즉 단어 단위의 Task에서는 단어의 앞과 뒤의 양쪽 방향의 맥락을 모두 고려하는 것이 바람직하기 때문에, 본 BERT 모델이 나오게 되었다.

Transformer의 구조를 생각할 때, 인코더와 디코더가 떠오를 것이다. 인코더만을 사용해 양방향 문맥을 이해하는 것이 BERT, 미리 훈련된 디코더만을 사용해 문장을 생성하는 것이 GPT(Generative Pre-trained Transformer)이다.

[너무 좋은 글](https://medium.com/@hugmanskj/%EA%B0%80%EC%9E%A5-%EC%84%B1%EA%B3%B5%EC%A0%81%EC%9D%B8-%ED%8A%B8%EB%9E%9C%EC%8A%A4%ED%8F%AC%EB%A8%B8%EC%9D%98-%EB%B3%80%ED%98%95-bert%EC%99%80-gpt-%EC%86%8C%EA%B0%9C-0b18fb7e563b)
