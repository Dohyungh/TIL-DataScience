# 요약 모델 구현

- 요구사항: 네이버 뉴스 정치, 경제, 사회, 생활/문화, IT/과학, 세계의 6개 카테고리 중에서 3개를 뽑아 클라이언트가 요청하면 그 "요약문"을 뉴스 브리핑의 형태로 제공한다.

- 데이터: 매 정각마다 수행되는 자동화된 크롤링 봇 (직접 구현, Docker run + cron) 이 모은 뉴스 기사 원문

- 하드웨어 사양: 4코어 CPU, 16GB RAM, 320 GB 저장소, 6TB 네트워크 transfer

## 고려사항

- 하드웨어 사양이 낮기 때문에, 고도의 모델을 쓰기 어렵다.

  - 직접 GPU 서버를 구매하는 SSAFY 팀도 있나 보다.
  - 학습을 위한 GPU는 주어진다.
  - 하지만, Inference/Serving을 위한 GPU 서버는 주어지지 않는다.

- 대용량, 다수의 문서를 요약해야 한다.

- 크롤링된 데이터에 무맥락의 데이터 (예: OOO 기자, 이미지에 달린 캡션 (...가 ... 하고 있다 등)) 등이 포함되어 있다.

- 짧은 요약문과 긴 요약문의 두 버전으로 나누어져 있다.

## 최초 구현

Kobart 모델을 사용했다.

```python
from typing import Union, Literal, List
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration
from utils import get_file_patterns, get_merged_document

app = FastAPI(root_path="/ai/briefing")

class JSON_Briefing(BaseModel):
    tags: List[Literal['100','101','102','103','104','105']]

class Briefing(BaseModel):
    shortBriefing : str
    longBriefing : str

class JSON_Briefing_Out(BaseModel):
    data: Briefing


@app.post("/", response_model=JSON_Briefing_Out)
def read_briefing(json: JSON_Briefing):

    shortBriefing = ""
    longBriefing = ""

    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-summarization')
    model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-summarization')

    for tag in json.tags:
        file_names = get_file_patterns(tag)
        merged_document = get_merged_document(tag, file_names)
        raw_input_ids = tokenizer.encode(merged_document)
        input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]
        input_ids = torch.tensor([input_ids])
        long_summary_text_ids = model.generate(
            input_ids = input_ids,
            bos_token_id = model.config.bos_token_id,
            eos_token_id = model.config.eos_token_id,
            length_penalty=1.5, # 길이에 대한 penalty 값. 1보다 작은 경우 더 짧은 문장을 생성하도록 유도하며, 1보다 클 경우 길이가 더 긴 문장을 유도
            max_length = 1024, # 요약문의 최대 길이 설정
            min_length = 1, # 요약문의 최소 길이 설정
            num_beams = 8) # 문장 생성 시 다음 단어를 탐색하는 영역의 개수
        short_summary_text_ids = model.generate(
            input_ids = input_ids,
            bos_token_id = model.config.bos_token_id,
            eos_token_id = model.config.eos_token_id,
            length_penalty=0.5, # 길이에 대한 penalty 값. 1보다 작은 경우 더 짧은 문장을 생성하도록 유도하며, 1보다 클 경우 길이가 더 긴 문장을 유도
            max_length = 256, # 요약문의 최대 길이 설정
            min_length = 1, # 요약문의 최소 길이 설정
            num_beams = 8) # 문장 생성 시 다음 단어를 탐색하는 영역의 개수
        shortBriefing += tokenizer.decode(short_summary_text_ids[0], skip_special_tokens=True)
        longBriefing += tokenizer.decode(long_summary_text_ids[0], skip_special_tokens=True)

    resp = {"shortBriefing": shortBriefing,
            "longBriefing" : longBriefing}

    return {"data": resp}
```

### 문제점

- 두 요약문 중에 하나는 공백이 출력된다.
- 요약이 제대로 되지 않는다.
  - 한 단어가 반복되거나
  - 전체 문장이 반영되지 않거나
  - 문장을 그대로 복사한 것이 요약문이랍시고 출력된다.
- Vocab_size가 모델의 Config 파일에서 정한 것보다 커져서 에러가 발생해 API도 먹통이다.

급하게 협업으로 정한 기한에 맞추기 위해 FastAPI 를 만들다 보니 정작 요청은 가지만 형편없는 작동 수준이었다.
로컬에는 4050 RTX 그래픽카드와 20코어 CPU가 존재하기 때문에 EC2환경에서 돌려보지 못한 것도 문제였다.

## 해결방법 구상

### ~~Topic Modeling~~

다른 AI 담당 팀원이 구현한 LDA 토픽모델링 결과를 바탕으로, 전체 뉴스기사를 토픽과 키워드에 맞추어 한번 걸러주는 작업을 거쳐야 할 것으로 보인다.
다음의 논문을 발견했다.

> [휴리스틱 클러스터링을 활용한 다중 문서 요약 시스템 : 반도체 산업 사례](http://journal.dcs.or.kr/xml/34061/34061.pdf)

Multi document Bart 모델에 대한 논문들이 몇 있는 것으로 보아 Bart 모델의 사용방식이 잘못된 것으로 보인다.

---

### Kmeans Clustering

더 정확히는 Kmedoids 로, 구체적으로 존재하는 문서를 클러스터의 중심으로 삼는다.

클러스터를 구성해 클러스터의 중심을 파악하고, 그에 해당하는 문서'만'을 요약한다. 주말 동안 컴퓨터 앞에 앉아 있지 않고 따로 고민하는 시간이 길었다.
제한된 하드웨어 자원 (4코어 CPU)로 LLM을 돌리는 것은 결국 불가능하다.

    1. Jupyter hub에 원격으로 모델을 돌릴 수 있는 방법을 찾거나
    2. 어쩔 수 없이 LLM 을 포기하고 다른 모델을 찾거나

위 두 가지 방법 중에 선택해야만 했다. 결국 Citrix를 통해서 접속해야만 하는 싸피의 환경상 발생할 수 있는 변수가 너무 많았기 때문에 두번째 방법을 선택했다.

실루엣, 혹은 elbow method로 클러스터의 개수를 선정하고 클러스터링을 돌려서 얻은 문서들에 대해서만 요약한다.

결국, [KOSPI-200 종목에 대한 클러스터링 프로젝트](https://github.com/Dohyungh/PJT-Clustering-KOSPI200) 에서 했던 것의 반복이라 많이 아쉽다.

---

### Bart 모델 사용

모델의 Encoder 부분과 Decoder 부분을 분리해서 생각해야 한다는 것을 알았다.
또, `<s>`, `</s>` 와 같이 `<bos>`, `<eos>` 를 지정해 주어야 하며,
Sequence의 길이가 대체로 **1024** 토크으로 제한되어 있다는 것 또한 알게 되었다.

결론적으로 주어진 문서를 쪼개고, 각각을 요약한 후에 요약본을 다시 요약하는.. 요약의요약의요약.. 느낌으로 수행해야 한다는 것을 알았다.

문서를 어떻게 쪼갤 것인지도 충분히 하나의 고민거리인 듯 하다. 변동사항이 생기면 추가하겠다.

참고링크: [박승재님의 글](https://int-i.github.io/python/2023-08-07/huggingface-pipeline-summarization/)
