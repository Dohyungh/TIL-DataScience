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

### Topic Modeling

다른 AI 담당 팀원이 구현한 LDA 토픽모델링 결과를 바탕으로, 전체 뉴스기사를 토픽과 키워드에 맞추어 한번 걸러주는 작업을 거쳐야 할 것으로 보인다.
다음의 논문을 발견했다.

> [휴리스틱 클러스터링을 활용한 다중 문서 요약 시스템 : 반도체 산업 사례](http://journal.dcs.or.kr/xml/34061/34061.pdf)

Multi document Bart 모델에 대한 논문들이 몇 있는 것으로 보아 Bart 모델의 사용방식이 잘못된 것으로 보인다.
