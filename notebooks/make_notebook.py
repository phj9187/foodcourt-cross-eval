"""Builds cross_eval_colab.ipynb (run: python notebooks/make_notebook.py)."""
import json
from pathlib import Path

def md(s): return dict(cell_type='markdown', metadata={}, source=s)
def code(s): return dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=s)

cells = [
md("""# 교차 평가 (Cross-evaluation)

이미 끝난 AI의 본시험 기록(`paired_test.json`)을 **다시 학습하지 않고** 27개 점수표 모두로 다시 채점합니다.

- 런타임: **CPU, 고용량 RAM** (GPU 불필요: 시간 대부분이 Drive 파일 읽기와 JSON 해석)
- 순서대로 실행하세요. 1번 셀의 Drive 허용은 직접 눌러야 합니다.
- 한 번 뽑은 AI는 캐시되므로, 다시 실행하면 새로 끝난 AI만 읽습니다."""),
code("""# 1. Google Drive 연결
from google.colab import drive
drive.mount('/content/drive')"""),
code("""# 2. 코드 받기
REPO_COMMIT = 'main'
!rm -rf /content/foodcourt-cross-eval
!git clone -q https://github.com/phj9187/foodcourt-cross-eval /content/foodcourt-cross-eval
!cd /content/foodcourt-cross-eval && git checkout -q $REPO_COMMIT && git log --oneline -1
!pip -q install orjson
import os, sys
os.chdir('/content/foodcourt-cross-eval'); sys.path.insert(0, '/content/foodcourt-cross-eval')
print('CPU cores:', os.cpu_count())"""),
code("""# 3. 대상 실험 (끝난 AI만 읽음)
BASE = '/content/drive/MyDrive/foodcourt_adaptive_sim'
EXPERIMENTS = ['extended_main_20261108_001', 'extended_taste_20261110_001']
OUT = f'{BASE}/cross_eval'
WORKERS = 0   # 0 = 코어 수와 메모리로 자동 결정
for e in EXPERIMENTS:
    print(e, os.path.isdir(f'{BASE}/{e}'))"""),
code("""# 4. 추출: paired_test.json -> 작은 .npz (병렬, 캐시)
for e in EXPERIMENTS:
    print('=====', e)
    !python -m cross_eval.extract "{BASE}/{e}" "{OUT}/{e}/cache" --workers {WORKERS}"""),
code("""# 5. 재채점과 요약
for e in EXPERIMENTS:
    print('=====', e)
    !python -m cross_eval.analyze "{OUT}/{e}/cache" "{OUT}/{e}/results"
"""),
code("""# 6. 결과 보기
import pandas as pd
from IPython.display import display, Image
for e in EXPERIMENTS:
    r = f'{OUT}/{e}/results'
    print('=====', e)
    display(pd.read_csv(f'{r}/self_vs_best_stable.csv'))
    display(pd.read_csv(f'{r}/generalist_stable.csv').head(10))
    display(pd.read_csv(f'{r}/common_metrics_stable.csv').sort_values('d_profit_won_per_court_day', ascending=False))
    display(Image(f'{r}/heatmap_stable.png'))"""),
]
nb = dict(nbformat=4, nbformat_minor=5, cells=cells,
          metadata=dict(kernelspec=dict(name='python3', display_name='Python 3'), language_info=dict(name='python'),
                        colab=dict(provenance=[])))
Path(__file__).with_name('cross_eval_colab.ipynb').write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
