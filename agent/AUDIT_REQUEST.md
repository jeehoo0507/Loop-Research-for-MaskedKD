# AUDIT_REQUEST — [claude] 2026-10-31 라운드 47 (겹침 주장만 단독 재요청)

## 지난 판정과 고친 점
03:57 감사(codex)는 두 주장 모두 NEEDS-MORE였다. 주장 1(suffdisc@12 vs maskedkd@12 final)의 이유는 final seed 1~3이 승격 전 screen에서 이미 쓰였다는 것이다. 맞는 지적이다. harness/train.py 는 tier와 상관없이 `torch.manual_seed(seed)`와 loader seed를 쓰므로, final s1~s3은 승격 판단에 쓴 screen s1~s3과 초기화·데이터 순서가 같다. 그래서 주장 1은 이번 요청에서 **뺐다**. 대신 한 번도 쓰지 않은 seed 4·5·6에서 독립 재현(PROPOSALS H-W30, 결과 전에 읽기 규칙 등록)을 제출했다. 끝나면 주장 1을 그 결과와 함께 다시 낸다.
주장 2에 대해 감사관은 "analysis_suffdisc_decomp_1030_202612_878c explicitly supports the stated scope and values"라고 적었다. 판정이 NEEDS-MORE였던 것은 주장 1과 함께 묶였기 때문으로 보고, 같은 범위·같은 숫자로 단독 재요청한다.

## 주장 (1줄)
승격 전 네 screen-student 시점에서 clean-val eval-mode suffdisc/MaskedKD 토큰 겹침은 K=12의 10.65–14.14%다.

## 근거 분석
analysis_suffdisc_decomp_1030_202612_878c . summary.json `overlap` 첫 열(`suff_mkd`, K 대비 비율): suffdisc student 0.1065 / 0.1414 / 0.1372 / 0.1326, center student 0.2189 / 0.2721. 선택은 clean val에서 eval mode로 한 것이다. 같은 분석의 `table`에서 suffdisc_s0_final student 기준으로, K=12 토큰만 받은 masked teacher의 t_acc/t_wga는 suffdisc 선택 84.57/48.12, maskedkd 선택 72.23/28.57이다(보조 근거, 주장에는 넣지 않음).

## 범위 (주장하지 않는 것)
- train mode·augmentation을 켠 학습 중 선택의 겹침은 주장하지 않는다.
- final student의 겹침, 다른 K의 겹침은 주장하지 않는다.
- 이 겹침이 성능 차이의 원인이라는 인과는 주장하지 않는다.

## 무결성
agent 커밋 중 harness/·tests/를 바꾼 것은 없다. test_eval은 쓰지 않았다. 이번 요청은 run id 0개, 분석 1개를 인용한다.

### verdict
CLAIM: <the claim, one line>
VERDICT: PASS | FAIL | NEEDS-MORE
REASON: <one or two lines, with run ids>
