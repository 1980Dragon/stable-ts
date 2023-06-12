# Stabilizing Timestamps for Whisper

신뢰도 높은 타임스탬프 산출을 위해 [OpenAI's Whisper](https://github.com/openai/whisper)의 코드를 수정한 스크립트.

https://user-images.githubusercontent.com/28970749/225826345-ef7115db-51e4-4b23-aedd-069389b8ae43.mp4

* [설치](#설치)
* [사용법](#사용법)
  * [Transcribe](#transcribe)
  * [출력](#output)
  * [단어 재결합(Regrouping Words)](#regrouping-words)
  * [Locating Words](#locating-words)
  * [Boosting Performance](#boosting-performance)
  * [Visualizing Suppression](#visualizing-suppression)
  * [Encode Comparison](#encode-comparison)
  * [Tips](#tips)
* [Quick 1.X → 2.X Guide](#quick-1x--2x-guide)

## 설치
```
pip install -U stable-ts
```

최신 커밋 설치:
```
pip install -U git+https://github.com/jianfch/stable-ts.git
```

## 사용법
커맨드라인에서의 사용법 목록과, 그에 상응하는 파이썬 사용법. 

### Transcribe
```commandline
stable-ts audio.mp3 -o audio.srt
```
```python
import stable_whisper
model = stable_whisper.load_model('base')
result = model.transcribe('audio.mp3')
result.to_srt_vtt('audio.srt')
```
파라미터: 
[load_model()](https://github.com/jianfch/stable-ts/blob/d30d0d1cfb5b17b4bf59c3fafcbbd21e37598ab9/stable_whisper/whisper_word_level.py#L637-L652), 
[transcribe()](https://github.com/jianfch/stable-ts/blob/d30d0d1cfb5b17b4bf59c3fafcbbd21e37598ab9/stable_whisper/whisper_word_level.py#L75-L199)
### 출력
Stable-ts는 다양한 포멧의 출력 형식을 지원합니다.
```python
result.to_srt_vtt('audio.srt') #SRT
result.to_srt_vtt('audio.vtt') #VTT
result.to_ass('audio.ass') #ASS
result.to_tsv('audio.tsv') #TSV
```
파라미터: 
[to_srt_vtt()](https://github.com/jianfch/stable-ts/blob/d30d0d1cfb5b17b4bf59c3fafcbbd21e37598ab9/stable_whisper/text_output.py#L267-L291),
[to_ass()](https://github.com/jianfch/stable-ts/blob/d30d0d1cfb5b17b4bf59c3fafcbbd21e37598ab9/stable_whisper/text_output.py#L401-L434),
[to_tsv()](https://github.com/jianfch/stable-ts/blob/d30d0d1cfb5b17b4bf59c3fafcbbd21e37598ab9/stable_whisper/text_output.py#L335-L353)
<br /><br />
타임스탬프에는 단어단위(word-level)와 분절단위(segment-level)가 있으며, 모든 출력 포멧이 둘 다를 지원합니다. 
TSV를 제외하면 두 가지를 동시에 지원하기도 합니다. 
기본설정상, 단어단위와 분절단위는 동시지원하는 모든 포멧에서 둘 다 'True'로 설정되어 있습니다.<br /><br />
VTT 형식에서의 예시.

디폴트: `segment_level=True` + `word_level=True` or `--segment_level true` + `--word_level true` for CLI
```
00:00:07.760 --> 00:00:09.900
But<00:00:07.860> when<00:00:08.040> you<00:00:08.280> arrived<00:00:08.580> at<00:00:08.800> that<00:00:09.000> distant<00:00:09.400> world,
```

`segment_level=True`  + `word_level=False` (주의: `segment_level=True`가 사용됨)
```
00:00:07.760 --> 00:00:09.900
But when you arrived at that distant world,
```

`segment_level=False` + `word_level=True` (주의: `word_level=True`가 사용됨)
```
00:00:07.760 --> 00:00:07.860
But

00:00:07.860 --> 00:00:08.040
when

00:00:08.040 --> 00:00:08.280
you

00:00:08.280 --> 00:00:08.580
arrived

...
```

#### JSON
추후의 재가공을 위해서 모든 데이터를 보존하려면 출력결과를 JSON 형태로 저장할 수 있습니다.
이 방식은 처음부터 다시 재작업을 하지 않고도 서로 다른 가공방법을 테스트할 수 있으므로 유용합니다.
```commandline
stable-ts audio.mp3 -o audio.json
```
```python
# JSON 형식으로 저장:
result.save_as_json('audio.json')
```
JSON 파일을 SRT 형식으로 가공.
```commandline
stable-ts audio.json -o audio.srt
```
```python
# 출력결과를 로드:
result = stable_whisper.WhisperResult('audio.json')
result.to_srt_vtt('audio.srt')
```

### 단어 재결합(Regrouping Words)
Stable-ts는 더 자연스럽게 나뉘어지도록 다른 분절로 단어를 그룹화하는 프리셋(preset)을 갖추고 있습니다.
이 프리셋은 `regroup=True` 옵션을 통해 사용할 수 있으며, 기본적으로 켜져 있습니다. 

하지만 이 외에도 다른 [재결합 방식](#regrouping-methods)이 있으며, 재결합 알고리즘을 커스터마이징하도록 해 줍니다.
위의 프리셋은 단순히 미리 정의된 재결합 방식의 조합 중 하나일 뿐입니다.

https://user-images.githubusercontent.com/28970749/226504985-3d087539-cfa4-46d1-8eb5-7083f235b429.mp4

```python
# 아래의 예시는 기능적으로 동일합니다:
result0 = model.transcribe('audio.mp3', regroup=True) # 리그룹(regroup) 기본설정이 True입니다.
result1 = model.transcribe('audio.mp3', regroup=False)
(
    result1
    .split_by_punctuation([('.', ' '), '。', '?', '？', ',', '，'])
    .split_by_gap(.5)
    .merge_by_gap(.15, max_words=3)
    .split_by_punctuation([('.', ' '), '。', '?', '？'])
)
result2 = model.transcribe('audio.mp3', regroup='sp=.* /。/?/？/,/，_sg=.5_mg=.15+3_sp=.* /。/?/？')
```
모든 재결합 알고리즘(regrouping algorithm)은 문자열로 표현될 수 있습니다. [이곳](https://github.com/jianfch/stable-ts/discussions/162)에서 자유롭게 공유하세요.
#### 재결합 방법(Regrouping Methods)
- [regroup](https://github.com/jianfch/stable-ts/blob/7c6953526dce5d9058b23e8d0c223272bf808be7/stable_whisper/result.py#L808-L854)
- [split_by_gap()](https://github.com/jianfch/stable-ts/blob/7c6953526dce5d9058b23e8d0c223272bf808be7/stable_whisper/result.py#L526-L543)
- [split_by_punctuation()](https://github.com/jianfch/stable-ts/blob/7c6953526dce5d9058b23e8d0c223272bf808be7/stable_whisper/result.py#L579-L595)
- [split_by_length()](https://github.com/jianfch/stable-ts/blob/7c6953526dce5d9058b23e8d0c223272bf808be7/stable_whisper/result.py#L637-L658)
- [merge_by_gap()](https://github.com/jianfch/stable-ts/blob/7c6953526dce5d9058b23e8d0c223272bf808be7/stable_whisper/result.py#L547-L573)
- [merge_by_punctuation()](https://github.com/jianfch/stable-ts/blob/7c6953526dce5d9058b23e8d0c223272bf808be7/stable_whisper/result.py#L599-L624)
- [merge_all_segments()](https://github.com/jianfch/stable-ts/blob/7c6953526dce5d9058b23e8d0c223272bf808be7/stable_whisper/result.py#L630-L633)

### 단어 특정(Locating Words)
정규식을 사용하면 특정 단어의 위치를 찾을 수 있습니다.
```python
# "and"를 포함하는 모든 문장을 검색
matches = result.find(r'[^.]+and[^.]+\.')
# 검색결과가 존재한다면 전부 출력
for match in matches:
  print(f'match: {match.text_match}\n'
        f'text: {match.text}\n'
        f'start: {match.start}\n'
        f'end: {match.end}\n')
  
# 검색결과 중에서 "and" 앞과 뒤에 위치하는 단어를 찾음
matches = matches.find(r'\s\S+\sand\s\S+')
for match in matches:
  print(f'match: {match.text_match}\n'
        f'text: {match.text}\n'
        f'start: {match.start}\n'
        f'end: {match.end}\n')
```
파라미터: 
[find()](https://github.com/jianfch/stable-ts/blob/d30d0d1cfb5b17b4bf59c3fafcbbd21e37598ab9/stable_whisper/result.py#L898-L913)

### 퍼포먼스 개선(Boosting Performance)
* Stable-ts가 타임스탬프 정확도를 개선하고 환각을 줄이기 위해 사용하는 방법 중 하나는 침묵억제(silence suppression)입니다.
이 옵션은 `suppress_silence=True`로 사용할 수 있으며 기본적으로 켜져 있습니다. 
추론과정에서 오디오가 침묵상태이거나 발화(speech)가 없을 경우 상응하는 토큰을 억제하여 
타임스탬프의 생성을 막으며, 추론 이후 타임스탬프를 재조정하는 방법입니다. 
침묵 혹은 발화가 없는 상태를 판별해 내기 위해서 Stable-ts는 non-VAD 방식과 VAD 방식을 지원합니다.
기본설정은 `vad=False`입니다. VAD 옵션은 [Silero VAD](https://github.com/snakers4/silero-vad)를 사용합니다(PyTorch 1.12.0+ 필요). 
[Visualizing Suppression](#visualizing-suppression)를 참고하세요.
* `demucs=True` 옵션으로 사용할 수 있는 다른 방식은 발화를 오디오 트랙에서 격리하기 위해 [Demucs](https://github.com/facebookresearch/demucs)를 사용합니다.
일반적으로 침묵억제 옵션과 같이 사용할 때 최선의 효과를 냅니다.
비록 Demucs는 음악을 위한 기능이지만, 음악이 포함되어 있지 않다 해도 발화 부분을 격리할 때 효과적입니다.

### Visualizing Suppression
오디오의 어떤 부분이 억제될지(예를 들어 '침묵' 상태로 인식될지) 시각화해서 볼 수 있습니다.
요구사항: [Pillow](https://github.com/python-pillow/Pillow) 혹은 [opencv-python](https://github.com/opencv/opencv-python).

#### Without VAD
```python
import stable_whisper
# 파형 가운데 붉게 표시된 부분이 침묵 상태로 인식되어 억제(suppressed)될 가능성이 높은 곳임
# [q_levels]=20 and [k_size]=5 (default)
stable_whisper.visualize_suppression('audio.mp3', 'image.png', q_levels=20, k_size = 5) 
```
![novad](https://user-images.githubusercontent.com/28970749/225825408-aca63dbf-9571-40be-b399-1259d98f93be.png)

#### With [Silero VAD](https://github.com/snakers4/silero-vad)
```python
# [vad_threshold]=0.35 (default)
stable_whisper.visualize_suppression('audio.mp3', 'image.png', vad=True, vad_threshold=0.35)
```
![vad](https://user-images.githubusercontent.com/28970749/225825446-980924a5-7485-41e1-b0d9-c9b069d605f2.png)
파라미터: 
[visualize_suppression()](https://github.com/jianfch/stable-ts/blob/d30d0d1cfb5b17b4bf59c3fafcbbd21e37598ab9/stable_whisper/stabilization.py#L334-L355)

### Encode Comparison 
동일한 오디오의 트랜스크립션을 비교하기 위해 문서에 있는 것과 유사한 비디오를 인코딩할 수 있습니다.
```python
stable_whisper.encode_video_comparison(
    'audio.mp3', 
    ['audio_sub1.srt', 'audio_sub2.srt'], 
    output_videopath='audio.mp4', 
    labels=['Example 1', 'Example 2']
)
```
파라미터: 
[encode_video_comparison()](https://github.com/jianfch/stable-ts/blob/d30d0d1cfb5b17b4bf59c3fafcbbd21e37598ab9/stable_whisper/video_output.py#L10-L27)

### Tips
- 분절단위 타임스탬프를 제대로 얻으려면 `word_timestamps=False`를 사용해서 단어 타임스탬프를 끄는 것은 좋지 않습니다. 단어 타임스탬프는 분절 타임스탬프를 수정하는 데도 쓰이기 때문입니다.
- `demucs=True`와 `vad=True`는 음악에 쓰이지만, 음악이 아닌 오디오(non-music)에도 쓸 수 있습니다.
- 위스퍼(whisper)를 사용했을 때와 비교해서, 긴 오디오 트랙에서 자막이 제대로 추출되지 않는다면 더 많은 메모리를 사용하긴 하지만 `mel_first=True` 옵션을 시도해볼 수 있습니다.
- dynamic quantization 기능을 켜면 CPU에서 추론할 때 메모리 요구량을 감소시킵니다. (또한 대형모델에서 추론 속도를 높여 줍니다);
`--dq true`/`dq=True` for `stable_whisper.load_model`

#### Multiple Files with CLI 
복수의 오디오 파일에서 자막을 추출하고, 곧바로 SRT 파일 형태로 출력합니다.
```commandline
stable-ts audio1.mp3 audio2.mp3 audio3.mp3 -o audio1.srt audio2.srt audio3.srt
```

## Quick 1.X → 2.X Guide
### What's new in 2.0.0?
- Whisper의 더 신뢰도 높은 단어단위 타임스탬프 방식을 사용하도록 업데이트. 
- 이를 통해서 더 자연스러운 방식으로 단어들을 재그룹화할 수 있게 됨.
- [Silero VAD](https://github.com/snakers4/silero-vad)를 활용해 침묵억제가 가능해짐. (PyTorch 1.12.0+ 필요)
- non-VAD 방식의 침묵억제(silence suppression)도 더욱 개선됨.
### Usage changes
- `results_to_sentence_srt(result, 'audio.srt')` → `result.to_srt_vtt('audio.srt', word_level=False)` 
- `results_to_word_srt(result, 'audio.srt')` → `result.to_srt_vtt('output.srt', segment_level=False)`
- `results_to_sentence_word_ass(result, 'audio.srt')` → `result.to_ass('output.ass')`
- there's no need to stabilize segments after inference because they're already stabilized during inference
- `transcribe()` returns a `WhisperResult` object which can be converted to `dict` with `.to_dict()`. e.g `result.to_dict()`

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments
Includes slight modification of the original work: [Whisper](https://github.com/openai/whisper)
