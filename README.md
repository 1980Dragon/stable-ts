# Stabilizing Timestamps for Whisper

신뢰도 높은 타임스탬프 산출을 위해 [OpenAI's Whisper](https://github.com/openai/whisper)의 코드를 수정한 스크립트.

https://user-images.githubusercontent.com/28970749/225826345-ef7115db-51e4-4b23-aedd-069389b8ae43.mp4

* [Setup](#setup)
* [Usage](#usage)
  * [Transcribe](#transcribe)
  * [Output](#output)
  * [Regrouping Words](#regrouping-words)
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

## Usage
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

`segment_level=True`  + `word_level=False` (노트: `segment_level=True`가 기본 설정임)
```
00:00:07.760 --> 00:00:09.900
But when you arrived at that distant world,
```

`segment_level=False` + `word_level=True` (노트: `word_level=True`가 기본 설정임)
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
# The following all functionally equivalent:
result0 = model.transcribe('audio.mp3', regroup=True) # regroup is True by default
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
Any regrouping algorithm can be expressed as a string. Please feel free share your strings [here](https://github.com/jianfch/stable-ts/discussions/162)
#### Regrouping Methods
- [regroup](https://github.com/jianfch/stable-ts/blob/7c6953526dce5d9058b23e8d0c223272bf808be7/stable_whisper/result.py#L808-L854)
- [split_by_gap()](https://github.com/jianfch/stable-ts/blob/7c6953526dce5d9058b23e8d0c223272bf808be7/stable_whisper/result.py#L526-L543)
- [split_by_punctuation()](https://github.com/jianfch/stable-ts/blob/7c6953526dce5d9058b23e8d0c223272bf808be7/stable_whisper/result.py#L579-L595)
- [split_by_length()](https://github.com/jianfch/stable-ts/blob/7c6953526dce5d9058b23e8d0c223272bf808be7/stable_whisper/result.py#L637-L658)
- [merge_by_gap()](https://github.com/jianfch/stable-ts/blob/7c6953526dce5d9058b23e8d0c223272bf808be7/stable_whisper/result.py#L547-L573)
- [merge_by_punctuation()](https://github.com/jianfch/stable-ts/blob/7c6953526dce5d9058b23e8d0c223272bf808be7/stable_whisper/result.py#L599-L624)
- [merge_all_segments()](https://github.com/jianfch/stable-ts/blob/7c6953526dce5d9058b23e8d0c223272bf808be7/stable_whisper/result.py#L630-L633)

### Locating Words
You can locate words with regular expression.
```python
# Find every sentence that contains "and"
matches = result.find(r'[^.]+and[^.]+\.')
# print the all matches if there are any
for match in matches:
  print(f'match: {match.text_match}\n'
        f'text: {match.text}\n'
        f'start: {match.start}\n'
        f'end: {match.end}\n')
  
# Find the word before and after "and" in the matches
matches = matches.find(r'\s\S+\sand\s\S+')
for match in matches:
  print(f'match: {match.text_match}\n'
        f'text: {match.text}\n'
        f'start: {match.start}\n'
        f'end: {match.end}\n')
```
Parameters: 
[find()](https://github.com/jianfch/stable-ts/blob/d30d0d1cfb5b17b4bf59c3fafcbbd21e37598ab9/stable_whisper/result.py#L898-L913)

### Boosting Performance
* One of the methods that Stable-ts uses to increase timestamp accuracy 
and reduce hallucinations is silence suppression, enabled with `suppress_silence=True` (default).
This method essentially suppresses the timestamps where the audio is silent or contain no speech
by suppressing the corresponding tokens during inference and also readjusting the timestamps after inference. 
To figure out which parts of the audio track are silent or contain no speech, Stable-ts supports non-VAD and VAD methods.
The default is `vad=False`. The VAD option uses [Silero VAD](https://github.com/snakers4/silero-vad) (requires PyTorch 1.12.0+). 
See [Visualizing Suppression](#visualizing-suppression).
* The other method, enabled with `demucs=True`, uses [Demucs](https://github.com/facebookresearch/demucs)
to isolate speech from the rest of the audio track. Generally best used in conjunction with silence suppression.
Although Demucs is for music, it is also effective at isolating speech even if the track contains no music.

### Visualizing Suppression
You can visualize which parts of the audio will likely be suppressed (i.e. marked as silent). 
Requires: [Pillow](https://github.com/python-pillow/Pillow) or [opencv-python](https://github.com/opencv/opencv-python).

#### Without VAD
```python
import stable_whisper
# regions on the waveform colored red are where it will likely be suppressed and marked as silent
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
Parameters: 
[visualize_suppression()](https://github.com/jianfch/stable-ts/blob/d30d0d1cfb5b17b4bf59c3fafcbbd21e37598ab9/stable_whisper/stabilization.py#L334-L355)

### Encode Comparison 
You can encode videos similar to the ones in the doc for comparing transcriptions of the same audio. 
```python
stable_whisper.encode_video_comparison(
    'audio.mp3', 
    ['audio_sub1.srt', 'audio_sub2.srt'], 
    output_videopath='audio.mp4', 
    labels=['Example 1', 'Example 2']
)
```
Parameters: 
[encode_video_comparison()](https://github.com/jianfch/stable-ts/blob/d30d0d1cfb5b17b4bf59c3fafcbbd21e37598ab9/stable_whisper/video_output.py#L10-L27)

### Tips
- for reliable segment timestamps, do not disable word timestamps with `word_timestamps=False` because word timestamps are also used to correct segment timestamps
- use `demucs=True` and `vad=True` for music but also works for non-music
- if audio is not transcribing properly compared to whisper, try `mel_first=True` at the cost of more memory usage for long audio tracks
- enable dynamic quantization to decrease memory usage for inference on CPU (also increases inference speed for large model);
`--dq true`/`dq=True` for `stable_whisper.load_model`

#### Multiple Files with CLI 
Transcribe multiple audio files then process the results directly into SRT files.
```commandline
stable-ts audio1.mp3 audio2.mp3 audio3.mp3 -o audio1.srt audio2.srt audio3.srt
```

## Quick 1.X → 2.X Guide
### What's new in 2.0.0?
- updated to use Whisper's more reliable word-level timestamps method. 
- the more reliable word timestamps allow regrouping all words into segments with more natural boundaries.
- can now suppress silence with [Silero VAD](https://github.com/snakers4/silero-vad) (requires PyTorch 1.12.0+)
- non-VAD silence suppression is also more robust
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
