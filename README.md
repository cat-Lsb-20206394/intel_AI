-------------------------------------------------------------------------------------------------------------------------------------
# proj1
openvino를 이용한 Camera Multi detecting

## 주요기능
여러 대의 카메라를 이용해서 움직이는 사람들을 찾아주고, 전체 사람 수를 실시간으로 그래프를 이용해서 보여줌.

### import
 from utils.network_wrappers import Detector, VectorCNN, MaskRCNN, DetectionsFromFileReader
 from mc_tracker.mct import MultiCameraTracker
 from utils.analyzer import save_embeddings
 from utils.misc import read_py_config, check_pressed_keys
 from utils.video import MulticamCapture, NormalizerCLAHE
 from utils.visualization import visualize_multicam_detections, get_target_size, plot_timeline
 from openvino import Core, get_version
 
1. 해당 import를 실행하는 과정에서 utils,mc_tracker 가 필요하기에 기존모델에있던 파일에서 복사해서 작업공간에 붙여넣어줬습니다.

2. –config를 할 때 configs파일 안에 python.py를 사용하기 때문에 configs파일 또한 기존모델에 있던 파일에서 복사해서 작업공간에 붙여넣어줬습니다.

3. 사용한xml파일은
–m_detector 에서는  /intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml
–m_reid      에서는  /intel/person-reidentification-retail-0277/FP16/person-reidentification-retail-0277.xml
각각의 xml파일을사용하였습니다.

4. 영상파일은 같은 작업공간 안에 video_1.mp4,video_2.mp4라는 이름으로 저장하면 됩니다.

5. 추가적으로 파이썬에서의 __file__ 주피터에서는 기능하지 않아 os.getcwd()로 수정하였습니다.

-------------------------------------------------------------------------------------------------------------------------------------
# proj2
Llama3.0과 finetuning을 이용한 간단한 챗봇

## 주요기능
자연어 처리 모델에 서울시 공영주차장 정보를 finetuning해서 주차장에 대한 정보를 알려주는 챗봇을 구현함.

### import
주요 import 항목들은 전부 코드 내에 명시되어 있음.

### json 파일
ipynp 혹은 py 파일과 동일 폴더에 위치
park.json 파일만 다운로드후 사용 park_data.json은 코드에서 데이터정제 후에 자동으로 생성.  

### huggingface
Llama3.0모델을 허깅페이스에서 가져와 사용하기 때문에 huggingface token이 필요함,
추가적으로 특정 Llama 모델들은 모델 사용허가를 받아야 사용할 수 있으므로 미리 granted 받아야 함.

### 실행
따로 프론트를 구현하지 않았기 때문에 실행 후 코드를 추가하여 질문하면 됨
gen("질문 (ex.도봉구 창동에 있는 공영주차장에 대해서 알려줘)")

데이터 출처 : https://data.seoul.go.kr/dataList/OA-13122/S/1/datasetView.do

-------------------------------------------------------------------------------------------------------------------------------------
# proj3
OpenAi의 Whisper 모델과 youtube API 그리고 Bert모델을 이용한 실시간 주요 정보 알아보기 서비스.

## 주요기능
1. youtube api를 이용해 최신 영상의 mp3파일 불러오기(채널ID, 영상ID 불러오기)
2. whisper 이용하여 음성을 텍스트로 변환
3. "keybert 모델"을 사용하여 텍스트에서 키워드 추출
  3-1. 첫번째는 직접 모델을 불러와서 파이썬에서 바로 적용 -> 키워드 바로 제공
  3-2. 두번째는 onnx파일로 모델을 가져와서 "openvino"로 최적화 -> 임베딩 벡터 형태로 제공
     -> 텍스트를 토큰화하여 추출된 임베딩 벡터로 유사도 계산하여 키워드 추출
4. openai 에서 gpt 4를 이용해서 키워드 기반으로 문장 생성
5. Hugging face 에서 llama3.2를 이용해서 키워드 기반으로 문장 생성
   -> onnx파일로 모델을 가져와서 "openvino"로 최적화

### import
주요 import 항목들은 전부 코드 내에 명시되어 있음.

### API_KEY
해당 프로젝트에서는 youtube API_KEY, huggingface token, OpenAi의 API_KEY가 필요하니 미리 발급받아야 함.
OpenAi는 유료서비스이므로 token을 미리 1-5달러 정도 결제해놓아야 함. (실제 사용량은 1달러 이내로 해결이 됨.) - 혹시나 모를 추가 결제가 될 수도 있으니 limit를 걸어놓고 사용하는 것을 추천.
추가적으로 huggingface token의 경우 특정 Llama 모델들은 모델 사용허가를 받아야 사용할 수 있으므로 미리 granted 받아야 함.

### openvino
해당 프로젝트는 openvino를 이용해서 CPU에 최적화시키는 것을 해보는 프로젝트로 인텔코어 10세대 이상의 사양이 요구됨.

### 실행
음성변환의 정확도를 위해 whisper의 large 모델을 사용했기 때문에 다소 시간이 소요될 수 있음.
(large, middle, small, base 모델 순서로 정확도는 높으나 순서가 느림. 빨리 진행을 원하면 small 모델 사용 추천)


-------------------------------------------------------------------------------------------------------------------------------------
