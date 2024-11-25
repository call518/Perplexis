# Arguments - OllamaLLM

```python
class OllamaLLM(
    *,
    name: str | None = None,
    cache: BaseCache | bool | None = None,
    verbose: bool = _get_verbosity,
    callbacks: Callbacks = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    custom_get_token_ids: ((str) -> list[int]) | None = None,
    callback_manager: BaseCallbackManager | None = None,
    model: str,
    mirostat: int | None = None,
    mirostat_eta: float | None = None,
    mirostat_tau: float | None = None,
    num_ctx: int | None = None,
    num_gpu: int | None = None,
    num_thread: int | None = None,
    num_predict: int | None = None,
    repeat_last_n: int | None = None,
    repeat_penalty: float | None = None,
    temperature: float | None = None,
    stop: List[str] | None = None,
    tfs_z: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    format: Literal['', 'json'] = "",
    keep_alive: int | str | None = None,
    base_url: str | None = None,
    client_kwargs: dict | None = {}
)
```

## Parameters

### `name`: `str` | `None` (default: `None`)
- **의미**: 이 인스턴스의 이름을 지정합니다.
- **역할**: 여러 모델 인스턴스를 사용할 때 구분하기 위해 사용됩니다.

### `cache`: `BaseCache` | `bool` | `None` (default: `None`)
- **의미**: 응답을 캐싱할지 여부 또는 캐시 인스턴스를 지정합니다.
- **역할**: 동일한 입력에 대한 반복적인 요청 시 성능을 향상시킵니다.

### `verbose`: `bool` (default: `_get_verbosity`)
- **의미**: 상세한 로그를 출력할지 여부를 지정합니다.
- **역할**: 디버깅이나 개발 시 유용한 정보를 출력합니다.

### `callbacks`: `Callbacks` (default: `None`)
- **의미**: 호출할 콜백 함수들의 집합입니다.
- **역할**: 모델의 실행 과정에서 특정 이벤트에 대해 콜백을 실행합니다.

### `tags`: `list[str]` | `None` (default: `None`)
- **의미**: 이 인스턴스와 관련된 태그의 리스트입니다.
- **역할**: 로그나 추적을 위해 인스턴스를 식별하는 데 사용됩니다.

### `metadata`: `dict[str, Any]` | `None` (default: `None`)
- **의미**: 인스턴스에 대한 추가 메타데이터입니다.
- **역할**: 사용자 정의 정보를 저장하고 추적하는 데 사용됩니다.

### `custom_get_token_ids`: `((str) -> list[int])` | `None` (default: `None`)
- **의미**: 텍스트를 토큰 ID의 리스트로 변환하는 사용자 정의 함수입니다.
- **역할**: 토크나이저를 커스터마이징할 때 사용됩니다.

### `callback_manager`: `BaseCallbackManager` | `None` (default: `None`)
- **의미**: 콜백 매니저를 지정합니다.
- **역할**: 콜백의 등록과 관리를 담당합니다.

### `model`: `str`
- **의미**: 사용할 Ollama 모델의 이름입니다.
- **역할**: 특정 모델을 선택하여 응답의 품질과 성능을 결정합니다.
- **주의**: 필수 매개변수로, 반드시 지정해야 합니다.

### `mirostat`: `int` | `None` (default: `None`)
- **의미**: Mirostat 알고리즘의 버전을 지정합니다.
- **역할**: 텍스트 생성 시 동적 엔트로피 조절을 위해 사용됩니다.
- **값의 범위**: 1 또는 2를 지정할 수 있습니다. None이면 Mirostat을 사용하지 않습니다.

### `mirostat_eta`: `float` | `None` (default: `None`)
- **의미**: Mirostat 알고리즘의 학습 속도 파라미터입니다.
- **역할**: Mirostat 알고리즘의 반응성을 조절합니다.

### `mirostat_tau`: `float` | `None` (default: `None`)
- **의미**: Mirostat 알고리즘의 목표 엔트로피 값입니다.
- **역할**: 생성되는 텍스트의 엔트로피 수준을 조절합니다.

### `num_ctx`: `int` | `None` (default: `None`)
- **의미**: 모델의 컨텍스트 크기(최대 토큰 수)입니다.
- **역할**: 입력과 출력의 최대 길이를 설정합니다.

### `num_gpu`: `int` | `None` (default: `None`)
- **의미**: 사용할 GPU의 개수입니다.
- **역할**: 모델 추론 시 병렬 처리를 위해 GPU를 활용합니다.

### `num_thread`: `int` | `None` (default: `None`)
- **의미**: CPU에서 사용할 스레드의 수입니다.
- **역할**: CPU 병렬 처리를 통해 성능을 향상시킵니다.

### `num_predict`: `int` | `None` (default: `None`)
- **의미**: 생성할 예측 토큰의 수입니다.
- **역할**: 출력 텍스트의 길이를 조절합니다.

### `repeat_last_n`: `int` | `None` (default: `None`)
- **의미**: 반복 페널티를 적용할 마지막 N개의 토큰 수입니다.
- **역할**: 최근 생성된 토큰의 반복을 억제합니다.

### `repeat_penalty`: `float` | `None` (default: `None`)
- **의미**: 반복 페널티의 강도를 지정합니다.
- **역할**: 높은 값일수록 반복되는 패턴을 더 강하게 억제합니다.

### `temperature`: `float` | `None` (default: `None`)
- **의미**: 샘플링 온도를 지정합니다.
- **역할**: 출력의 다양성과 창의성을 조절합니다. 낮은 값은 일관된 출력을, 높은 값은 다양하고 창의적인 출력을 생성합니다.

### `stop`: `List[str]` | `None` (default: `None`)
- **의미**: 텍스트 생성을 중단할 시퀀스(문자열) 목록입니다.
- **역할**: 특정 문자열이 생성되면 응답을 중단합니다.

### `tfs_z`: `float` | `None` (default: `None`)
- **의미**: Tail Free Sampling의 Z 값입니다.
- **역할**: 샘플링 시 낮은 확률의 토큰을 제거하여 생성 품질을 향상시킵니다.

### `top_k`: `int` | `None` (default: `None`)
- **의미**: 다음 토큰을 선택할 때 고려할 상위 K개의 토큰 수입니다.
- **역할**: 샘플링의 범위를 제한하여 출력의 일관성을 높입니다.

### `top_p`: `float` | `None` (default: `None`)
- **의미**: 누적 확률이 top_p가 될 때까지의 토큰들로만 샘플링합니다.
- **역할**: 샘플링의 다양성을 조절합니다.

### `format`: `Literal['', 'json']` (default: `""`)
- **의미**: 출력 형식을 지정합니다.
- **역할**: "" (빈 문자열): 일반 텍스트 형식으로 출력합니다. "json": JSON 형식으로 출력합니다.

### `keep_alive`: `int` | `str` | `None` (default: `None`)
- **의미**: 세션을 유지할 시간 또는 세션 ID를 지정합니다.
- **역할**: 여러 요청 간에 세션을 유지하여 컨텍스트를 공유합니다.

### `base_url`: `str` | `None` (default: `None`)
- **의미**: Ollama API의 기본 URL입니다.
- **역할**: 커스텀 엔드포인트나 프록시를 사용할 때 기본 URL을 설정합니다.

### `client_kwargs`: `dict` | `None` (default: `{}`)
- **의미**: 클라이언트에 전달할 추가 매개변수의 딕셔너리입니다.
- **역할**: HTTP 요청 시 추가적인 설정을 적용합니다.
