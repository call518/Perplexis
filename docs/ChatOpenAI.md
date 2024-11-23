# Arguments - ChatOpenAI

```python
class ChatOpenAI(
    *,
    name: str | None = None,
    cache: BaseCache | bool | None = None,
    verbose: bool = _get_verbosity,
    callbacks: Callbacks = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    custom_get_token_ids: ((str) -> list[int]) | None = None,
    callback_manager: BaseCallbackManager | None = deprecated(name="callback_manager", since="0.1.7", removal="1.0", alternative="callbacks")(Field(default=None, exclude=True, description="Callback manager to add to the r")),
    rate_limiter: BaseRateLimiter | None = None,
    disable_streaming: bool | Literal['tool_calling'] = False,
    client: Any = None,
    async_client: Any = None,
    root_client: Any = None,
    root_async_client: Any = None,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
    model_kwargs: Dict[str, Any] = dict,
    api_key: SecretStr | None = secret_from_env("OPENAI_API_KEY", default=None),
    base_url: str | None = None,
    organization: str | None = None,
    openai_proxy: str | None = from_env("OPENAI_PROXY", default=None),
    timeout: float | Tuple[float, float] | Any | None = None,
    max_retries: int = 2,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    seed: int | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    logit_bias: Dict[int, int] | None = None,
    streaming: bool = False,
    n: int = 1,
    top_p: float | None = None,
    max_tokens: int | None = None,
    tiktoken_model_name: str | None = None,
    default_headers: Mapping[str, str] | None = None,
    default_query: Mapping[str, object] | None = None,
    http_client: Any | None = None,
    http_async_client: Any | None = None,
    stop_sequences: List[str] | str | None = None,
    extra_body: Mapping[str, Any] | None = None,
    include_response_headers: bool = False,
    disabled_params: Dict[str, Any] | None = None,
    stream_usage: bool = False
)
```

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

### `custom_get_token_ids`: `(str) -> list[int]` | `None` (default: `None`)
- **의미**: 텍스트를 토큰 ID의 리스트로 변환하는 사용자 정의 함수입니다.
- **역할**: 토크나이저를 커스터마이징할 때 사용됩니다.

### `callback_manager`: `BaseCallbackManager` | `None` (deprecated)
- **의미**: 콜백 매니저를 지정합니다. 하지만 이 매개변수는 `callbacks`로 대체되었습니다.
- **역할**: 이전 버전과의 호환성을 위해 존재하며, `callbacks`를 사용하는 것이 권장됩니다.

### `rate_limiter`: `BaseRateLimiter` | `None` (default: `None`)
- **의미**: 요청 속도를 제한하는 레이트 리미터입니다.
- **역할**: API 호출 시 속도를 제어하여 API 제한에 도달하지 않도록 합니다.

### `disable_streaming`: `bool` | `'tool_calling'` (default: `False`)
- **의미**: 스트리밍을 비활성화할지 여부를 지정합니다. `'tool_calling'`으로 설정하면 도구 호출 시에만 스트리밍을 비활성화합니다.
- **역할**: 스트리밍 기능을 제어하여 응답 방식을 결정합니다.

### `client`: `Any` (default: `None`)
- **의미**: OpenAI의 동기 클라이언트 인스턴스입니다.
- **역할**: 별도의 클라이언트를 지정하여 API 호출을 수행합니다.

### `async_client`: `Any` (default: `None`)
- **의미**: OpenAI의 비동기 클라이언트 인스턴스입니다.
- **역할**: 비동기 환경에서 클라이언트를 지정하여 API 호출을 수행합니다.

### `root_client`: `Any` (default: `None`)
- **의미**: 루트 동기 클라이언트 인스턴스입니다.
- **역할**: 하위 클래스나 래퍼에서 기본 클라이언트를 지정할 때 사용됩니다.

### `root_async_client`: `Any` (default: `None`)
- **의미**: 루트 비동기 클라이언트 인스턴스입니다.
- **역할**: 하위 클래스나 래퍼에서 기본 비동기 클라이언트를 지정할 때 사용됩니다.

### `model`: `str` (default: `"gpt-3.5-turbo"`)
- **의미**: 사용할 OpenAI의 채팅 모델 이름입니다.
- **역할**: 특정 모델을 선택하여 응답의 품질과 성능을 결정합니다.

### `temperature`: `float` (default: `0.7`)
- **의미**: 샘플링 온도를 지정합니다.
- **역할**: 출력의 다양성과 창의성을 조절합니다. 낮은 값은 일관된 출력을, 높은 값은 다양하고 창의적인 출력을 생성합니다.

### `model_kwargs`: `Dict[str, Any]` (default: `dict`)
- **의미**: 모델 호출 시 추가로 전달할 매개변수입니다.
- **역할**: 모델의 동작을 세부적으로 조정합니다.

### `api_key`: `SecretStr` | `None` (default: `None`)
- **의미**: OpenAI API 키입니다.
- **역할**: API 인증을 위해 사용되며, 환경 변수에서 안전하게 가져옵니다.

### `base_url`: `str` | `None` (default: `None`)
- **의미**: OpenAI API의 기본 URL입니다.
- **역할**: 커스텀 엔드포인트나 프록시를 사용할 때 기본 URL을 설정합니다.

### `organization`: `str` | `None` (default: `None`)
- **의미**: OpenAI 조직 ID입니다.
- **역할**: 조직별로 API 사용량을 관리할 때 사용됩니다.

### `openai_proxy`: `str` | `None` (default: `None`)
- **의미**: OpenAI API에 연결할 때 사용할 프록시 서버의 주소입니다.
- **역할**: 네트워크 환경에 따라 프록시 설정을 적용합니다.

### `timeout`: `float` | `Tuple[float, float]` | `Any` | `None` (default: `None`)
- **의미**: API 요청의 타임아웃 시간입니다.
- **역할**: 요청이 지정된 시간 내에 완료되지 않으면 실패로 처리합니다.

### `max_retries`: `int` (default: `2`)
- **의미**: API 요청 실패 시 재시도할 최대 횟수입니다.
- **역할**: 일시적인 네트워크 오류나 API 제한에 대응합니다.

### `presence_penalty`: `float` | `None` (default: `None`)
- **의미**: 텍스트 생성 시 새로운 토픽을 도입할 가능성을 조절하는 매개변수입니다.
- **역할**: 높은 값은 새로운 내용의 도입을 촉진합니다.

### `frequency_penalty`: `float` | `None` (default: `None`)
- **의미**: 텍스트 생성 시 반복을 줄이는 매개변수입니다.
- **역할**: 높은 값은 동일한 구나 단어의 반복을 억제합니다.

### `seed`: `int` | `None` (default: `None`)
- **의미**: 랜덤 시드 값을 지정합니다.
- **역할**: 재현 가능한 결과를 얻기 위해 랜덤 시드를 설정합니다.

### `logprobs`: `bool` | `None` (default: `None`)
- **의미**: 각 토큰의 로그 확률을 반환할지 여부를 지정합니다.
- **역할**: 토큰별 확률 정보를 얻을 수 있습니다.

### `top_logprobs`: `int` | `None` (default: `None`)
- **의미**: 각 단계에서 상위 N개의 토큰의 로그 확률을 반환합니다.
- **역할**: 다음에 올 수 있는 토큰의 후보를 파악할 수 있습니다.

### `logit_bias`: `Dict[int, int]` | `None` (default: `None`)
- **의미**: 특정 토큰의 선택 확률에 편향을 적용합니다.
- **역할**: 특정 토큰의 출력 가능성을 높이거나 낮춥니다.

### `streaming`: `bool` (default: `False`)
- **의미**: 스트리밍 모드를 사용할지 여부를 지정합니다.
- **역할**: 응답을 토큰 단위로 스트리밍하여 실시간으로 출력합니다.

### `n`: `int` (default: `1`)
- **의미**: 각 프롬프트에 대해 생성할 응답의 수입니다.
- **역할**: 여러 개의 응답을 한 번에 생성하여 비교할 수 있습니다.

### `top_p`: `float` | `None` (default: `None`)
- **의미**: 누적 확률이 `top_p`가 될 때까지의 토큰들로만 샘플링합니다.
- **역할**: 샘플링의 다양성을 조절합니다.

### `max_tokens`: `int` | `None` (default: `None`)
- **의미**: 생성할 최대 토큰 수를 지정합니다.
- **역할**: 응답의 길이를 제한합니다.

### `tiktoken_model_name`: `str` | `None` (default: `None`)
- **의미**: 토큰화를 위해 사용할 `tiktoken` 모델의 이름입니다.
- **역할**: 토큰 카운트를 정확하게 계산하기 위해 사용됩니다.

### `default_headers`: `Mapping[str, str]` | `None` (default: `None`)
- **의미**: 모든 요청에 적용할 기본 HTTP 헤더입니다.
- **역할**: 공통적으로 필요한 헤더를 설정합니다.

### `default_query`: `Mapping[str, object]` | `None` (default: `None`)
- **의미**: 모든 요청에 적용할 기본 쿼리 매개변수입니다.
- **역할**: 공통적인 쿼리 파라미터를 설정합니다.

### `http_client`: `Any` | `None` (default: `None`)
- **의미**: 동기 HTTP 요청에 사용할 클라이언트입니다.
- **역할**: 커스텀 HTTP 클라이언트를 지정하여 네트워크 설정을 커스터마이징합니다.

### `http_async_client`: `Any` | `None` (default: `None`)
- **의미**: 비동기 HTTP 요청에 사용할 클라이언트입니다.
- **역할**: 비동기 환경에서 커스텀 클라이언트를 사용합니다.

### `stop_sequences`: `List[str]` | `str` | `None` (default: `None`)
- **의미**: 텍스트 생성을 중단할 시퀀스(문자열) 목록입니다.
- **역할**: 특정 문자열이 생성되면 응답을 중단합니다.

### `extra_body`: `Mapping[str, Any]` | `None` (default: `None`)
- **의미**: 요청 본문에 추가로 포함할 데이터입니다.
- **역할**: API 호출 시 추가적인 파라미터를 전달합니다.

### `include_response_headers`: `bool` (default: `False`)
- **의미**: 응답 헤더를 포함할지 여부를 지정합니다.
- **역할**: 응답의 메타데이터를 확인할 때 사용합니다.

### `disabled_params`: `Dict[str, Any]` | `None` (default: `None`)
- **의미**: 비활성화할 매개변수의 딕셔너리입니다.
- **역할**: 특정 매개변수를 사용하지 않도록 설정합니다.

### `stream_usage`: `bool` (default: `False`)
- **의미**: 스트리밍 모드에서 토큰 사용량 정보를 수집할지 여부입니다.
- **역할**: 스트리밍 중에도 토큰 사용량을 추적합니다.
