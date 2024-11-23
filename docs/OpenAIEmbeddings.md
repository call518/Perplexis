# Arguments - OpenAIEmbeddings

```python
class OpenAIEmbeddings(
    *,
    client: Any = None,
    async_client: Any = None,
    model: str = "text-embedding-ada-002",
    dimensions: int | None = None,
    deployment: str | None = model,
    api_version: str | None = from_env("OPENAI_API_VERSION", default=None),
    base_url: str | None = from_env("OPENAI_API_BASE", default=None),
    openai_api_type: str | None = from_env("OPENAI_API_TYPE", default=None),
    openai_proxy: str | None = from_env("OPENAI_PROXY", default=None),
    embedding_ctx_length: int = 8191,
    api_key: SecretStr | None = secret_from_env("OPENAI_API_KEY", default=None),
    organization: str | None = from_env(["OPENAI_ORG_ID", "OPENAI_ORGANIZATION"], default=None),
    allowed_special: Set[str] | Literal['all'] | None = None,
    disallowed_special: Set[str] | Sequence[str] | Literal['all'] | None = None,
    chunk_size: int = 1000,
    max_retries: int = 2,
    timeout: float | Tuple[float, float] | Any | None = None,
    headers: Any = None,
    tiktoken_enabled: bool = True,
    tiktoken_model_name: str | None = None,
    show_progress_bar: bool = False,
    model_kwargs: Dict[str, Any] = dict,
    skip_empty: bool = False,
    default_headers: Mapping[str, str] | None = None,
    default_query: Mapping[str, object] | None = None,
    retry_min_seconds: int = 4,
    retry_max_seconds: int = 20,
    http_client: Any | None = None,
    http_async_client: Any | None = None,
    check_embedding_ctx_length: bool = True
)
```

## Parameters

### `client`: `Any` (default: `None`)
- **의미**: 이미 생성된 OpenAI 클라이언트 인스턴스를 전달할 때 사용합니다.
- **역할**: 별도의 클라이언트를 생성하지 않고 기존 클라이언트를 재사용합니다.

### `async_client`: `Any` (default: `None`)
- **의미**: 비동기 작업을 위한 OpenAI 클라이언트 인스턴스를 전달합니다.
- **역할**: 비동기 환경에서 비동기 클라이언트를 사용하여 성능을 향상시킵니다.

### `model`: `str` (default: `"text-embedding-ada-002"`)
- **의미**: 사용할 OpenAI 임베딩 모델의 이름입니다.
- **역할**: 임베딩을 생성할 때 특정 모델을 선택합니다.

### `dimensions`: `int` | `None` (default: `None`)
- **의미**: 생성되는 임베딩 벡터의 차원 수를 지정합니다.
- **역할**: 모델에 따라 임베딩의 차원을 커스터마이징합니다. text-embedding-3 이후 모델에서 지원됩니다.

### `deployment`: `str` | `None` (default: `model`)
- **의미**: Azure OpenAI 서비스에서 사용되는 배포 이름입니다.
- **역할**: Azure 환경에서 특정 배포를 지정하여 모델을 호출합니다.

### `api_version`: `str` | `None` (default: `from_env("OPENAI_API_VERSION", default=None)`)
- **의미**: 사용할 OpenAI API의 버전입니다.
- **역할**: API의 버전을 명시적으로 지정하거나 환경 변수에서 가져옵니다.

### `base_url`: `str` | `None` (default: `from_env("OPENAI_API_BASE", default=None)`)
- **의미**: OpenAI API의 기본 URL입니다.
- **역할**: 커스텀 엔드포인트나 Azure API를 사용할 때 기본 URL을 설정합니다.

### `openai_api_type`: `str` | `None` (default: `from_env("OPENAI_API_TYPE", default=None)`)
- **의미**: 사용할 OpenAI API의 유형입니다 (예: 'azure').
- **역할**: API 호출 시 특정 API 타입을 지정합니다.

### `openai_proxy`: `str` | `None` (default: `from_env("OPENAI_PROXY", default=None)`)
- **의미**: OpenAI API에 연결할 때 사용할 프록시 서버의 주소입니다.
- **역할**: 네트워크 환경에 따라 프록시 설정을 적용합니다.

### `embedding_ctx_length`: `int` (default: `8191`)
- **의미**: 임베딩 모델의 최대 컨텍스트 길이(토큰 수)입니다.
- **역할**: 입력 텍스트의 길이를 제한하여 모델의 제약을 준수합니다.

### `api_key`: `SecretStr` | `None` (default: `secret_from_env("OPENAI_API_KEY", default=None)`)
- **의미**: OpenAI API 키입니다.
- **역할**: API 인증을 위해 사용되며, 환경 변수에서 안전하게 가져옵니다.

### `organization`: `str` | `None` (default: `from_env(["OPENAI_ORG_ID", "OPENAI_ORGANIZATION"], default=None)`)
- **의미**: OpenAI 조직 ID입니다.
- **역할**: 조직별로 API 사용량을 관리할 때 사용됩니다.

### `allowed_special`: `Set[str]` | `Literal['all']` | `None` (default: `None`)
- **의미**: 입력 텍스트에서 허용되는 특수 토큰의 집합입니다.
- **역할**: 토크나이징 과정에서 특정 특수 문자를 허용합니다.

### `disallowed_special`: `Set[str]` | `Sequence[str]` | `Literal['all']` | `None` (default: `None`)
- **의미**: 입력 텍스트에서 허용되지 않는 특수 토큰의 집합입니다.
- **역할**: 토크나이징 과정에서 특정 특수 문자를 제거하거나 에러를 발생시킵니다.

### `chunk_size`: `int` (default: `1000`)
- **의미**: 긴 텍스트를 처리할 때 나누는 청크의 크기입니다.
- **역할**: 대량의 데이터를 효율적으로 처리하기 위해 텍스트를 분할합니다.

### `max_retries`: `int` (default: `2`)
- **의미**: API 요청 실패 시 재시도할 최대 횟수입니다.
- **역할**: 일시적인 네트워크 오류나 API 제한에 대응합니다.

### `timeout`: `float` | `Tuple[float, float]` | `Any` | `None` (default: `None`)
- **의미**: API 요청의 타임아웃 시간입니다.
- **역할**: 요청이 지정된 시간 내에 완료되지 않으면 실패로 처리합니다.

### `headers`: `Any` (default: `None`)
- **의미**: 추가적인 HTTP 헤더를 지정합니다.
- **역할**: 요청 시 사용자 정의 헤더를 포함합니다.

### `tiktoken_enabled`: `bool` (default: `True`)
- **의미**: 토큰화에 tiktoken 라이브러리를 사용할지 여부입니다.
- **역할**: tiktoken을 사용하여 토큰화를 최적화합니다.

### `tiktoken_model_name`: `str` | `None` (default: `None`)
- **의미**: tiktoken에서 사용할 모델 이름입니다.
- **역할**: 특정 모델에 맞는 토큰화를 적용합니다.

### `show_progress_bar`: `bool` (default: `False`)
- **의미**: 진행률 표시줄을 표시할지 여부입니다.
- **역할**: 대량의 데이터를 처리할 때 진행 상황을 모니터링합니다.

### `model_kwargs`: `Dict[str, Any]` (default: `dict`)
- **의미**: 모델 호출 시 추가로 전달할 매개변수입니다.
- **역할**: 모델의 동작을 세부적으로 조정합니다.

### `skip_empty`: `bool` (default: `False`)
- **의미**: 빈 문자열을 임베딩할 때 건너뛸지 여부입니다.
- **역할**: 데이터 전처리 단계에서 빈 입력을 무시합니다.

### `default_headers`: `Mapping[str, str]` | `None` (default: `None`)
- **의미**: 모든 요청에 적용할 기본 HTTP 헤더입니다.
- **역할**: 공통적으로 필요한 헤더를 설정합니다.

### `default_query`: `Mapping[str, object]` | `None` (default: `None`)
- **의미**: 모든 요청에 적용할 기본 쿼리 매개변수입니다.
- **역할**: 공통적인 쿼리 파라미터를 설정합니다.

### `retry_min_seconds`: `int` (default: `4`)
- **의미**: 재시도 간 최소 대기 시간입니다.
- **역할**: 재시도 시 백오프 전략을 구현합니다.

### `retry_max_seconds`: `int` (default: `20`)
- **의미**: 재시도 간 최대 대기 시간입니다.
- **역할**: 재시도 시 대기 시간의 상한을 설정합니다.

### `http_client`: `Any` | `None` (default: `None`)
- **의미**: 동기 HTTP 요청에 사용할 클라이언트입니다.
- **역할**: 커스텀 HTTP 클라이언트를 지정하여 네트워크 설정을 커스터마이징합니다.

### `http_async_client`: `Any` | `None` (default: `None`)
- **의미**: 비동기 HTTP 요청에 사용할 클라이언트입니다.
- **역할**: 비동기 환경에서 커스텀 클라이언트를 사용합니다.

### `check_embedding_ctx_length`: `bool` (default: `True`)
- **의미**: 입력 텍스트의 길이가 모델의 최대 컨텍스트 길이를 초과하는지 확인할지 여부입니다.
- **역할**: 입력 길이로 인한 오류를 사전에 방지합니다.
