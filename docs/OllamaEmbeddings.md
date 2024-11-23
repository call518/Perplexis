# Arguments - OllamaEmbeddings

```python
class OllamaEmbeddings(
    *,
    model: str,
    base_url: str | None = None,
    client_kwargs: dict | None = {}
)
```

## Parameters

### `model`: `str`
- **의미**: 사용할 Ollama 임베딩 모델의 이름입니다.
- **역할**: 임베딩을 생성할 때 특정 모델을 선택합니다. Ollama에서 지원하는 다양한 모델 중 하나를 지정해야 합니다.

### `base_url`: `str` | `None` (default: `None`)
- **의미**: Ollama 모델이 호스팅되는 기본 URL입니다.
- **역할**: 로컬이 아닌 원격 서버나 커스텀 호스트에서 Ollama 모델을 사용하려는 경우 기본 URL을 지정합니다. 기본값은 None이며, 이 경우 로컬 호스트를 사용합니다.

### `client_kwargs`: `dict` | `None` (default: `{}`)
- **의미**: Ollama 클라이언트에 전달할 추가 매개변수의 딕셔너리입니다.
- **역할**: 클라이언트의 동작을 세부적으로 조정하거나 추가 설정을 제공할 때 사용합니다.
