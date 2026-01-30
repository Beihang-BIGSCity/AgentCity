# CoMaPOI Migration to LibCity

## Overview

This document describes the adaptation of the CoMaPOI multi-agent POI prediction system to LibCity's trajectory location prediction framework.

## Original Model

**Repository**: `/home/wangwenrui/shk/AgentCity/repos/CoMaPOI`

**Key Components**:
- `agents.py`: Multi-agent definitions (CustomDialogAgent, CustomReActAgent, CustomDictDialogAgent)
- `inference_forward_new.py`: Multi-agent orchestration pipeline
- `utils.py`: Data processing utilities
- `prompt_provider.py`: Agent prompt templates
- `rag/RAG.py`: RAG-based candidate retrieval

**Architecture**:
CoMaPOI uses a 3-agent LLM-powered system for next POI prediction:
1. **Profiler Agent**: Analyzes historical trajectories to build long-term user profiles
2. **Forecaster Agent**: Analyzes current trajectory for short-term mobility patterns
3. **Final_Predictor Agent**: Combines insights from both agents to predict next POI

## Adapted Model

**Location**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/CoMaPOI.py`

**Configuration**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traj_loc_pred/CoMaPOI.json`

## Key Adaptations

### 1. AbstractModel Inheritance

The adapted model inherits from `AbstractModel` as required by LibCity:

```python
class CoMaPOI(AbstractModel):
    def __init__(self, config, data_feature):
        super(CoMaPOI, self).__init__(config, data_feature)
```

### 2. Required Methods Implementation

- **`forward(batch)`**: Runs the multi-agent prediction pipeline
- **`predict(batch)`**: Returns POI prediction scores
- **`calculate_loss(batch)`**: Returns cross-entropy loss for evaluation

### 3. Data Format Conversion

The model converts LibCity's batch format to agent prompts:

```python
def _batch_to_trajectory_str(self, batch, batch_idx):
    # Extracts user_id, current_loc, current_tim, history_loc, history_tim
    # Converts to JSON strings for LLM prompts
```

### 4. LLM Client Wrapper

A new `LLMClient` class wraps OpenAI-compatible API calls:

```python
class LLMClient:
    def __init__(self, model_name, api_base_url, api_key, temperature, max_tokens):
        # Initializes OpenAI client

    def generate(self, prompt, system_prompt=""):
        # Generates LLM response
```

### 5. Fallback Mode

Since LLM inference may not always be available, a fallback mode returns random predictions:

```python
def _fallback_predict(self, batch_size):
    return torch.rand(batch_size, self.loc_size, device=self.device)
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `llm_model_path` | `Qwen/Qwen2.5-1.5B-Instruct` | LLM model name/path |
| `api_base_url` | `http://localhost:7863/v1` | LLM API endpoint |
| `api_key` | `EMPTY` | API key for authentication |
| `temperature` | `0.0` | LLM sampling temperature |
| `max_tokens` | `512` | Maximum tokens per response |
| `top_k` | `10` | Number of POI predictions |
| `num_candidate` | `25` | Number of candidate POIs |
| `use_rag` | `false` | Enable RAG candidate retrieval |
| `fallback_mode` | `true` | Use random predictions when LLM unavailable |

## Data Feature Requirements

| Key | Description |
|-----|-------------|
| `loc_size` | Number of POI locations |
| `uid_size` | Number of users |
| `poi_info` | Optional POI metadata (category, lat, lon) |

## Usage

### Basic Usage

```python
from libcity.config import ConfigParser
from libcity.data import get_dataset
from libcity.model import get_model

config = ConfigParser(
    task='traj_loc_pred',
    model='CoMaPOI',
    dataset='foursquare_nyc'
)

dataset = get_dataset(config)
model = get_model(config, dataset.data_feature)

# Run prediction
batch = next(iter(dataset.get_data()))
predictions = model.predict(batch)
```

### With Local LLM Server

Start a local LLM server (e.g., vLLM):

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --port 7863
```

Then run the model:

```python
config['api_base_url'] = 'http://localhost:7863/v1'
config['llm_model_path'] = 'Qwen/Qwen2.5-1.5B-Instruct'
```

## Limitations

1. **LLM Dependency**: Requires external LLM server for inference
2. **Not Trainable**: Uses pre-trained LLMs, no gradient-based training
3. **Performance**: Depends on LLM quality and prompt engineering
4. **Latency**: LLM inference adds significant latency per sample
5. **RAG Not Fully Integrated**: RAG module requires additional setup

## Future Improvements

1. Integrate RAG module for better candidate retrieval
2. Add batch LLM inference for efficiency
3. Implement caching for repeated prompts
4. Add support for more LLM providers (Anthropic, Google, etc.)

## Files Modified

- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/__init__.py`: Added CoMaPOI import
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`: Added CoMaPOI to allowed models

## Files Created

- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/CoMaPOI.py`: Main model file
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traj_loc_pred/CoMaPOI.json`: Configuration file
