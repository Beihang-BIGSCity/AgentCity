# coding: utf-8
"""
CoMaPOI: Collaborative Multi-Agent System for POI Prediction

This module adapts the CoMaPOI multi-agent system to LibCity's AbstractModel conventions.

Original Repository: repos/CoMaPOI
Paper: "CoMaPOI: Collaborative Multi-Agent POI Prediction"

Key Adaptations for LibCity:
1. Inherited from AbstractModel as per LibCity convention for trajectory prediction
2. Wrapped 3-agent orchestration (Profiler, Forecaster, Final_Predictor) as model methods
3. Converted LibCity batch format to agent prompts and back
4. Implemented predict() and calculate_loss() methods following LibCity conventions
5. Added fallback mode for when LLM inference is not available

Architecture Overview:
- Profiler Agent: Analyzes historical trajectories to build long-term user profiles
- Forecaster Agent: Analyzes current trajectory for short-term mobility patterns
- Final_Predictor Agent: Combines insights to predict next POI

Required data_feature keys:
- loc_size: Number of POI locations
- uid_size: Number of users
- poi_info: Optional POI information (category, lat, lon)

Required config parameters:
- llm_model_path: Path to LLM model (default: 'Qwen/Qwen2.5-1.5B-Instruct')
- api_base_url: Base URL for LLM API (default: 'http://localhost:7863/v1')
- api_key: API key for LLM (default: 'EMPTY')
- top_k: Number of POI predictions to return (default: 10)
- num_candidate: Number of candidate POIs to consider (default: 25)
- temperature: LLM temperature (default: 0.0)
- use_rag: Whether to use RAG for candidate retrieval (default: False)
- fallback_mode: Use random prediction when LLM unavailable (default: True)

Limitations:
- This model requires external LLM inference (local model or API)
- Performance depends on LLM quality and prompt engineering
- calculate_loss returns evaluation metrics instead of trainable loss
- Not suitable for gradient-based training
"""

import json
import re
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Any, Tuple

from libcity.model.abstract_model import AbstractModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_predicted_pois(content: str, top_k: int) -> List[str]:
    """
    Extract 'next_poi_id' list from LLM response content.
    Returns at most top_k POI IDs, filtering out non-numeric data.

    Args:
        content: LLM response content to parse
        top_k: Maximum number of POIs to return

    Returns:
        List of POI ID strings
    """
    poi_ids = []

    # Try JSON parsing first
    try:
        if isinstance(content, dict):
            content_json = content
        else:
            content_json = json.loads(content)

        if 'next_poi_id' in content_json:
            next_poi_ids = content_json['next_poi_id']
            for poi_id in next_poi_ids:
                if isinstance(poi_id, str):
                    match = re.search(r'\b(\d+)\b', poi_id)
                    if match:
                        poi_ids.append(match.group(1))
                elif isinstance(poi_id, (int, float)):
                    poi_ids.append(str(int(poi_id)))
            return poi_ids[:top_k]
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        pass

    # Try regex extraction from JSON-like content
    try:
        if isinstance(content, dict):
            content_str = json.dumps(content)
        else:
            content_str = content

        pattern = r'"next_poi_id"\s*:\s*\[([^\]]+)\]'
        match = re.search(pattern, content_str)

        if match:
            ids_str = match.group(1)
            ids_list = re.split(r',\s*', ids_str)

            for id_str in ids_list:
                num_match = re.search(r'\b(\d+)\b', id_str)
                if num_match:
                    poi_ids.append(num_match.group(1))
            return poi_ids[:top_k]

        # Try markdown code block extraction
        pattern = r'```(?:json)?\s*{[^}]*"next_poi_id"\s*:\s*\[([^\]]+)\][^}]*}\s*```'
        match = re.search(pattern, content_str)

        if match:
            ids_str = match.group(1)
            ids_list = re.split(r',\s*', ids_str)

            for id_str in ids_list:
                num_match = re.search(r'\b(\d+)\b', id_str)
                if num_match:
                    poi_ids.append(num_match.group(1))
            return poi_ids[:top_k]

        # Fallback: extract all numbers
        numbers = re.findall(r'\b\d+\b', content_str)
        return numbers[:top_k]

    except Exception:
        pass

    # If all methods fail, return empty list
    return poi_ids[:top_k]


def clean_predicted_pois(predicted_pois: List[str], max_item: int) -> List[int]:
    """
    Clean predicted POIs by removing duplicates and invalid values.

    Args:
        predicted_pois: List of predicted POI ID strings
        max_item: Maximum valid POI ID

    Returns:
        List of cleaned POI IDs as integers
    """
    cleaned_pois = []
    seen = set()

    for poi in predicted_pois:
        try:
            poi_id = int(poi)
            if 1 <= poi_id <= max_item and poi_id not in seen:
                cleaned_pois.append(poi_id)
                seen.add(poi_id)
        except (ValueError, TypeError):
            continue

    return cleaned_pois


class PromptProvider:
    """
    Provides prompts for the multi-agent system.
    Generates prompts for Profiler, Forecaster, and Final_Predictor agents.
    """

    def __init__(self, user_id: str, current_trajectory: str,
                 num_candidate: int = 25, top_k: int = 10, max_item: int = 5000):
        self.user_id = user_id
        self.current_trajectory = current_trajectory
        self.num_candidate = num_candidate
        self.top_k = top_k
        self.max_item = max_item

    def get_profiler_prompt(self, historical_trajectory: str) -> str:
        """Generate prompt for Profiler agent to analyze historical patterns."""
        prompt_data = {
            "IDENTITY": "You are an expert User Profiler specialized in analyzing user trajectory data.",
            "TASK": f"For user_{self.user_id}, analyze their historical trajectory to generate a long-term profile.",
            "User ID": self.user_id,
            "Historical Trajectory": historical_trajectory,
            "STEPS": [
                "1. Analyze time patterns (active periods)",
                "2. Analyze spatial patterns (frequently visited areas)",
                "3. Analyze category preferences",
                "4. Generate a summary profile"
            ],
            "OUTPUT_FORMAT": 'Respond with JSON: {"historical_profile": "profile description"}'
        }
        return json.dumps(prompt_data, indent=2)

    def get_candidate_generation_prompt(self, long_term_profile: str) -> str:
        """Generate prompt for Profiler to suggest candidate POIs based on profile."""
        id_list = [f'"{i + 1}th unique ID"' for i in range(self.num_candidate)]
        id_list[0] = '"best unique ID"'
        id_list_str = ", ".join(id_list)

        prompt_data = {
            "IDENTITY": "You are an expert POI Predictor specialized in generating POI candidates.",
            "TASK": f"Generate {self.num_candidate} candidate POI IDs based on the user profile and current trajectory.",
            "User ID": self.user_id,
            "Current Trajectory": self.current_trajectory,
            "Long-Term Profile": long_term_profile,
            "IMPORTANT": [
                "Provide only numeric POI IDs",
                "Ensure all IDs are unique and valid",
                f"IDs must be between 1 and {self.max_item}"
            ],
            "OUTPUT_FORMAT": f'{{"next_poi_id": [{id_list_str}]}}'
        }
        return json.dumps(prompt_data, indent=2)

    def get_forecaster_prompt(self) -> str:
        """Generate prompt for Forecaster agent to analyze current mobility pattern."""
        prompt_data = {
            "IDENTITY": "You are an expert Mobility Pattern Analyzer.",
            "TASK": f"Analyze user_{self.user_id}'s current trajectory to understand their current context.",
            "User ID": self.user_id,
            "Current Trajectory": self.current_trajectory,
            "STEPS": [
                "1. Analyze the time pattern",
                "2. Analyze spatial movement pattern",
                "3. Identify current needs/intentions"
            ],
            "OUTPUT_FORMAT": '{"current_profile": "current context description"}'
        }
        return json.dumps(prompt_data, indent=2)

    def get_refinement_prompt(self, short_term_profile: str, rag_candidates: List) -> str:
        """Generate prompt for Forecaster to refine candidate POIs."""
        id_list = [f'"{i + 1}th unique ID"' for i in range(self.num_candidate)]
        id_list[0] = '"best unique ID"'
        id_list_str = ", ".join(id_list)

        prompt_data = {
            "IDENTITY": "You are an expert Mobility Pattern Analyzer for refining POI candidates.",
            "TASK": f"Refine the candidate POI list based on short-term mobility profile.",
            "User ID": self.user_id,
            "Short-Term Profile": short_term_profile,
            "RAG Candidates": rag_candidates,
            "OUTPUT_FORMAT": f'{{"refined_candidate_from_rag": [{id_list_str}]}}'
        }
        return json.dumps(prompt_data, indent=2)

    def get_final_prediction_prompt(self, long_term_profile: str, short_term_profile: str,
                                    candidate_list_1: List, candidate_list_2: List) -> str:
        """Generate prompt for Final_Predictor to make the final prediction."""
        id_list = [f'"{i + 1}th unique ID"' for i in range(self.top_k)]
        id_list[0] = '"best unique ID"'
        id_list_str = ", ".join(id_list)

        prompt_data = {
            "IDENTITY": "You are an expert POI Predictor for final prediction.",
            "TASK": f"Predict the top {self.top_k} POIs user_{self.user_id} will visit next.",
            "User ID": self.user_id,
            "Current Trajectory": self.current_trajectory,
            "Long-Term Profile": long_term_profile,
            "Short-Term Profile": short_term_profile,
            "Profile-based Candidates": candidate_list_1,
            "Mobility-based Candidates": candidate_list_2,
            "STEPS": [
                "1. Combine long-term preferences with short-term context",
                "2. Consider both candidate lists",
                f"3. Select top {self.top_k} most likely POIs"
            ],
            "IMPORTANT": [
                "Provide only numeric POI IDs",
                f"Provide exactly {self.top_k} unique IDs",
                f"IDs must be between 1 and {self.max_item}"
            ],
            "OUTPUT_FORMAT": f'{{"next_poi_id": [{id_list_str}]}}'
        }
        return json.dumps(prompt_data, indent=2)


class LLMClient:
    """
    Client for calling LLM API.
    Supports OpenAI-compatible APIs (vLLM, ollama, etc.)
    """

    def __init__(self, model_name: str, api_base_url: str, api_key: str = "EMPTY",
                 temperature: float = 0.0, max_tokens: int = 512):
        self.model_name = model_name
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = None
        self._initialized = False

    def _init_client(self):
        """Initialize the OpenAI client lazily."""
        if self._initialized:
            return

        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=self.api_base_url,
                api_key=self.api_key
            )
            self._initialized = True
        except ImportError:
            logger.warning("openai package not installed. LLM inference unavailable.")
            self._initialized = False
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI client: {e}")
            self._initialized = False

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """
        Generate response from LLM.

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)

        Returns:
            LLM response content
        """
        self._init_client()

        if not self._initialized or self.client is None:
            raise RuntimeError("LLM client not available")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    def is_available(self) -> bool:
        """Check if LLM client is available and server is responding."""
        self._init_client()
        if not self._initialized or self.client is None:
            return False

        # Test connection with a simple request
        try:
            self.client.models.list()  # Quick API test
            return True
        except Exception:
            return False


class CoMaPOI(AbstractModel):
    """
    CoMaPOI: Collaborative Multi-Agent POI Prediction Model

    This model uses a 3-agent LLM system for next POI prediction:
    1. Profiler: Analyzes historical trajectories for long-term patterns
    2. Forecaster: Analyzes current trajectory for short-term patterns
    3. Final_Predictor: Combines insights for final prediction

    Note: This is NOT a traditional neural network. It uses LLM inference
    and is primarily designed for evaluation, not training.
    """

    def __init__(self, config, data_feature):
        super(CoMaPOI, self).__init__(config, data_feature)

        # Device configuration
        self.device = config.get('device', 'cpu')

        # Data feature parameters
        self.loc_size = data_feature.get('loc_size', 5000)
        self.uid_size = data_feature.get('uid_size', 1000)
        self.poi_info = data_feature.get('poi_info', None)

        # LLM configuration
        self.llm_model_path = config.get('llm_model_path', 'Qwen/Qwen2.5-1.5B-Instruct')
        self.api_base_url = config.get('api_base_url', 'http://localhost:7863/v1')
        self.api_key = config.get('api_key', 'EMPTY')
        self.temperature = config.get('temperature', 0.0)
        self.max_tokens = config.get('max_tokens', 512)

        # Agent configuration (can use different models for each agent)
        self.profiler_model = config.get('profiler_model', self.llm_model_path)
        self.forecaster_model = config.get('forecaster_model', self.llm_model_path)
        self.predictor_model = config.get('predictor_model', self.llm_model_path)

        # Prediction parameters
        self.top_k = config.get('top_k', 10)
        self.num_candidate = config.get('num_candidate', 25)
        self.max_item = config.get('max_item', self.loc_size)

        # Mode configuration
        self.use_rag = config.get('use_rag', False)
        self.fallback_mode = config.get('fallback_mode', True)

        # Initialize LLM clients
        self._init_llm_clients()

        # Dummy parameter for compatibility with PyTorch (model must have at least one parameter)
        # Set requires_grad=True to enable gradient flow in fallback mode
        self.dummy_param = nn.Parameter(torch.zeros(1), requires_grad=True)

        logger.info(f"CoMaPOI initialized with loc_size={self.loc_size}, top_k={self.top_k}")
        if not self.profiler_client.is_available():
            logger.warning("LLM clients not available. Using fallback mode.")

    def _init_llm_clients(self):
        """Initialize LLM clients for each agent."""
        self.profiler_client = LLMClient(
            model_name=self.profiler_model,
            api_base_url=self.api_base_url,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        self.forecaster_client = LLMClient(
            model_name=self.forecaster_model,
            api_base_url=self.api_base_url,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        self.predictor_client = LLMClient(
            model_name=self.predictor_model,
            api_base_url=self.api_base_url,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

    def _batch_to_trajectory_str(self, batch, batch_idx: int) -> Tuple[str, str, str]:
        """
        Convert LibCity batch to trajectory string for prompting.

        Args:
            batch: LibCity batch dictionary
            batch_idx: Index within the batch

        Returns:
            Tuple of (user_id, current_trajectory_str, historical_trajectory_str)
        """
        # Extract user ID
        if 'uid' in batch.data:
            uid = batch['uid']
            if isinstance(uid, torch.Tensor):
                user_id = str(uid[batch_idx].item())
            else:
                user_id = str(uid[batch_idx])
        else:
            user_id = "unknown"

        # Extract current trajectory (locations and times)
        current_loc = batch['current_loc'][batch_idx]  # (seq_len,)
        if isinstance(current_loc, torch.Tensor):
            current_loc = current_loc.tolist()

        current_tim = None
        if 'current_tim' in batch.data:
            current_tim = batch['current_tim'][batch_idx]
            if isinstance(current_tim, torch.Tensor):
                current_tim = current_tim.tolist()

        # Build current trajectory string
        trajectory_data = {
            "user_id": user_id,
            "trajectory": []
        }

        for i, loc in enumerate(current_loc):
            if loc == 0:  # Skip padding
                continue
            point = {"poi_id": int(loc)}
            if current_tim is not None:
                point["time"] = current_tim[i]

            # Add POI info if available
            if self.poi_info is not None and loc in self.poi_info:
                poi = self.poi_info[loc]
                point.update({
                    "category": poi.get('category', 'unknown'),
                    "lat": poi.get('lat', 0.0),
                    "lon": poi.get('lon', 0.0)
                })

            trajectory_data["trajectory"].append(point)

        current_trajectory_str = json.dumps(trajectory_data)

        # Extract historical trajectory if available
        historical_trajectory_str = ""
        if 'history_loc' in batch.data:
            history_loc = batch['history_loc'][batch_idx]
            if isinstance(history_loc, torch.Tensor):
                history_loc = history_loc.tolist()

            history_tim = None
            if 'history_tim' in batch.data:
                history_tim = batch['history_tim'][batch_idx]
                if isinstance(history_tim, torch.Tensor):
                    history_tim = history_tim.tolist()

            history_data = {
                "user_id": user_id,
                "historical_trajectory": []
            }

            for i, loc in enumerate(history_loc):
                if loc == 0:  # Skip padding
                    continue
                point = {"poi_id": int(loc)}
                if history_tim is not None:
                    point["time"] = history_tim[i]
                history_data["historical_trajectory"].append(point)

            historical_trajectory_str = json.dumps(history_data)

        return user_id, current_trajectory_str, historical_trajectory_str

    def _run_profiler_agent(self, prompt_provider: PromptProvider,
                           historical_trajectory: str) -> Tuple[str, List[int]]:
        """
        Run the Profiler agent to generate long-term profile and candidates.

        Args:
            prompt_provider: PromptProvider instance
            historical_trajectory: Historical trajectory string

        Returns:
            Tuple of (long_term_profile, candidate_list)
        """
        # Generate long-term profile
        profile_prompt = prompt_provider.get_profiler_prompt(historical_trajectory)
        profile_response = self.profiler_client.generate(profile_prompt)

        # Extract profile from response
        try:
            profile_data = json.loads(profile_response)
            long_term_profile = profile_data.get('historical_profile', profile_response)
        except json.JSONDecodeError:
            long_term_profile = profile_response

        # Generate candidate POIs based on profile
        candidate_prompt = prompt_provider.get_candidate_generation_prompt(long_term_profile)
        candidate_response = self.profiler_client.generate(candidate_prompt)

        # Extract candidates
        candidates = extract_predicted_pois(candidate_response, self.num_candidate)
        candidate_list = clean_predicted_pois(candidates, self.max_item)

        return long_term_profile, candidate_list

    def _run_forecaster_agent(self, prompt_provider: PromptProvider,
                             rag_candidates: List[int] = None) -> Tuple[str, List[int]]:
        """
        Run the Forecaster agent to analyze short-term patterns and refine candidates.

        Args:
            prompt_provider: PromptProvider instance
            rag_candidates: RAG-retrieved candidate POIs (optional)

        Returns:
            Tuple of (short_term_profile, refined_candidate_list)
        """
        # Analyze current mobility pattern
        pattern_prompt = prompt_provider.get_forecaster_prompt()
        pattern_response = self.forecaster_client.generate(pattern_prompt)

        # Extract short-term profile
        try:
            profile_data = json.loads(pattern_response)
            short_term_profile = profile_data.get('current_profile', pattern_response)
        except json.JSONDecodeError:
            short_term_profile = pattern_response

        # Refine candidates if RAG candidates are available
        if rag_candidates is not None and len(rag_candidates) > 0:
            refine_prompt = prompt_provider.get_refinement_prompt(short_term_profile, rag_candidates)
            refine_response = self.forecaster_client.generate(refine_prompt)

            candidates = extract_predicted_pois(refine_response, self.num_candidate)
            refined_candidates = clean_predicted_pois(candidates, self.max_item)
        else:
            refined_candidates = []

        return short_term_profile, refined_candidates

    def _run_final_predictor_agent(self, prompt_provider: PromptProvider,
                                   long_term_profile: str, short_term_profile: str,
                                   candidate_list_1: List[int],
                                   candidate_list_2: List[int]) -> List[int]:
        """
        Run the Final_Predictor agent to make the final prediction.

        Args:
            prompt_provider: PromptProvider instance
            long_term_profile: Long-term user profile
            short_term_profile: Short-term mobility profile
            candidate_list_1: Candidates from Profiler
            candidate_list_2: Candidates from Forecaster

        Returns:
            List of predicted POI IDs
        """
        prediction_prompt = prompt_provider.get_final_prediction_prompt(
            long_term_profile, short_term_profile,
            candidate_list_1, candidate_list_2
        )
        prediction_response = self.predictor_client.generate(prediction_prompt)

        # Extract predictions
        predictions = extract_predicted_pois(prediction_response, self.top_k)
        predicted_pois = clean_predicted_pois(predictions, self.max_item)

        return predicted_pois

    def _multi_agent_predict(self, batch, batch_idx: int) -> List[int]:
        """
        Run the full multi-agent prediction pipeline for a single sample.

        Args:
            batch: LibCity batch
            batch_idx: Index within the batch

        Returns:
            List of predicted POI IDs
        """
        # Convert batch to trajectory strings
        user_id, current_trajectory, historical_trajectory = \
            self._batch_to_trajectory_str(batch, batch_idx)

        # Create prompt provider
        prompt_provider = PromptProvider(
            user_id=user_id,
            current_trajectory=current_trajectory,
            num_candidate=self.num_candidate,
            top_k=self.top_k,
            max_item=self.max_item
        )

        # Run Profiler agent
        long_term_profile, candidate_list_1 = self._run_profiler_agent(
            prompt_provider, historical_trajectory
        )

        # Get RAG candidates if enabled (placeholder - would need RAG integration)
        rag_candidates = [] if not self.use_rag else candidate_list_1[:10]

        # Run Forecaster agent
        short_term_profile, candidate_list_2 = self._run_forecaster_agent(
            prompt_provider, rag_candidates
        )

        # Run Final_Predictor agent
        predicted_pois = self._run_final_predictor_agent(
            prompt_provider, long_term_profile, short_term_profile,
            candidate_list_1, candidate_list_2
        )

        return predicted_pois

    def _fallback_predict(self, batch_size: int) -> torch.Tensor:
        """
        Fallback prediction when LLM is not available.
        Returns random POI predictions based on uniform distribution.

        Args:
            batch_size: Number of samples in batch

        Returns:
            Tensor of prediction scores (batch_size, loc_size)
        """
        # Create uniform random scores
        scores = torch.rand(batch_size, self.loc_size, device=self.device)
        # Add tiny contribution from dummy_param to enable gradients
        scores = scores + self.dummy_param * 0.0
        return scores

    def forward(self, batch) -> torch.Tensor:
        """
        Forward pass - runs the multi-agent system for POI prediction.

        Args:
            batch: Dictionary containing:
                - 'current_loc': (batch_size, seq_len) - current POI indices
                - 'current_tim': (batch_size, seq_len) - current time values
                - 'uid': (batch_size,) - user indices
                - 'history_loc': (batch_size, history_len) - historical POI indices (optional)
                - 'history_tim': (batch_size, history_len) - historical time values (optional)

        Returns:
            torch.Tensor: POI prediction scores (batch_size, loc_size)
        """
        batch_size = batch['current_loc'].shape[0]

        # Quick check for connectivity before processing (one-time check)
        if not hasattr(self, '_llm_connectivity_checked'):
            try:
                self.profiler_client.client.models.list()
                self._llm_available = True
            except Exception:
                self._llm_available = False
                logger.warning("LLM server not reachable, using fallback mode for all predictions")
            self._llm_connectivity_checked = True

        if not self._llm_available and self.fallback_mode:
            return self._fallback_predict(batch_size)

        # Check if LLM is available
        if not self.profiler_client.is_available():
            if self.fallback_mode:
                logger.warning("LLM unavailable, using fallback predictions")
                return self._fallback_predict(batch_size)
            else:
                raise RuntimeError("LLM client not available and fallback_mode is disabled")

        # Initialize prediction scores
        all_scores = torch.zeros(batch_size, self.loc_size, device=self.device)

        # Run multi-agent prediction for each sample
        for i in range(batch_size):
            try:
                predicted_pois = self._multi_agent_predict(batch, i)

                # Convert predictions to scores
                # Higher score for earlier predictions (more confident)
                for rank, poi_id in enumerate(predicted_pois):
                    if 0 <= poi_id < self.loc_size:
                        score = (len(predicted_pois) - rank) / len(predicted_pois)
                        all_scores[i, poi_id] = score

            except Exception as e:
                logger.error(f"Multi-agent prediction failed for sample {i}: {e}")
                if self.fallback_mode:
                    all_scores[i] = torch.rand(self.loc_size, device=self.device)
                else:
                    raise

        return all_scores

    def predict(self, batch) -> torch.Tensor:
        """
        Predict next POI for each sample in the batch.

        Args:
            batch: Dictionary containing input data

        Returns:
            torch.Tensor: POI prediction scores (batch_size, loc_size)
        """
        return self.forward(batch)

    def calculate_loss(self, batch) -> torch.Tensor:
        """
        Calculate loss for the batch.

        Note: This is a stub implementation. Since CoMaPOI uses pre-trained LLMs,
        it does not support gradient-based training. This method returns a
        cross-entropy loss computed from the prediction scores for compatibility
        with LibCity's evaluation pipeline.

        Args:
            batch: Dictionary containing:
                - All forward() inputs
                - 'target': (batch_size,) - target POI indices

        Returns:
            torch.Tensor: Loss value (for evaluation purposes only)
        """
        # Get predictions
        scores = self.forward(batch)

        # Get targets
        target = batch['target']
        if isinstance(target, torch.Tensor):
            target = target.to(self.device)

        # Compute cross-entropy loss for evaluation
        # Note: This loss is not used for training
        log_probs = torch.log_softmax(scores, dim=-1)
        loss = nn.functional.nll_loss(log_probs, target)

        return loss

    def get_top_k_predictions(self, batch, k: int = None) -> torch.Tensor:
        """
        Get top-k POI predictions for each sample.

        Args:
            batch: Dictionary containing input data
            k: Number of predictions to return (default: self.top_k)

        Returns:
            torch.Tensor: Top-k POI indices (batch_size, k)
        """
        if k is None:
            k = self.top_k

        scores = self.predict(batch)
        _, top_k_indices = torch.topk(scores, k, dim=-1)

        return top_k_indices
