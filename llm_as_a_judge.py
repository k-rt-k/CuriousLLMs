"""
LLM-as-a-Judge for evaluating answers using Google's Gemini models or Tinker models.
Supports both synchronous and batch API calls.

Example usage:

    # Using Gemini
    def make_prompt(question: str, reference: str) -> str:
        return f"Evaluate the answer. Question: {question}\\nReference: {reference}"
    
    gemini_judge = GeminiJudge(
        model_name="gemini-2.0-flash-lite",
        system_prompt=make_prompt
    )
    results = gemini_judge.judge_synchronous(questions, answers, references)
    
    # Using Tinker (requires tinker-cookbook)
    tinker_judge = TinkerJudge(
        base_model="meta-llama/Llama-3.1-8B-Instruct",
        renderer_name="llama3",
        system_prompt=make_prompt
    )
    results = tinker_judge.judge_synchronous(questions, answers, references)
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Iterable

from google.genai import Client as Gemini, types
from openai import OpenAI

from pydantic import BaseModel, Field
from enum import Enum

import asyncio
import tinker
from google.api_core.exceptions import ResourceExhausted, TooManyRequests
from tenacity import (
            retry,
            stop_after_attempt,
            wait_exponential,
            retry_if_exception_type,
            RetryError
        )

class JudgeResult(BaseModel):
    """Structured output format for LLM judge evaluations."""
    
    explanation: str = Field(..., description="Explanation provided by the judge for the score.")
    
    class Verdict(str, Enum):
        """Possible verdicts for answer evaluation."""
        UNCERTAIN = "uncertain"
        CORRECT = "correct"
        INCORRECT = "incorrect"

    verdict: Verdict = Field(
        Verdict.UNCERTAIN, 
        description="One of: uncertain, correct, incorrect."
    )

## TODO: prompt design
GEMINI_MATH_JUDGE_SYSTEM_PROMPT: Callable[[str, str], str] = \
    lambda question, reference: f"""
You are an objective automatic grader. You will be given:
- QUESTION: {question}
- REFERENCE ANSWER: {reference}

Task:
- Compare the candidate response (provided as the user message) to the question and reference.
- Return a concise judgement and a brief explanation.

Verdict rules:
- "correct" — the response answers the given question, is mathematically/semantically correct AND the reasoning process is valid and complete.
- "incorrect" — the answer has a clear error, wrong final result, or misses essential necessary steps.
- "uncertain" — the answer is ambiguous, incomplete, or cannot be judged from the information provided.

Output requirements (strict):
- Return exactly one valid JSON object and nothing else.
- The object must contain exactly two fields:
  {{ "explanation": "<short justification, 1-3 sentences>", "verdict": "correct" | "incorrect" | "uncertain" }}
- Do NOT include any additional text, markdown, code fences, or fields.
- Keep the explanation factual and cite the key reason (e.g., wrong arithmetic step, missing assumption, matches reference).

Be concise and precise.
""".strip()

TINKER_SYSTEM_PROMPT: Callable[[str, str], str] = GEMINI_MATH_JUDGE_SYSTEM_PROMPT

class jsonlParser:
    """Utility class for reading and writing JSONL (JSON Lines) files."""

    @staticmethod
    def write_records(path: str, records: Iterable[Dict[str, Any]]) -> None:
        """Write a list of JSON-serializable dicts to a JSONL file (one JSON object per line)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        

    @staticmethod
    def read_records(path: str) -> List[Dict[str, Any]]:
        """Read a JSONL file and return a list of dicts (skips empty lines)."""
        records: List[Dict[str, Any]] = []
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    @staticmethod
    def write_openai_chat_jsonl(
        path: str,
        messages: Iterable[Dict[str, str]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write JSONL file formatted for OpenAI-style/GenAI chat batch APIs."""
        if metadata is None:
            metadata = {}
        jsonlParser.write_records(path, map(lambda m: {"input": m, **metadata}, messages))

    @staticmethod
    def read_openai_chat_jsonl(path: str) -> List[Dict[str, Any]]:
        """Read JSONL file formatted for OpenAI-style/GenAI chat batch APIs."""
        records = jsonlParser.read_records(path)
        return [
            {
                "messages": [
                    {"role": "user", "content": rec["user"]},
                    {"role": "assistant", "content": rec["assistant"]},
                ],
                "metadata": {
                    "question": rec["question"],
                    "reference": rec["reference"],
                } if "metadata" in rec else None
            }
            for rec in records
        ]


class GeminiJudge:
    """LLM-as-a-Judge using Google's Gemini models for answer evaluation."""
    
    def __init__(
        self, 
        model_name: str = "gemini-2.0-flash-lite", 
        system_prompt: Optional[Callable[[str, str], str]] = GEMINI_MATH_JUDGE_SYSTEM_PROMPT
    ):
        """
        Initialize the Gemini judge.
        
        Args:
            model_name: Name of the Gemini model to use.
            system_prompt: Function that takes (question, reference) and returns a system prompt string.
        """
        self.model_name = model_name

        API_KEY = os.getenv("GEMINI_API_KEY")
        if not API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        self.gemini_client = Gemini(api_key=API_KEY)
        self.openai_client = OpenAI(
            api_key=API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.system_prompt = system_prompt

    def judge_synchronous(
        self, 
        questions: List[str], 
        answers: List[str], 
        references: List[str], 
        retry_on_error: bool = True
    ) -> List[JudgeResult]:
        """
        Evaluate answers synchronously (faster but more expensive for large batches).
        
        Args:
            questions: List of questions.
            answers: List of answers to evaluate.
            references: List of reference answers.
            retry_on_error: Whether to retry on API errors (not yet implemented).
        """
        if not (len(questions) == len(answers) == len(references)):
            raise ValueError("questions, answers and references must have the same length")
        
        
        
        @retry(
            retry=retry_if_exception_type((ResourceExhausted, TooManyRequests, Exception)),
            wait=wait_exponential(multiplier=1, min=2, max=60),
            stop=stop_after_attempt(3) if retry_on_error else stop_after_attempt(1),
            reraise=False
        )
        def judge_with_retry(q: str, a: str, r: str) -> JudgeResult:
            return self.judge_single(q, a, r)
        
        results: List[JudgeResult] = []
        for q, a, r in zip(questions, answers, references):
            try:
                result = judge_with_retry(q, a, r)
                results.append(result)
            except RetryError as e:
                # All retries exhausted
                results.append(
                    JudgeResult(
                        explanation=f"Error after retries: {str(e.last_attempt.exception())}",
                        verdict=JudgeResult.Verdict.UNCERTAIN
                    )
                )
            except Exception as e:
                # Unexpected error
                results.append(
                    JudgeResult(
                        explanation=f"Unexpected error: {str(e)}",
                        verdict=JudgeResult.Verdict.UNCERTAIN
                    )
                )
        return results

    def judge_single(self, question: str, answer: str, reference: str) -> JudgeResult:
        """
        Evaluate a single answer using the LLM judge.
        
        Args:
            question: The question being answered.
            answer: The answer to evaluate.
            reference: The reference/correct answer.
        """
        if not self.system_prompt:
            raise ValueError("system_prompt must be provided to use judge_single")
        
        response = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt(question, reference)},
                {"role": "user", "content": answer}
            ],
            temperature=0.0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "judge_result",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "explanation": {"type": "string"},
                            "verdict": {"type": "string", "enum": ["uncertain", "correct", "incorrect"]},
                        },
                        "required": ["explanation", "verdict"],
                    },
                    "strict": True
                }
            }
        )
        
        if not response.choices or not response.choices[0].message.content:
            raise RuntimeError("Failed to parse judge response")
        
        return JudgeResult.model_validate_json(response.choices[0].message.content)

    def judge_batch(
        self, 
        questions: List[str], 
        answers: List[str], 
        references: List[str], 
        filename_fingerprint: Optional[str] = None, 
        retry_on_error: bool = False
    ) -> List[JudgeResult]:
        """
        Evaluate answers using batch API (slower but cheaper for large datasets).
        Polls for completion up to 24 hours.
        
        Args:
            questions: List of questions.
            answers: List of answers to evaluate.
            references: List of reference answers.
            filename_fingerprint: Optional identifier for batch files (default: timestamp).
            retry_on_error: Whether to retry failed requests (not yet implemented). If false, errors will be evaluated as 'uncertain'.
        """
        if retry_on_error:
            raise NotImplementedError("retry_on_error is not yet implemented for judge_batch")

        if not (len(questions) == len(answers) == len(references)):
            raise ValueError("questions, answers and references must have the same length")
        
        if not self.system_prompt:
            raise ValueError("system_prompt must be provided to use judge_batch")
        
        if not filename_fingerprint:
            filename_fingerprint = datetime.now().strftime("%y%m%d-%H%M%S")

        # Step 1: Prepare batch input in OpenAI Batch API format
        batch_input_file = os.path.join("batch_data", f'{self.model_name}-{filename_fingerprint}-input.jsonl')
        batch_records = self._create_batch_records(questions, answers, references)
        jsonlParser.write_records(batch_input_file, batch_records)

        # Step 2: Upload batch input file
        uploaded_file = self._upload_batch_file(batch_input_file, filename_fingerprint)
        
        if not uploaded_file.name:
            raise RuntimeError("File upload failed - no file ID returned")
        
        # Step 3: Create and monitor batch job
        batch = self._create_batch_job(uploaded_file.name)
        completed_batch = self._wait_for_batch_completion(batch.id)
        
        if not completed_batch.output_file_id:
            raise RuntimeError("Batch completed but no output file ID available")
        
        # Step 4: Download and parse results
        results = self._parse_batch_results(completed_batch.output_file_id, filename_fingerprint)
        
        return results
    
    def _create_batch_records(
        self, 
        questions: List[str], 
        answers: List[str], 
        references: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Create batch input records in OpenAI Batch API format.
        
        Format: {"custom_id": "request-N", "method": "POST", "url": "/v1/chat/completions", "body": {...}}
        """
        if not self.system_prompt:
            raise ValueError("system_prompt must be set before creating batch records")
        
        batch_records = []
        for idx, (q, a, r) in enumerate(zip(questions, answers, references)):
            batch_records.append({
                "custom_id": f"request-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": self.system_prompt(q, r)},
                        {"role": "user", "content": a}
                    ],
                    "temperature": 0.0
                }
            })
        return batch_records
    
    def _upload_batch_file(self, file_path: str, display_name: str) -> types.File:
        """Upload batch input file to Gemini."""
        uploaded_file = self.gemini_client.files.upload(
            file=file_path,
            config=types.UploadFileConfig(
                display_name=f'{self.model_name}-{display_name}',
                mime_type='application/jsonl'
            )
        )
        
        return uploaded_file
    
    def _create_batch_job(self, input_file_id: str):
        """Create a batch processing job."""
        return self.openai_client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
    
    def _wait_for_batch_completion(self, batch_id: str):
        """Poll batch job status until completion (polls every 30 seconds)."""
        time.sleep(5)  # Initial wait before polling
        while True:
            batch = self.openai_client.batches.retrieve(batch_id)
            if batch.status in ('completed', 'failed', 'cancelled', 'expired'):
                break
            print(f"Batch not finished. Current state: {batch.status}. Waiting 30 seconds...")
            time.sleep(30)
        
        print(f"Batch finished: {batch}")
        
        if batch.status != 'completed':
            raise RuntimeError(f"Batch processing failed with status: {batch.status}")
        
        return batch
    
    def _clean_json_string(self, s: str) -> str:
        """Remove markdown fences from a string that should be JSON."""
        s = s.strip()
        if s.startswith("```json"):
            s = s[7:]
        elif s.startswith("```"):
            s = s[3:]
        
        if s.endswith("```"):
            s = s[:-3]
            
        return s.strip()

    def _parse_batch_results(self, output_file_id: str, filename_fingerprint: str) -> List[JudgeResult]:
        """
        Download and parse batch results, maintaining original order.
        
        Expected output format: {"id": "...", "custom_id": "request-N", "response": {"body": {...}}, "error": {...}}
        """
        # Download results
        file_content = self.gemini_client.files.download(file=output_file_id).decode('utf-8')

        # Save the raw output file
        output_file_path = os.path.join("batch_data", f'{self.model_name}-{filename_fingerprint}-output.jsonl')
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(file_content)
        print(f"Batch output saved to {output_file_path}")

        # Parse JSONL output
        output_records = []
        for line in file_content.strip().split('\n'):
            if line:
                output_records.append(json.loads(line))
        
        # Sort by custom_id to maintain original order
        output_records.sort(key=lambda x: int(x['custom_id'].split('-')[1]))
        
        # Extract JudgeResult from each response
        results: List[JudgeResult] = []
        for record in output_records:
            try:
                if 'error' not in record and record.get('response') and record['response'].get('body'):
                    body = record['response']['body']
                    if body.get('choices') and len(body['choices']) > 0:
                        content = body['choices'][0].get('message', {}).get('content')
                        if content:
                            cleaned_content = self._clean_json_string(content)
                            judge_result = JudgeResult.model_validate_json(cleaned_content)
                            results.append(judge_result)
                        else:
                            raise ValueError("No content in message")
                    else:
                        raise ValueError("No choices in response body")
                else:
                    raise ValueError("Error in response or missing body")
            except Exception as e:
                # Handle error cases gracefully
                error_msg = record.get('error', {}).get('message', str(e))
                print(f"Error in record {record.get('custom_id')}: {error_msg}")
                results.append(
                    JudgeResult(
                        explanation=f"Error: {error_msg}", 
                        verdict=JudgeResult.Verdict.UNCERTAIN
                    )
                )
        
        return results
    
################################################################################## 
class TinkerJudge:
    """
    LLM-as-a-Judge using Tinker's models for answer evaluation.
    
    Requires: tinker-cookbook to be installed and available in the environment.
    """
    
    def __init__(
        self,
        base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        renderer_name: str = "llama3",
        system_prompt: Optional[Callable[[str, str], str]] = None,
        service_client: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ):
        """
        Initialize the Tinker judge.
        
        Args:
            base_model: Name of the base model to use (e.g., "meta-llama/Llama-3.1-8B-Instruct").
            renderer_name: Name of the renderer to use (e.g., "llama3", "role_colon").
            system_prompt: Function that takes (question, reference) and returns a system prompt string.
            service_client: Optional tinker.ServiceClient instance. If None, creates a new one.
            tokenizer: Optional tokenizer. If None, will load from base_model.
        """
        raise NotImplementedError("code hasnt been read, far from tested")

        import tinker
        from tinker_cookbook import renderers  # type: ignore
        from tinker_cookbook.tokenizer_utils import get_tokenizer  # type: ignore
        
        self.base_model = base_model
        self.renderer_name = renderer_name
        self.system_prompt = system_prompt
        
        # Initialize Tinker clients
        if service_client is None:
            service_client = tinker.ServiceClient()
        self.service_client = service_client
        self.sampling_client = service_client.create_sampling_client(base_model=base_model)
        
        # Initialize tokenizer and renderer
        if tokenizer is None:
            tokenizer = get_tokenizer(base_model)
        self.tokenizer = tokenizer
        self.renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    
    def judge_synchronous(
        self,
        questions: List[str],
        answers: List[str],
        references: List[str],
        retry_on_error: bool = True,
    ) -> List[JudgeResult]:
        """
        Evaluate answers synchronously (faster but more expensive for large batches).
        
        Args:
            questions: List of questions.
            answers: List of answers to evaluate.
            references: List of reference answers.
            retry_on_error: Whether to retry on API errors (not yet implemented).
        """
        if not (len(questions) == len(answers) == len(references)):
            raise ValueError("questions, answers and references must have the same length")
        
        return [self.judge_single(q, a, r) for q, a, r in zip(questions, answers, references)]
    
    def judge_single(self, question: str, answer: str, reference: str) -> JudgeResult:
        """
        Evaluate a single answer using the LLM judge.
        
        Args:
            question: The question being answered.
            answer: The answer to evaluate.
            reference: The reference/correct answer.
        """
        if not self.system_prompt:
            raise ValueError("system_prompt must be provided to use judge_single")
        
        
        
        # Build the conversation messages
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt(question, reference)},
            {"role": "user", "content": answer},
        ]
        
        # Render the conversation for the model
        model_input = self.renderer.build_generation_prompt(messages)
        
        # Sample from the model (use async wrapper)
        async def async_sample():
            response = await self.sampling_client.sample_async(
                model_input,
                num_samples=1,
                sampling_params=tinker.SamplingParams(
                    temperature=0.0,
                    max_tokens=512,
                    stop=self.renderer.get_stop_sequences(),
                ),
            )
            return response
        
        # Run async function
        response = asyncio.run(async_sample())
        
        # Decode the response
        parsed_message, _success = self.renderer.parse_response(response.sequences[0].tokens)
        response_text = parsed_message.get("content", "").strip()
        
        # Parse the response into JudgeResult
        try:
            judge_result = self._parse_structured_output(response_text)
        except (json.JSONDecodeError, ValueError):
            # Fallback: try to extract verdict and explanation from text
            judge_result = self._parse_text_output(response_text)
        
        return judge_result
    
    def judge_batch(
        self,
        questions: List[str],
        answers: List[str],
        references: List[str],
        filename_fingerprint: Optional[str] = None,
        retry_on_error: bool = False,
    ) -> List[JudgeResult]:
        """
        Evaluate answers in batch (currently falls back to synchronous evaluation).
        
        Args:
            questions: List of questions.
            answers: List of answers to evaluate.
            references: List of reference answers.
            filename_fingerprint: Optional identifier (unused, for API compatibility).
            retry_on_error: Whether to retry failed requests (not yet implemented).
        """
        # Note: Tinker doesn't have a native batch API like Gemini
        # So we fall back to synchronous evaluation
        return self.judge_synchronous(questions, answers, references, retry_on_error)
    
    def _parse_structured_output(self, text: str) -> JudgeResult:
        """Try to parse structured JSON output."""
        # Clean up markdown code blocks if present
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        data = json.loads(text)
        return JudgeResult.model_validate(data)
    
    def _parse_text_output(self, text: str) -> JudgeResult:
        """Parse unstructured text output and extract verdict."""
        text_lower = text.lower()
        
        # Try to detect verdict
        if "correct" in text_lower and "incorrect" not in text_lower:
            verdict = JudgeResult.Verdict.CORRECT
        elif "incorrect" in text_lower or "wrong" in text_lower:
            verdict = JudgeResult.Verdict.INCORRECT
        else:
            verdict = JudgeResult.Verdict.UNCERTAIN
        
        return JudgeResult(
            explanation=text,
            verdict=verdict
        )

