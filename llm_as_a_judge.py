"""
LLM-as-a-Judge for evaluating answers using Google's Gemini models.
Supports both synchronous and batch API calls.
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
GEMINI_SYSTEM_PROMPT: Callable[[str, str], str] = lambda question, reference: f''''''

    

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
        system_prompt: Optional[Callable[[str, str], str]] = None
    ):
        """
        Initialize the Gemini judge.
        
        Args:
            model_name: Name of the Gemini model to use.
            system_prompt: Function that takes (question, reference) and returns a system prompt string.
        """
        self.model_name = model_name

        self.gemini_client = Gemini()
        self.openai_client = OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
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
        
        response = self.openai_client.responses.parse(
            model=self.model_name,
            input=[
                {"role": "system", "content": self.system_prompt(question, reference)},
                {"role": "user", "content": answer}
            ],
            temperature=0.0,
            text_format=JudgeResult,
        )
        
        if not response.output_parsed:
            raise RuntimeError("Failed to parse judge response")
        
        return response.output_parsed

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
            retry_on_error: Whether to retry failed requests (not yet implemented).
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
        batch_input_file = f'{self.model_name}-{filename_fingerprint}-input.jsonl'
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
        results = self._parse_batch_results(completed_batch.output_file_id)
        
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
                    "temperature": 0.0,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "judge_result",
                            "schema": JudgeResult.model_json_schema(),
                            "strict": True
                        }
                    }
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
    
    def _parse_batch_results(self, output_file_id: str) -> List[JudgeResult]:
        """
        Download and parse batch results, maintaining original order.
        
        Expected output format: {"id": "...", "custom_id": "request-N", "response": {"body": {...}}, "error": {...}}
        """
        # Download results
        file_content = self.gemini_client.files.download(file=output_file_id).decode('utf-8')

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
            if record.get('response') and record['response'].get('body'):
                content = record['response']['body']['choices'][0]['message']['content']
                judge_result = JudgeResult.model_validate_json(content)
                results.append(judge_result)
            else:
                # Handle error cases gracefully
                error_msg = record.get('error', {}).get('message', 'Unknown error')
                print(f"Error in record {record.get('custom_id')}: {error_msg}")
                results.append(
                    JudgeResult(
                        explanation=f"Error: {error_msg}", 
                        verdict=JudgeResult.Verdict.UNCERTAIN
                    )
                )
        
        return results
    

