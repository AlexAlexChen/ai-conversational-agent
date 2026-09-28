#!/usr/bin/env python
"""
Benchmark Script

This script benchmarks the AI agent on a set of questions and measures
response time, success rate, and other metrics.
"""

import os
import sys
import argparse
import json
import time
import asyncio
import statistics
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

# Add the project root to the path so we can import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.agent import AIAgent
from app.services.llm_service import LLMService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def benchmark_agent(questions: List[str], num_runs: int = 3, timeout: int = 30):
    """Benchmark the AI agent on a set of questions

    Args:
        questions: List of questions to benchmark
        num_runs: Number of times to run each question
        timeout: Timeout in seconds for each run

    Returns:
        Dictionary with benchmark results
    """
    llm_service = LLMService()
    agent = AIAgent(llm_service)

    results = []
    start_time_total = time.time()

    for i, question in enumerate(questions):
        logger.info(f"Benchmarking question {i+1}/{len(questions)}: {question}")
        question_results = []

        for run in range(num_runs):
            logger.info(f"  Run {run+1}/{num_runs}")
            start_time = time.time()

            try:
                # Set a timeout for the run
                response, sources = await asyncio.wait_for(
                    agent.process_message(
                        message=question,
                        session_id=f"benchmark_{i}_{run}"
                    ),
                    timeout=timeout
                )

                success = True
                error = None
            except asyncio.TimeoutError:
                response = "Timeout"
                sources = []
                success = False
                error = "Timeout"
            except Exception as e:
                response = "Error"
                sources = []
                success = False
                error = str(e)

            end_time = time.time()
            elapsed_time = end_time - start_time

            question_results.append({
                "run": run + 1,
                "time": elapsed_time,
                "success": success,
                "error": error,
                "response_length": len(response) if success else 0,
                "num_sources": len(sources) if success else 0
            })

            logger.info(f"    Time: {elapsed_time:.2f}s, Success: {success}")

        # Calculate statistics for this question
        successful_runs = [r for r in question_results if r["success"]]
        success_rate = len(successful_runs) / num_runs if num_runs > 0 else 0

        if successful_runs:
            times = [r["time"] for r in successful_runs]
            avg_time = statistics.mean(times)
            median_time = statistics.median(times)
            min_time = min(times)
            max_time = max(times)
            stddev_time = statistics.stdev(times) if len(times) > 1 else 0

            # Count sources
            source_counts = [r["num_sources"] for r in successful_runs]
            avg_sources = statistics.mean(source_counts)

            # Response length
            response_lengths = [r["response_length"] for r in successful_runs]
            avg_response_length = statistics.mean(response_lengths)
        else:
            avg_time = median_time = min_time = max_time = stddev_time = 0
            avg_sources = avg_response_length = 0

        question_result = {
            "question": question,
            "success_rate": success_rate,
            "avg_time": avg_time,
            "median_time": median_time,
            "min_time": min_time,
            "max_time": max_time,
            "stddev_time": stddev_time,
            "avg_sources": avg_sources,
            "avg_response_length": avg_response_length,
            "runs": question_results
        }

        results.append(question_result)

        logger.info(f"  Success rate: {success_rate:.2f}, Avg time: {avg_time:.2f}s")

    # Calculate overall statistics
    end_time_total = time.time()
    total_time = end_time_total - start_time_total

    overall_success_rate = sum(r["success_rate"] for r in results) / len(results) if results else 0
    successful_times = [r["avg_time"] for r in results if r["avg_time"] > 0]
    overall_avg_time = statistics.mean(successful_times) if successful_times else 0

    overall_stats = {
        "total_time": total_time,
        "num_questions": len(questions),
        "num_runs": num_runs,
        "overall_success_rate": overall_success_rate,
        "overall_avg_time": overall_avg_time
    }

    benchmark_results = {
        "overall": overall_stats,
        "questions": results
    }

    return benchmark_results

def load_questions(file_path: str) -> List[str]:
    """Load benchmark questions from a file

    Args:
        file_path: Path to file containing questions

    Returns:
        List of questions
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            # If it's a list of strings
            if all(isinstance(item, str) for item in data):
                return data
            # If it's a list of dictionaries with a "question" field
            elif all(isinstance(item, dict) and "question" in item for item in data):
                return [item["question"] for item in data]

        # If it's a dictionary with a "questions" field
        elif isinstance(data, dict) and "questions" in data and isinstance(data["questions"], list):
            if all(isinstance(item, str) for item in data["questions"]):
                return data["questions"]
            elif all(isinstance(item, dict) and "question" in item for item in data["questions"]):
                return [item["question"] for item in data["questions"]]

        raise ValueError("Unsupported questions format")
    except Exception as e:
        logger.error(f"Error loading questions: {str(e)}")
        raise

async def main():
    """Main function to run benchmarks"""
    # Load environment variables
    load_dotenv()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Benchmark the AI agent")
    parser.add_argument("--questions", help="JSON file with questions to benchmark")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per question")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout in seconds for each run")
    parser.add_argument("--output", help="Output file for benchmark results")

    args = parser.parse_args()

    try:
        # Load questions
        if args.questions:
            questions = load_questions(args.questions)
        else:
            # Default questions
            questions = [
                "What is artificial intelligence?",
                "How does a neural network work?",
                "What are the latest developments in NLP?",
                "Tell me about quantum computing.",
                "What is the weather like today?"
            ]

        logger.info(f"Benchmarking {len(questions)} questions, {args.runs} runs each...")
        results = await benchmark_agent(questions, args.runs, args.timeout)

        # Print summary
        logger.info("\nBenchmark Results:")
        logger.info(f"Total time: {results['overall']['total_time']:.2f}s")
        logger.info(f"Overall success rate: {results['overall']['overall_success_rate']:.2f}")
        logger.info(f"Overall average time: {results['overall']['overall_avg_time']:.2f}s")

        logger.info("\nQuestion Results:")
        for result in results["questions"]:
            logger.info(f"Question: {result['question']}")
            logger.info(f"  Success Rate: {result['success_rate']:.2f}")
            logger.info(f"  Avg Time: {result['avg_time']:.2f}s")
            logger.info(f"  Avg Sources: {result['avg_sources']:.1f}")

        # Save results
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"\nDetailed results saved to {args.output}")

    except Exception as e:
        logger.error(f"Error running benchmark: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
