import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional
from collections import defaultdict

import requests
from dynamo.runtime import DistributedRuntime
from dynamo.llm import BlockManagerClient


# Common sampling parameters
DEFAULT_SAMPLING_PARAMS = {
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "max_tokens": 32,
    "temperature": 0.0,
    "top_p": 0.001
}

# Common request configuration
DEFAULT_REQUEST_CONFIG = {
    "url": "http://localhost:8000/v1/chat/completions",
    "headers": {
        "Content-Type": "application/json"
    },
    "timeout": 30
}

# Global test cases
TEST_CASES = [
    {
        "name": "Test 1 - Lorem ipsum",
        "content": "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
        "sampling_params": {"top_p": 0.0001}
    },
    {
        "name": "Test 2 - Lorem ipsum with prefix",
        "content": "2 Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
        "sampling_params": {"top_p": 0.001}
    },
    {
        "name": "Test 3 - Crab + Snake Story",
        "content": "Beneath the coral reef's embrace, a serpent met Crab with claws aglow. Their passion sparked like lightning while she hissed demands, he'd snap for more. \"Your shell's too hard!\" the snake cried, \"Your fangs too sharp!\" the crab barked, yet neither fled. They'd wrestle in sandy beds, stubborn, full. The ocean trembled at fights, then sighed at embrace, two creatures bound by fierce",
        "sampling_params": {"top_p": 0.001}
    }
]

# Keep only TEST_CASES as global - no global DRT, client, or loop variables


def create_message(content: str, role: str = "user") -> Dict[str, str]:
    """Create a message dictionary for the chat completion request."""
    return {"role": role, "content": content}


def make_request(
    content: str,
    sampling_params: Optional[Dict[str, Any]] = None,
    request_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Make a chat completion request with common sampling parameters.

    Args:
        content: The message content to send
        sampling_params: Override default sampling parameters
        request_config: Override default request configuration

    Returns:
        Dictionary containing response data and metadata
    """
    # Merge parameters with defaults
    final_sampling_params = {**DEFAULT_SAMPLING_PARAMS}
    if sampling_params:
        final_sampling_params.update(sampling_params)

    final_request_config = {**DEFAULT_REQUEST_CONFIG}
    if request_config:
        final_request_config.update(request_config)

    # Create the request payload
    payload = {
        **final_sampling_params,
        "messages": [create_message(content)]
    }

    # Make the request
    start_time = time.time()
    try:
        response = requests.post(
            final_request_config["url"],
            headers=final_request_config["headers"],
            json=payload,
            timeout=final_request_config["timeout"]
        )
        response.raise_for_status()

        end_time = time.time()

        return {
            "success": True,
            "response": response.json(),
            "status_code": response.status_code,
            "duration": end_time - start_time,
            "request_payload": payload
        }
    except requests.exceptions.RequestException as e:
        end_time = time.time()
        return {
            "success": False,
            "error": str(e),
            "status_code": getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None,
            "duration": end_time - start_time,
            "request_payload": payload
        }


def issue_request_by_index(index: int) -> Dict[str, Any]:
    """
    Issue a request based on the test case index.

    Args:
        index: Index of the test case (0-based)

    Returns:
        Dictionary containing response data and metadata
    """
    if index < 0 or index >= len(TEST_CASES):
        return {
            "success": False,
            "error": f"Invalid test case index: {index}. Available indices: 0-{len(TEST_CASES)-1}",
            "status_code": None,
            "duration": 0,
            "request_payload": None
        }

    test_case = TEST_CASES[index]
    print(f"\n--- {test_case['name']} (Index: {index}) ---")

    result = make_request(
        content=test_case["content"],
        sampling_params=test_case.get("sampling_params", {})
    )

    # Print result summary
    if result["success"]:
        print(f"✓ Success (Status: {result['status_code']}, Duration: {result['duration']:.2f}s)")
        response_data = result["response"]
        if "choices" in response_data and response_data["choices"]:
            content = response_data["choices"][0].get("message", {}).get("content", "")
            print(f"  Response: {content[:100]}{'...' if len(content) > 100 else ''}")
    else:
        print(f"✗ Failed: {result['error']}")

    return result



def setup_dynamo_client():
    """Set up the Dynamo runtime and block manager client."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Note: The linter has issues with the exact signature, but this is the intended usage
        # Based on the bindings, the constructor should work with these parameters
        drt = None
        client = None

        # Try to create the distributed runtime
        try:
            # This might work at runtime even if linter complains
            drt = DistributedRuntime(loop, False)  # type: ignore
            ns = drt.namespace("test")  # type: ignore
            cp = ns.component("kvbm")
            instance_id = 7587887961983545002
            client = BlockManagerClient(cp, instance_id)
        except Exception as init_error:
            print(f"Dynamo client initialization failed: {init_error}")
            print("Continuing with HTTP request testing only...")

        return drt, client, loop
    except Exception as e:
        print(f"Error setting up Dynamo client: {e}")
        return None, None, None


@dataclass
class UnaryResponse:
    response: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_hash: str


class ResponseTracker:
    """
    Tracks response hashes and their counts per request index.

    Structure: Dict[int, Dict[str, int]]
    - Outer key: request index
    - Inner key: response hash
    - Inner value: count of how many times this hash occurred for this request index
    """

    def __init__(self):
        # Using defaultdict to automatically create inner dicts
        self.responses: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def record_response(self, request_index: int, response_hash: str) -> None:
        """
        Record a response hash for a given request index.

        Args:
            request_index: The index of the test case
            response_hash: The SHA256 hash of the response
        """
        self.responses[request_index][response_hash] += 1

    def record_unary_response(self, request_index: int, unary_response: UnaryResponse) -> None:
        """
        Record a UnaryResponse for a given request index.

        Args:
            request_index: The index of the test case
            unary_response: The UnaryResponse object containing the hash
        """
        if unary_response:
            self.record_response(request_index, unary_response.response_hash)

    def get_count(self, request_index: int, response_hash: str) -> int:
        """
        Get the count for a specific request index and response hash.

        Args:
            request_index: The index of the test case
            response_hash: The SHA256 hash of the response

        Returns:
            Count of occurrences (0 if not found)
        """
        return self.responses[request_index][response_hash]

    def get_all_hashes_for_index(self, request_index: int) -> Dict[str, int]:
        """
        Get all response hashes and their counts for a given request index.

        Args:
            request_index: The index of the test case

        Returns:
            Dictionary mapping response hashes to their counts
        """
        return dict(self.responses[request_index])

    def get_total_responses_for_index(self, request_index: int) -> int:
        """
        Get the total number of responses recorded for a given request index.

        Args:
            request_index: The index of the test case

        Returns:
            Total count of all responses for this index
        """
        return sum(self.responses[request_index].values())

    def get_unique_hashes_for_index(self, request_index: int) -> int:
        """
        Get the number of unique response hashes for a given request index.

        Args:
            request_index: The index of the test case

        Returns:
            Number of unique response hashes
        """
        return len(self.responses[request_index])

    def print_summary(self) -> None:
        """Print a summary of all tracked responses."""
        print("\n=== Response Tracking Summary ===")
        for request_index in sorted(self.responses.keys()):
            hashes = self.responses[request_index]
            total_responses = sum(hashes.values())
            unique_hashes = len(hashes)

            print(f"\nRequest Index {request_index}:")
            print(f"  Total Responses: {total_responses}")
            print(f"  Unique Hashes: {unique_hashes}")

            for hash_value, count in sorted(hashes.items()):
                hash_short = hash_value[:16] + "..." if len(hash_value) > 16 else hash_value
                print(f"    {hash_short}: {count} times")

    def to_dict(self) -> Dict[int, Dict[str, int]]:
        """
        Convert the tracking data to a regular dictionary.

        Returns:
            Dictionary representation of the tracking data
        """
        return {
            request_index: dict(hashes)
            for request_index, hashes in self.responses.items()
        }


def parse_response(result: Dict[str, Any]) -> UnaryResponse:
    """
    Parse the response from make_request and extract usage data.

    Args:
        result: Response dictionary from make_request

    Returns:
        UnaryResponse object with extracted data

    Raises:
        ValueError: If the response is unsuccessful or missing required data
    """
    if not result.get("success", False):
        error_msg = result.get("error", "Unknown error")
        raise ValueError(f"Request failed: {error_msg}")

    response_data = result.get("response", {})

    # Extract response text
    choices = response_data.get("choices", [])
    if not choices:
        raise ValueError("Response missing choices data")

    message = choices[0].get("message", {})
    response_text = message.get("content", "")

    # Extract usage data
    usage = response_data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)

    # Compute SHA256 hash of response
    response_hash = hashlib.sha256(response_text.encode('utf-8')).hexdigest()

    return UnaryResponse(
        response=response_text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        response_hash=response_hash
    )


def issue_request(idx: int) -> Dict[str, Any]:
    """
    Issue a request based on the test case index.

    Args:
        idx: Index of the test case (0-based)

    Returns:
        Dictionary containing response data and metadata

    Raises:
        IndexError: If the index is out of range
    """
    if idx < 0 or idx >= len(TEST_CASES):
        raise IndexError(f"Test case index {idx} is out of range. Available indices: 0-{len(TEST_CASES)-1}")

    test_case = TEST_CASES[idx]
    # print(f"\n--- {test_case['name']} (Index: {idx}) ---")

    result = make_request(
        content=test_case["content"],
        sampling_params=test_case.get("sampling_params", {})
    )

    return result


def execute_request(idx: int) -> UnaryResponse | None:
    """
    Execute a request by index and parse the response.

    Args:
        idx: Index of the test case (0-based)

    Returns:
        UnaryResponse object with parsed data

    Raises:
        IndexError: If the index is out of range
        ValueError: If the request fails or response parsing fails
        Exception: For other unexpected errors
    """
    try:
        # Issue the request
        result = issue_request(idx)

        # Parse the response
        unary_response = parse_response(result)

        # Print success summary
        # print(f"✓ Success (Status: {result['status_code']}, Duration: {result['duration']:.2f}s)")
        # print(f"  Prompt Tokens: {unary_response.prompt_tokens}")
        # print(f"  Completion Tokens: {unary_response.completion_tokens}")
        # print(f"  Total Tokens: {unary_response.total_tokens}")
        # print(f"  Response Hash: {unary_response.response_hash}")
        # print(f"  Response: {unary_response.response[:100]}{'...' if len(unary_response.response) > 100 else ''}")

        return unary_response

    except (IndexError, ValueError) as e:
        print(f"✗ Request failed: {e}")
        return None
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return None

def reset_all_pools(controller):
    while True:
        try:
            controller.reset_all_pools()
            break
        except Exception as e:
            print(f"✗ Unexpected error: {e}")

def reset_pool(controller, pool):
    while True:
        try:
            controller.reset_pool(pool)
            break
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            time.sleep(0.2)

def run_experiment_1(controller, control, request_id, repeat=5):
    """
    Issue the same request `repeat` times, and record the results
    and compare the results with the control group
    """
    print(f"\n\n=== Experiment 1: request_id: {request_id} ===")
    reset_all_pools(controller)
    tracker = ResponseTracker()
    for i in range(repeat):
        output = execute_request(request_id)
        if output:
            tracker.record_unary_response(request_id, output)

    tracker.print_summary()

    # hashes = tracker.get_all_hashes_for_index(request_id)
    # keys = list(hashes.keys())
    # assert len(keys) == 1
    # assert hashes[keys[0]] == repeat
    # assert keys[0] == control[request_id][0].response_hash

def run_experiment_2(controller, control, request_id, repeat=5):
    """
    Issue the same request `repeat` times, but in between each request,
    reset the device pool.
    """
    print(f"\n\n=== Experiment 2: request_id: {request_id} ===")
    reset_all_pools(controller)
    tracker = ResponseTracker()
    for i in range(repeat):
        output = execute_request(request_id)
        if output:
            tracker.record_unary_response(request_id, output)
        reset_pool(controller, "g1")

    tracker.print_summary()

    # hashes = tracker.get_all_hashes_for_index(request_id)
    # keys = list(hashes.keys())
    # assert len(keys) == 1
    # assert hashes[keys[0]] == repeat
    # assert keys[0] == control[request_id][0].response_hash

def main():
    """Main function to run all tests."""
    print("=== Dynamo Test Suite ===")

    # Set up Dynamo client
    drt, controller, _loop = setup_dynamo_client()

    if not controller:
        print("Controller not available, running requests only...")
        return

    controller.reset_all_pools()

    # Initialize response tracker
    control = []

    for i in range(len(TEST_CASES)):
        output = execute_request(i)
        status = controller.status("g1")
        # the pool is empty, the request is completed
        # the inactive blocks are the tail -> root of the last request sequence
        active_blocks = list(reversed(status.inactive_blocks()))
        print(active_blocks)
        control.append((output, active_blocks))
        time.sleep(2)
        controller.reset_all_pools()

    for i in range(len(TEST_CASES)):
        run_experiment_1(controller, control, i)

    for i in range(len(TEST_CASES)):
        run_experiment_2(controller, control, i)

    # Clean up
    if drt:
        drt.shutdown()


if __name__ == "__main__":
    main()

