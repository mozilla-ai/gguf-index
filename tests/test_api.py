"""Tests for HuggingFace API wrapper."""

import threading
from unittest.mock import MagicMock, patch

import pytest

from gguf_index.api import HuggingFaceAPI


class TestRequestCounter:
    """Tests for HTTP request counting."""

    def test_initial_count_is_zero(self):
        """Request count should start at zero."""
        api = HuggingFaceAPI()
        assert api.request_count == 0

    def test_reset_request_count(self):
        """reset_request_count should set count to zero."""
        api = HuggingFaceAPI()
        # Manually set the counter to simulate requests
        api._request_count = 10
        api.reset_request_count()
        assert api.request_count == 0

    def test_request_count_is_thread_safe(self):
        """Request count should be thread-safe."""
        api = HuggingFaceAPI()
        num_threads = 10
        increments_per_thread = 100

        def increment_counter():
            for _ in range(increments_per_thread):
                with api._request_count_lock:
                    api._request_count += 1

        threads = [threading.Thread(target=increment_counter) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert api.request_count == num_threads * increments_per_thread

    @patch("gguf_index.api.HuggingFaceAPI._get_http_client")
    def test_rate_limited_request_increments_count(self, mock_get_client):
        """_rate_limited_request should increment count on success."""
        api = HuggingFaceAPI(requests_per_second=0)  # Disable rate limiting

        # Mock HTTP client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.get.return_value = mock_response
        mock_get_client.return_value = mock_client

        # Make a request
        api._rate_limited_request("GET", "https://example.com/test")

        assert api.request_count == 1

    @patch("gguf_index.api.HuggingFaceAPI._get_http_client")
    def test_rate_limited_request_increments_on_each_success(self, mock_get_client):
        """Each successful request should increment the counter."""
        api = HuggingFaceAPI(requests_per_second=0)  # Disable rate limiting

        # Mock HTTP client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.get.return_value = mock_response
        mock_get_client.return_value = mock_client

        # Make multiple requests
        for _ in range(5):
            api._rate_limited_request("GET", "https://example.com/test")

        assert api.request_count == 5

    @patch("gguf_index.api.HuggingFaceAPI._get_http_client")
    def test_failed_request_does_not_increment_count(self, mock_get_client):
        """Failed requests should not increment the counter."""
        api = HuggingFaceAPI(requests_per_second=0)  # Disable rate limiting

        # Mock HTTP client to raise exception
        mock_client = MagicMock()
        mock_client.get.side_effect = Exception("Connection error")
        mock_get_client.return_value = mock_client

        # Make a request that fails
        result = api._rate_limited_request("GET", "https://example.com/test", max_retries=1)

        assert result is None
        assert api.request_count == 0

    def test_repo_info_increments_count(self):
        """repo_info wrapper should increment count."""
        api = HuggingFaceAPI()

        # Mock the underlying HfApi
        mock_result = MagicMock()
        mock_result.sha = "abc123"
        api.api.repo_info = MagicMock(return_value=mock_result)

        api.repo_info("test/repo")

        assert api.request_count == 1

    def test_get_repo_info_increments_count(self):
        """get_repo_info should increment count."""
        api = HuggingFaceAPI()

        # Mock the underlying HfApi
        mock_result = MagicMock()
        mock_result.id = "test/repo"
        mock_result.author = "test"
        mock_result.downloads = 100
        mock_result.likes = 10
        mock_result.tags = []
        api.api.model_info = MagicMock(return_value=mock_result)

        api.get_repo_info("test/repo")

        assert api.request_count == 1
