import os
import atexit
import logging
from langfuse import Langfuse
from typing import Optional

logger = logging.getLogger(__name__)

# Global Langfuse client instance
_langfuse_client: Optional[Langfuse] = None

def initialize_langfuse_client():
    """
    Initializes the Langfuse client from environment variables.
    Registers a shutdown hook to flush traces on application exit.
    This function should be called once at the application's startup.
    """
    global _langfuse_client

    if _langfuse_client is not None:
        logger.info("Langfuse client already initialized. Skipping re-initialization.")
        return

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com") # Default to US cloud

    if not public_key or not secret_key:
        logger.warning(
            "Langfuse API keys (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY) "
            "not found in environment variables. Langfuse will be disabled."
        )
        return

    try:
        _langfuse_client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            # Optional: Configure batching/flush intervals for production
            # flush_interval_seconds=5,
            # max_batch_size=100,
        )
        logger.info("Langfuse client initialized successfully, host: %s", host)
        # Register the shutdown hook
        atexit.register(_shutdown_langfuse_client)
    except Exception:
        logger.error("Failed to initialize Langfuse client. Langfuse will be disabled.", exc_info=True)
        _langfuse_client = None # Ensure it's None if initialization fails

def _shutdown_langfuse_client():
    """
    Internal function to gracefully shut down the Langfuse client and flush traces.
    Registered to be called automatically on program exit.
    """
    global _langfuse_client
    if _langfuse_client:
        logger.info("Shutting down Langfuse client and flushing remaining traces...")
        try:
            _langfuse_client.flush()
            # Removed: _langfuse_client.wait_for_flush()
            logger.info("Langfuse traces flush initiated successfully.")
        except Exception:
            logger.error("Failed to flush Langfuse traces during shutdown.", exc_info=True)
        finally:
            _langfuse_client = None # Clear the client

def get_langfuse_client_instance() -> Optional[Langfuse]:
    """
    Returns the globally initialized Langfuse client instance.
    If the client has not been initialized or initialization failed, returns None.
    """
    return _langfuse_client