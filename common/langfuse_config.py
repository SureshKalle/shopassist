import os
import atexit
from langfuse import Langfuse, get_client
from typing import Optional

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
        print("Langfuse client already initialized. Skipping re-initialization.")
        return

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com") # Default to US cloud

    if not public_key or not secret_key:
        print(
            "WARNING: Langfuse API keys (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY) "
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
        print(f"Langfuse client initialized successfully, host: {host}")
        # Register the shutdown hook
        atexit.register(_shutdown_langfuse_client)
    except Exception as e:
        print(f"ERROR: Failed to initialize Langfuse client: {e}. Langfuse will be disabled.")
        _langfuse_client = None # Ensure it's None if initialization fails

def _shutdown_langfuse_client():
    """
    Internal function to gracefully shut down the Langfuse client and flush traces.
    Registered to be called automatically on program exit.
    """
    global _langfuse_client
    if _langfuse_client:
        print("Shutting down Langfuse client and flushing remaining traces...")
        try:
            _langfuse_client.flush()
            # Removed: _langfuse_client.wait_for_flush()
            print("Langfuse traces flush initiated successfully.")
        except Exception as e:
            print(f"ERROR: Failed to flush Langfuse traces during shutdown: {e}")
        finally:
            _langfuse_client = None # Clear the client

def get_langfuse_client_instance() -> Optional[Langfuse]:
    """
    Returns the globally initialized Langfuse client instance.
    If the client has not been initialized or initialization failed, returns None.
    """
    return _langfuse_client

# Clear out any legacy environment pollution (good for testing)
os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
os.environ.pop("LANGFUSE_SECRET_KEY", None)
os.environ.pop("LANGFUSE_HOST", None)

# Example: Set mock credentials for initial testing if not already set (REMOVE FOR PRODUCTION)
if not os.getenv("LANGFUSE_PUBLIC_KEY"):
    os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-94c306c1-3e72-4b21-89d5-664a4b6a8ee5"
if not os.getenv("LANGFUSE_SECRET_KEY"):
    os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-288de4a5-7483-425e-af54-e24bca344645"
if not os.getenv("LANGFUSE_HOST"):
    os.environ["LANGFUSE_HOST"] = "https://us.cloud.langfuse.com"