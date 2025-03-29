from dotenv import load_dotenv
import pathlib
import sys
import os
import inspect
import traceback

# Print diagnostic information
print("=== Diagnostic MCP Agent ===")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

try:
    print("Importing dependencies...")
    from pydantic_ai import Agent
    from openai import AsyncOpenAI, OpenAI
    from pydantic_ai.models.openai import OpenAIModel

    print("Successfully imported pydantic_ai and OpenAI")

    # Check OpenAIModel constructor
    print("\nChecking OpenAIModel constructor:")
    sig = inspect.signature(OpenAIModel.__init__)
    params = list(sig.parameters.keys())

    # Skip 'self' parameter
    if 'self' in params:
        params.remove('self')

    print(f"OpenAIModel accepts these parameters: {params}")

    # Try to create a model with different argument patterns
    print("\nTrying different initialization methods:")

    # Try positional argument
    try:
        model = OpenAIModel("gpt-4o-mini")
        print("✓ Created model with positional argument")
    except Exception as e:
        print(f"✗ Failed with positional argument: {e}")

    # Try named arguments
    try:
        model = OpenAIModel(model_name="gpt-4o-mini")
        print("✓ Created model with model_name=")
    except Exception as e:
        print(f"✗ Failed with model_name=: {e}")

    try:
        model = OpenAIModel(model="gpt-4o-mini")
        print("✓ Created model with model=")
    except Exception as e:
        print(f"✗ Failed with model=: {e}")

    print("\nDiagnostic complete. Check the output above to determine the correct initialization method.")

except Exception as e:
    print(f"\nCritical error: {type(e).__name__}: {str(e)}")
    print("\nDetailed traceback:")
    traceback.print_exc()
