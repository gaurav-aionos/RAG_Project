"""
Main runner script for StudyMate AI application.
This script serves as the entry point to run the Streamlit app.
"""

import sys
import os
import subprocess

def main():
    """Main function to run the StudyMate AI application."""
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the frontend app
    app_path = os.path.join(current_dir, "frontend", "app.py")
    
    # Check if the app file exists
    if not os.path.exists(app_path):
        print(f"Error: App file not found at {app_path}")
        sys.exit(1)
    
    # Run the Streamlit app
    try:
        print("Starting StudyMate AI...")
        print(f"Running: streamlit run {app_path}")
        subprocess.run(["streamlit", "run", app_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running the application: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
    except FileNotFoundError:
        print("Error: Streamlit not found. Please install streamlit using: pip install streamlit")
        sys.exit(1)


if __name__ == "__main__":
    main()
