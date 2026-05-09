"""
Convenience launcher. Run from the project root:

    python run.py              # detection loop with display
    python run.py --headless   # no display window
    python run.py --dashboard  # Flask web dashboard on :5000
    python run.py --test       # run pytest suite
"""

import sys


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""

    if mode == "--test":
        import pytest
        sys.exit(pytest.main(["-v", "tests/"]))

    elif mode == "--dashboard":
        from dashboard.app import app
        import config
        print(f"[Run] Dashboard → http://localhost:{config.FLASK_PORT}")
        app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=False, threaded=True)

    else:
        headless = "--headless" in sys.argv
        from detection.detector import run
        run(headless=headless)


if __name__ == "__main__":
    main()
