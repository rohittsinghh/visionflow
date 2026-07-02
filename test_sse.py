"""
Small SSE client for testing detection events.
"""

import json
import urllib.request


def main():
    url = "http://127.0.0.1:8000/detections/events"

    with urllib.request.urlopen(url) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()

            if not line.startswith("data: "):
                continue

            payload = json.loads(line.removeprefix("data: "))
            print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
