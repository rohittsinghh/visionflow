"""
Small built-in UI pages.

The backend is still API-first, but this page is useful for manually checking
that first-appearance crop events are reaching the browser and that saved crop
files are being served correctly.
"""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse


router = APIRouter()


@router.get("/first-appearances", response_class=HTMLResponse)
def first_appearances_page():
    """
    Show first-appearance crop events with class names and images.
    """

    return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>First Appearance Crops</title>
    <style>
      :root {
        color-scheme: light;
        font-family: Arial, Helvetica, sans-serif;
        background: #f4f6f8;
        color: #18202a;
      }

      body {
        margin: 0;
      }

      header {
        align-items: center;
        background: #111827;
        color: #ffffff;
        display: flex;
        gap: 16px;
        justify-content: space-between;
        padding: 16px 24px;
      }

      h1 {
        font-size: 20px;
        margin: 0;
      }

      main {
        margin: 0 auto;
        max-width: 1200px;
        padding: 24px;
      }

      #status {
        background: #e8eef7;
        border: 1px solid #c8d3e4;
        border-radius: 6px;
        color: #27364b;
        font-size: 14px;
        padding: 10px 12px;
      }

      #grid {
        display: grid;
        gap: 16px;
        grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
        margin-top: 20px;
      }

      .crop-card {
        background: #ffffff;
        border: 1px solid #d7dde6;
        border-radius: 8px;
        overflow: hidden;
      }

      .crop-card img {
        aspect-ratio: 4 / 3;
        background: #eef2f7;
        display: block;
        object-fit: contain;
        width: 100%;
      }

      .crop-body {
        padding: 12px;
      }

      .crop-title {
        font-size: 18px;
        font-weight: 700;
        margin: 0 0 8px;
        text-transform: capitalize;
      }

      .crop-meta {
        color: #4b5563;
        font-size: 13px;
        line-height: 1.5;
        margin: 0;
      }
    </style>
  </head>
  <body>
    <header>
      <h1>First Appearance Crops</h1>
      <span id="connection">Connecting</span>
    </header>

    <main>
      <div id="status">
        Waiting for first-time classes. Start the video pipeline, then newly
        detected classes will appear here with their cropped images.
      </div>
      <section id="grid" aria-live="polite"></section>
    </main>

    <script>
      const grid = document.getElementById("grid");
      const statusBox = document.getElementById("status");
      const connection = document.getElementById("connection");
      const seenKeys = new Set();

      function imageUrl(cropUrl) {
        return new URL(cropUrl, window.location.origin).toString();
      }

      function renderCrop(eventData) {
        const key = `${eventData.run_id}:${eventData.class_name}`;

        if (seenKeys.has(key)) {
          return;
        }

        seenKeys.add(key);
        statusBox.textContent = `${seenKeys.size} first appearance crop(s) received.`;

        const card = document.createElement("article");
        card.className = "crop-card";

        const image = document.createElement("img");
        image.src = imageUrl(eventData.crop_url);
        image.alt = eventData.class_name;
        image.loading = "eager";

        const body = document.createElement("div");
        body.className = "crop-body";

        const title = document.createElement("h2");
        title.className = "crop-title";
        title.textContent = eventData.class_name;

        const meta = document.createElement("p");
        meta.className = "crop-meta";
        meta.textContent = `Frame ${eventData.frame} | Confidence ${Number(eventData.confidence).toFixed(2)} | ${eventData.run_id}`;

        body.appendChild(title);
        body.appendChild(meta);
        card.appendChild(image);
        card.appendChild(body);
        grid.prepend(card);
      }

      const events = new EventSource("/first-appearances/events");

      events.onopen = () => {
        connection.textContent = "Live";
      };

      events.onmessage = (message) => {
        renderCrop(JSON.parse(message.data));
      };

      events.onerror = () => {
        connection.textContent = "Reconnecting";
      };
    </script>
  </body>
</html>
    """
