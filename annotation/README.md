# Label Studio Annotation Setup

## Quick Start

1. **Start the service:**
   ```bash
   make annotation-up
   ```

2. **Open the UI:**
   Navigate to `http://localhost:8080` in your browser.

3. **Create a project:**
   - Click "Create New Project"
   - Under "Labeling Setup", paste the contents of `label_studio_config.xml`
   - Save the project

4. **Import tasks:**
   - Go to "Data" → "Import"
   - Upload `sample_tasks.json` (or your own task JSON)
   - Tasks with candidate predictions will show pre-annotated suggestions

## What's Included

| File | Purpose |
|------|---------|
| `label_studio_config.xml` | Shared labeling configuration (version-controlled) |
| `sample_tasks.json` | Example task data with candidate predictions |
| `sample_predictions.json` | Example completed annotation export |
| `docker-compose.annotation.yml` | Local Docker Compose deployment |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ANNOTATION_PORT` | `8080` | Local port for the Label Studio UI |
| `ANNOTATION_VIDEO_DIR` | `./data/videos` | Read-only mounted video directory |

## Important Notes

- **Never commit source videos** — they are mounted read-only
- **Never commit Label Studio database** — stored in Docker volume
- **Candidate suggestions are editable** — annotators can correct, delete, or supplement them
- **The complete active span must always be reviewed** — the confirmation checkbox is required for export
