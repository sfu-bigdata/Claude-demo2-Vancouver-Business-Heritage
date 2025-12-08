# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Machine learning project to predict whether a Vancouver business is located within 1km of a heritage site, using City of Vancouver Open Data.

## Project Type

Python project using scikit-learn for ML modeling.

## GitHub Issue Workflow

**IMPORTANT**: When working on tasks tracked in GitHub issues:

1. **Before starting**: Check the GitHub issue for task details using `gh issue view <number>`
2. **During work**:
   - Reference the issue number in commits
   - **Commit often** at every checkpoint (after completing subtasks, creating files, fixing bugs)
   - **Push code frequently** to keep remote repository up to date
3. **Upon completion**: Automatically close the issue with a summary using:
   ```bash
   gh issue close <number> --comment "Task completed.

   ## Completed Deliverables
   - ✅ [list what was delivered]

   [relevant details or next steps]"
   ```

## Project Structure

- `data/raw/` - Original datasets from Vancouver Open Data
- `data/processed/` - Cleaned datasets for ML
- `PROJECT_PLAN.md` - 5-task project plan
- GitHub issues #2-#6 track Tasks 1-5

## Data Sources

- Business Licences: ~132K records with coordinates
- Heritage Sites: ~2.5K heritage locations with coordinates

## Development Setup

### Virtual Environment

**IMPORTANT**: Always use a virtual environment to avoid polluting the global Python environment.

**Setup**:
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

**Deactivate** when done:
```bash
deactivate
```

The `venv/` directory is already in `.gitignore` and should never be committed.
