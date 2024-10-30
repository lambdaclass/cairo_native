---
title: "Daily Workflow Failure: Block Range {{ env.BLOCK_START }}-{{ env.BLOCK_END }}"
labels: bug
---

Comparing VM execution against Native in the given block range produced diffs:

- Commit: {{ env.COMMIT_SHA }}
- Block Start: {{ env.BLOCK_START }}
- Block End: {{ env.BLOCK_END }}
- Workflow URL: {{ env.WORKFLOW_URL }}

## Compare Output

```
{{ env.OUTPUT }}
```
