---
title: "Daily Workflow Failure: Block Range {{ env.BLOCK_START }}-{{ env.BLOCK_END }}"
labels: bug
---

Comparing VM execution against Native in the given block range produced diffs:

- Workflow URL: {{ env.WORKFLOW_URL }}
- Block Start: {{ env.BLOCK_START }}
- Block End: {{ env.BLOCK_END }}

## Compare Output

```
{{ env.OUTPUT }}
```
