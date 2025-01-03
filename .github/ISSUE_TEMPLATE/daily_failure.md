---
title: "Daily Workflow Failure"
labels: bug
---

Comparing VM execution against Native produced diffs:

- Commit: {{ env.COMMIT_SHA }}
- Workflow URL: {{ env.WORKFLOW_URL }}

## Compare Output

The transaction were not compared in order. You should rerun the whole failing block to find the error root

```
{{ env.OUTPUT }}
```
