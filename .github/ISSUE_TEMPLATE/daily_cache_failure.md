---
title: "CACHE MISS - Daily Workflow Failure"
labels: bug
---

Daily workflow failed to retrieve the blocks to execute from the cache. To save the blocks in the cache,
`cache-vm-daily` workflow should be run manually with the desired blocks. 

Input example: "740000,760000,780000,800000"

- Workflow URL: {{ env.WORKFLOW_URL }}
