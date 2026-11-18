# Data

Raw data is not committed. Point the code at it with `PATHSNAP_DATA`:

```
<PATHSNAP_DATA>/
  train/<trip_id>/<trip_id>.{track,nodes,arcs,route}
  student_test/<trip_id>/...
  evaluator_test/<trip_id>/...   # phase 2, API laptop only
```

Default (if unset): `../Student folder` relative to the repo.
