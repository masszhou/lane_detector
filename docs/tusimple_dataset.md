# 1. General Information
* [website link](https://github.com/TuSimple/tusimple-benchmark/tree/master/doc/lane_detection)
* [aws download link](https://github.com/TuSimple/tusimple-benchmark/issues/3)

# 2. Data Set Structure in this project
```
dataset
  |
  |----train_set/               # training root 
  |------|
  |------|----clips/            # video clips, 3626 clips
  |------|------|
  |------|------|----some_clip/
  |------|------|----...
  |
  |------|----label_data_0313.json      # Label data for lanes
  |------|----label_data_0531.json      # Label data for lanes
  |------|----label_data_0601.json      # Label data for lanes
  |
  |----test_set/               # testing root 
  |------|
  |------|----clips/
  |------|------|
  |------|------|----some_clip/
  |------|------|----...
  |
  |------|----test_label.json           # Test Submission Template
  |------|----test_tasks_0627.json      # Test Submission Template
```