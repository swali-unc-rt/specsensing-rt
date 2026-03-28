# specsensing-rt

In one terminal:
```
cd specsensing-rt/python
sudo ../../liblitmus/setsched GSN-EDF
sudo ../../feather-trace-tools/st-trace-schedule test1
```

In another terminal:
```
docker start ssrt1
docker attach ssrt1
cd /workspace/specsensing-rt
./bin/specsensing-rt
```

Once the system is cooling down, it won't close cleanly (in progress), but go go the first terminal for feather-trace, and hit ENTER.
On that same terminal, you can
```
python just_stats.py
```