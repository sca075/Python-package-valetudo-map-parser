# Performance Profiling for Valetudo Map Parser

This directory contains enhanced test files with comprehensive profiling capabilities for analyzing CPU and memory usage in the Valetudo Map Parser library.

## 🎯 Profiling Features

### Memory Profiling
- **Real-time memory tracking** using `tracemalloc` and `psutil`
- **Memory snapshots** at key points during image generation
- **Memory growth analysis** showing peak usage and leaks
- **Top memory allocations** comparison between snapshots

### CPU Profiling
- **Function-level timing** using `cProfile`
- **Line-by-line profiling** capabilities (with optional dependencies)
- **Operation timing** for specific image generation phases
- **Cumulative time analysis** for bottleneck identification

### System Profiling
- **Garbage collection statistics**
- **Process memory usage** (RSS, VMS, percentage)
- **Timing patterns** across multiple operations

## 📋 Setup

### Install Profiling Dependencies
```bash
pip install -r tests/profiling_requirements.txt
```

### Optional Advanced Profiling
For line-by-line CPU profiling (requires compilation):
```bash
pip install line-profiler
```

## 🚀 Usage

### Rand256 Vacuum Profiling
```bash
cd tests
python test_rand.py
```

### Hypfer Vacuum Profiling
```bash
cd tests
python test_hypfer_profiling.py
```

## 📊 Output Analysis

### Memory Report Example
```
🔍 Memory Usage Timeline:
  1. Test Setup Start              | RSS:   45.2MB | VMS:  234.1MB |  2.1%
  2. Before Image Gen - file1.bin  | RSS:   52.3MB | VMS:  241.2MB |  2.4%
  3. After Image Gen - file1.bin   | RSS:   48.1MB | VMS:  238.9MB |  2.2%

📈 Memory Growth Analysis:
   Start RSS: 45.2MB
   Peak RSS:  52.3MB (+7.1MB)
   End RSS:   48.1MB (+2.9MB from start)
```

### CPU Report Example
```
⚡ CPU USAGE ANALYSIS
Top 15 functions by cumulative time:
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      6    0.000    0.000    2.103    0.351 auto_crop.py:391(async_auto_trim_and_zoom_image)
     12    0.262    0.022    0.262    0.022 {built-in method scipy.ndimage._nd_image.find_objects}
```

### Timing Analysis Example
```
⏱️  TIMING ANALYSIS
📊 Timing Summary by Operation:
   Image                | Avg:  2247.3ms | Min:  2201.1ms | Max:  2289.5ms | Count: 6
   Generation           | Avg:  2247.3ms | Min:  2201.1ms | Max:  2289.5ms | Count: 6
```

## 🎯 Optimization Targets

The profiling will help identify:

### High-Impact Optimization Opportunities
1. **Memory hotspots** - Functions allocating the most memory
2. **CPU bottlenecks** - Functions consuming the most time
3. **Memory leaks** - Objects not being properly freed
4. **Inefficient algorithms** - Functions with high per-call costs

### Key Metrics to Monitor
- **Peak memory usage** during image generation
- **Memory growth patterns** across multiple images
- **Function call frequency** and cumulative time
- **Array allocation patterns** in NumPy operations

## 🔧 Customization

### Enable/Disable Profiling
```python
# Disable profiling for faster execution
test = TestRandImageHandler(enable_profiling=False)

# Enable only memory profiling
profiler = PerformanceProfiler(
    enable_memory_profiling=True,
    enable_cpu_profiling=False
)
```

### Add Custom Profiling Points
```python
# In your test code
if self.profiler:
    self.profiler.take_memory_snapshot("Custom Checkpoint")
    cpu_profiler = self.profiler.start_cpu_profile("Custom Operation")
    
    # Your code here
    
    self.profiler.stop_cpu_profile(cpu_profiler)
    self.profiler.time_operation("Custom Operation", start_time, end_time)
```

## 📈 Performance Baseline

Use these tests to establish performance baselines and track improvements:

1. **Run tests before optimization** to establish baseline
2. **Implement optimizations** in the library code
3. **Run tests after optimization** to measure improvements
4. **Compare reports** to validate performance gains

## 🚨 Important Notes

- **Memory profiling** adds ~5-10% overhead
- **CPU profiling** adds ~10-20% overhead  
- **Line profiling** (if enabled) adds ~50-100% overhead
- **Disable profiling** for production performance testing

## 📝 Profiling Data Files

The tests generate several output files:
- `profile_output_rand.prof` - cProfile data for Rand256 tests
- `profile_output_hypfer.prof` - cProfile data for Hypfer tests

These can be analyzed with tools like `snakeviz`:
```bash
pip install snakeviz
snakeviz profile_output_rand.prof
```
