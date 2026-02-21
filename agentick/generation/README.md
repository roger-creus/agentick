# Generation

Procedural generation utilities used by task `generate()` methods to build grids, mazes, rooms, and key-door sequences.

## Modules

### maze.py -- Maze generation algorithms

- `MazeGenerator` -- configurable generator that dispatches to one of five algorithms and applies post-processing (loop addition, dead-end removal).
- `MazeConfig` -- dataclass controlling algorithm choice, corridor width, dead-end density, loop frequency, and wall thickness.
- `recursive_backtracker(w, h, rng)` -- DFS-based; long winding corridors, few branches.
- `prims_maze(w, h, rng)` -- Prim's algorithm; many short corridors, frequent branching.
- `kruskals_maze(w, h, rng)` -- Kruskal's algorithm; small trees that gradually merge.
- `binary_tree_maze(w, h, rng)` -- fast but biased toward north/east.
- `recursive_division(w, h, rng)` -- starts open, subdivides with walls and passages.

All functions return a 2D `numpy` array of `CellType.WALL` / `CellType.EMPTY` values.

### room.py -- Room-based layouts

- `RoomGenerator` -- generates room layouts via BSP or random placement with L-shaped corridor connections.
- `Room` -- dataclass with position, size, `center()`, and `intersects()`.
- `bsp_rooms(w, h, rng)` -- Binary Space Partitioning; guarantees connected rooms.
- `random_rooms_with_corridors(w, h, rng)` -- random non-overlapping rooms linked sequentially.
- `place_key_door_sequence(grid, rooms, num_keys, rng)` -- places colored keys and doors across rooms with proper reachability ordering.

### validation.py -- Solvability checks and optimal paths

- `SolvabilityValidator` -- BFS-based validator supporting simple navigation, key-door, box-pushing (Sokoban), switch, and inventory mechanics.
- `ValidationResult` -- dataclass with `solvable`, `optimal_path`, `optimal_length`, `reason`.
- `verify_solvable(grid, start, goals, config)` -- quick boolean solvability check.
- `find_optimal_path(grid, start, goals, config)` -- returns `(path, length)` or `(None, 0)`.
- `compute_solution_stats(grid, start, goals, config)` -- returns dict with `optimal_length`, `branching_factor`, `turns_in_path`, `path_straightness`.

### difficulty.py -- Difficulty estimation and calibration

- `DifficultyEstimator` -- computes a weighted 0-1 difficulty score from solution length, branching factor, subgoal count, wall density, and path turns.
- `DifficultyMetrics` -- dataclass holding all metric components.
- `estimate_difficulty(grid, start, goals, config)` -- quick scalar estimate.
- `calibrate_difficulty(generator_fn, target, tolerance, max_attempts)` -- repeatedly generates levels until one matches the target difficulty.

### seed.py -- Deterministic seeding

- `SeedManager` -- manages `numpy.random.SeedSequence` for reproducible multi-stage generation. Holds fixed benchmark seed tables per task and difficulty.
- `get_benchmark_seed(task_id, difficulty, instance_idx)` -- returns a specific benchmark seed.
- `create_seed_sequence(base_seed, num_stages)` -- spawns independent seeds for multi-stage generators.
