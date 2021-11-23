# Project INF8215TP1 - Intelligent agent  for Quoridor game

## Quoridor AI based on Alpha Beta Prunning


### Requirement
*  File: my_player.py<br>
  
*  Imports (already in file):<br>
  ```python
    from quoridor import *
    import random
    import timeit
    import math
  ```
*  Tested on: Python 3.8.1
### How to run (Different terminals)
1.  Run our AI Agent command
  ```
    python my_player.py --bind localhost --port 8000
  ```
2.  Run your AI Agent command  
  ```
    python greedy_player.py --bind localhost --port 8080
  ```
3.  Run Game command
  ```
    python game.py http://localhost:8000 http://localhost:8080 --time 300
  ```
