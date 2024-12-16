# Remote-learning
Final version of the Remote Learning subject program. It takes a `.mol2` file from the `/inputs` directory as input and generates 3 output files in the `/results` directory:

- **.out1**: Prints energy contribution of each bond, angle and dihedral. Also prints the gradient in cartesian coordinates and the contribution of each set of internals to the gradient.
- **.out2**: Log of the optimization in cartesian coordinates using the BGFS Quasi-Newton optimization algorithm.
- **.out3**: Log of the optimization in internal coordinates using the BGFS Quasi-Newton optimization algorithm. 

# Execution
The program can be executed from the terminal as:
```bash
python3 geom_optimizer.py inputs/filename.mol2
```
Or by modifying in the script file the variable `default_filename`:
```python3
default_filename = os.getcwd() + "/inputs/methane.mol2"
```


# Results
## Cartesian optimization

| Filename                    | Cartesian opt. ref. energy / $kcal · mol^{-1}$ | Cartesian opt. energy / $kcal · mol^{-1}$ | Deviation /  $kcal · mol^{-1}$ |
| --------------------------- | ---------------------------------------------- | ----------------------------------------- | ------------------------------ |
| Ethane                      | $-0.18518363$                                    | $-0.18518368$                               | $0.00000005$                     |
| Ethane_dist                 | $-0.18343836$                                    | $-0.18518367$                               | $0.00174531$                     |
| Isobutane                   | $0.27391887$                                     | $0.27391883$                                | $0.00000004$                     |
| Methane                     | $0.00005305$                                     | $0.00005299$                                | $0.00000006$                     |
| Methylcyclohexane           | $3.49862154$                                     | $3.49862132$                                | $0.00000022$                     |
| n_Butane                    | $-0.08747283$                                    | $-0.08747224$                               | $0.00000059$                     |
| Pinane                      | $80.28771004$                                    |  $80.28770984$                              | $0.0000002$                      |
| Cholestane                  | $-$                                              | $50.31436993$                               | $-$                              |
| Cholestane_reordered        | $-$                                              | $50.31436993$                               | $-$                              |
| Methylcyclohexane_reordered | $-$                                              | $3.49862132$                                | $-$                              |
| n_Butane_reordered          | $-$                                              | $-0.08747224$                               | $-$                              |
| Pinane_reordered            | $-$                                              | $80.28770984$                               | $-$                              |


## Internal optimization


| Filename                    | Internal opt. ref. energy / $kcal · mol^{-1}$ | Internal opt. energy / $kcal · mol^{-1}$ | Deviation /  $kcal · mol^{-1}$ |
| --------------------------- | --------------------------------------------- | ---------------------------------------- | ------------------------------ |
| Ethane                      | $-0.18518368$                                 | $-0.18518368$                            | $0$                            |
| Ethane_dist                 | $-0.18486694$                                 | $-0.18518367$                            | $0.00031673$                   |
| Isobutane                   | $0.27391876$                                  | $0.27391883$                             | $0.0000007$                    |
| Methane                     | $0.00005298$                                  | $0.00005299$                             | $0.00000001$                   |
| Methylcyclohexane           | $3.49862132$                                  | $3.49862132$                             | $0$                            |
| n_Butane                    | $-0.08747223$                                 | $-0.08747224$                            | $0.00000001$                   |
| Cholestane                  | $-$                                           | $50.31436632$                            | $-$                            |
| Cholestane_reordered        | $-$                                           | $50.31436632$                            | $-$                            |
| Methylcyclohexane_reordered | $-$                                           | $3.49862132$                             | $-$                            |
| n_Butane_reordered          | $-$                                           | $-0.08747224$                            | $-$                            |
| Pinane_reordered            | $-$                                           | $80.28770984$                            | $-$                            |
| Pinane                      | $-$                                           | $80.28770984$                            | $-$                            |

