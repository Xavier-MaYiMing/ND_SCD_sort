### ND_SCD_sort

##### Reference: Yue C, Qu B, Liang J. A multiobjective particle swarm optimizer using ring topology for solving multimodal multiobjective problems[J]. IEEE Transactions on Evolutionary Computation, 2017, 22(5): 805-817.

ND_SCD_sort sorts the population of multi-objective evolutionary algorithms (MOEAs) based on Pareto domination relation and special crowding distance. ND_SCD_sort is one of the most popular methods to tackle multi-modal multi-objective optimization (MMO) problems.

##### Please refer to the reference for the details of ND_SCD_sort.

| Variables | Meaning                              |
| --------- | ------------------------------------ |
| pos       | Positions                            |
| objs      | Objectives                           |
| pfs       | Pareto fronts                        |
| rank      | Pareto rank                          |
| cd_x      | Crowding distance in decision space  |
| cd_f      | Crowding distance in objective space |
| scd       | Special crowding distance            |



#### Example

```python
if __name__ == '__main__':
    t_pos = np.random.random((10, 3))
    t_objs = np.random.random((10, 2))
    r_pos, r_objs = nd_scd_sort(t_pos, t_objs)[: 2]
    print('The original positions: ')
    print(t_pos)
    print('The original objectives: ')
    print(t_objs)
    print('------------------------')
    print('The sorted positions based on Pareto domination and special crowding distance: ')
    print(r_pos)
    print('The sorted objectives based on Pareto domination and special crowding distance: ')
    print(r_objs)
```

##### Output:

```python
The original positions: 
[[0.08484405 0.36677165 0.77498237]
 [0.00126532 0.64373405 0.97892205]
 [0.85226172 0.52998179 0.71642737]
 [0.44501516 0.92888961 0.55512165]
 [0.90569244 0.08479633 0.70259529]
 [0.90408263 0.07829138 0.74610056]
 [0.32341601 0.72087247 0.48613805]
 [0.1112431  0.58715168 0.53226198]
 [0.65088873 0.79914697 0.82516492]
 [0.92306252 0.91974271 0.98503334]]
The original objectives: 
[[0.51259806 0.65060918]
 [0.37876048 0.31724222]
 [0.7519934  0.22350311]
 [0.95518027 0.28658587]
 [0.50983785 0.22816687]
 [0.97878821 0.12471704]
 [0.73433777 0.28491689]
 [0.75185691 0.70928685]
 [0.03751547 0.80754622]
 [0.36776369 0.84922796]]
------------------------
The sorted positions based on Pareto domination and special crowding distance: 
[[0.00126532 0.64373405 0.97892205]
 [0.65088873 0.79914697 0.82516492]
 [0.90408263 0.07829138 0.74610056]
 [0.85226172 0.52998179 0.71642737]
 [0.90569244 0.08479633 0.70259529]
 [0.32341601 0.72087247 0.48613805]
 [0.08484405 0.36677165 0.77498237]
 [0.92306252 0.91974271 0.98503334]
 [0.44501516 0.92888961 0.55512165]
 [0.1112431  0.58715168 0.53226198]]
The sorted objectives based on Pareto domination and special crowding distance: 
[[0.37876048 0.31724222]
 [0.03751547 0.80754622]
 [0.97878821 0.12471704]
 [0.7519934  0.22350311]
 [0.50983785 0.22816687]
 [0.73433777 0.28491689]
 [0.51259806 0.65060918]
 [0.36776369 0.84922796]
 [0.95518027 0.28658587]
 [0.75185691 0.70928685]]
```

