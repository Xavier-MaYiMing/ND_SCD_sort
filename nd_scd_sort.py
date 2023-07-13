#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/12 18:10
# @Author  : Xavier Ma
# @Email   : xavier_mayiming@163.com
# @File    : nd_scd_sort.py
# @Statement : Sort the population of multi-objective evolutionary algorithms (MOEAs) based on Pareto domination relation and special crowding distance.
# @Reference : Yue C, Qu B, Liang J. A multiobjective particle swarm optimizer using ring topology for solving multimodal multiobjective problems[J]. IEEE Transactions on Evolutionary Computation, 2017, 22(5): 805-817.
import numpy as np


def nd_sort(objs):
    # fast non-dominated sort
    npop = len(objs)
    nobj = len(objs[0])
    n = np.zeros(npop, dtype=int)  # the number of particles that dominate this particle
    s = []  # the index of particles that this particle dominates
    rank = np.zeros(npop, dtype=int)  # Pareto rank
    ind = 1
    pfs = {ind: []}  # Pareto fronts
    for i in range(npop):
        s.append([])
        for j in range(npop):
            if i != j:
                less = equal = more = 0
                for k in range(nobj):
                    if objs[i, k] < objs[j, k]:
                        less += 1
                    elif objs[i, k] == objs[j, k]:
                        equal += 1
                    else:
                        more += 1
                if less == 0 and equal != nobj:
                    n[i] += 1
                elif more == 0 and equal != nobj:
                    s[i].append(j)
        if n[i] == 0:
            pfs[ind].append(i)
            rank[i] = ind
    while pfs[ind]:
        pfs[ind + 1] = []
        for i in pfs[ind]:
            for j in s[i]:
                n[j] -= 1
                if n[j] == 0:
                    pfs[ind + 1].append(j)
                    rank[j] = ind + 1
        ind += 1
    pfs.pop(ind)
    return pfs, rank


def special_crowding_distance(pos, objs, pfs):
    # calculate the special crowding distance (SCD)
    (npop, dim) = pos.shape
    nobj = objs.shape[1]
    cd_x = np.zeros(npop)  # CD in decision space
    cd_f = np.zeros(npop)  # CD in objective space
    scd = np.zeros(npop)  # SCD
    for key in pfs.keys():
        pf = np.array(pfs[key])
        if len(pf) == 1:
            cd_x[pf[0]] = 1
            cd_f[pf[0]] = 1
            continue
        temp_pos = pos[pf]
        temp_obj = objs[pf]

        # calculate CD in decision space
        xmin = np.min(temp_pos, axis=0)
        xmax = np.max(temp_pos, axis=0)
        dx = xmax - xmin
        for i in range(dim):
            if dx[i] == 0:
                for j in range(len(pf)):
                    cd_x[pf[j]] += 1
            else:
                rank = np.argsort(temp_pos[:, i])
                cd_x[pf[rank[0]]] += 2 * (pos[pf[rank[1]], i] - pos[pf[rank[0]], i]) / dx[i]
                cd_x[pf[rank[-1]]] += 2 * (pos[pf[rank[-1]], i] - pos[pf[rank[-2]], i]) / dx[i]
                for j in range(1, len(pf) - 1):
                    cd_x[pf[rank[j]]] += (pos[pf[rank[j + 1]], i] - pos[pf[rank[j - 1]], i]) / dx[i]
        cd_x[pf] /= dim

        # calculate CD in objective space
        fmin = np.min(temp_obj, axis=0)
        fmax = np.max(temp_obj, axis=0)
        df = fmax - fmin
        for i in range(nobj):
            if df[i] == 0:
                for j in range(len(pf)):
                    cd_f[pf[j]] += 1
            else:
                rank = np.argsort(temp_obj[:, i])
                cd_f[pf[rank[0]]] += 1
                cd_f[pf[rank[-1]]] += 0
                for j in range(1, len(pf) - 1):
                    cd_f[pf[rank[j]]] += (objs[pf[rank[j + 1]], i] - objs[pf[rank[j - 1]], i]) / df[i]
        cd_f[pf] /= nobj

        # calculate SCD
        cd_x_avg = np.mean(cd_x[pf])
        cd_f_avg = np.mean(cd_f[pf])
        flag = np.logical_or(cd_x[pf] > cd_x_avg, cd_f[pf] > cd_f_avg)
        scd[pf] = np.where(flag, np.max((cd_x[pf], cd_f[pf]), axis=0), np.min((cd_x[pf], cd_f[pf]), axis=0))
    return scd


def nd_scd_sort(pos, objs):
    # sort the particles according to the Pareto rank and special crowding distance
    pos = np.array(pos)
    objs = np.array(objs)
    npop = pos.shape[0]
    pfs, rank = nd_sort(objs)
    scd = special_crowding_distance(pos, objs, pfs)
    temp_list = []
    for i in range(len(pos)):
        temp_list.append([pos[i], objs[i], rank[i], scd[i]])
    temp_list.sort(key=lambda x: (x[2], -x[3]))
    next_pos = np.zeros((npop, pos.shape[1]))
    next_objs = np.zeros((npop, objs.shape[1]))
    next_rank = np.zeros(npop)
    for i in range(npop):
        next_pos[i] = temp_list[i][0]
        next_objs[i] = temp_list[i][1]
        next_rank[i] = temp_list[i][2]
    return next_pos, next_objs, next_rank


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
