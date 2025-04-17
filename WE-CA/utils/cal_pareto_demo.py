import torch
import numpy as np


class Pareto_sols(object):
    NAME = 'Pareto_sols'

    def __init__(self, p_size, pop_size=100, obj_num=3, eval_only=False):

        self.size = p_size  # the number of nodes in pdp
        self.pop_size = pop_size
        # self.pref = init_pref(pop_size)
        self.obj_num = obj_num

        # batch_size pop_size obj_num
        self.pareto_sols_num = int(1e5)
        self.pareto_sets_max_num = self.pareto_sols_num
        self.pareto_set = torch.ones((self.pareto_sols_num, obj_num)) * self.pareto_sets_max_num
        self.set_num = torch.zeros((obj_num))
        self.test = eval_only

        # self.max_sols_num = int(2e3)
        if self.test:
            self.sols = torch.ones((self.pareto_sols_num, self.size), dtype=torch.int64) * self.pareto_sets_max_num
            self.intinfs = torch.ones((self.pareto_sols_num, obj_num), dtype=torch.int64) * self.pareto_sets_max_num

        self.infs = torch.ones((self.pareto_sols_num, obj_num)) * self.pareto_sets_max_num

    def sort(self):
        bs = self.pareto_set.size(0)
        BATCH_IDX = torch.arange(bs)[:, None].expand(-1, self.pareto_sols_num)
        SORT_IDX = self.pareto_set[:, :, 0].argsort(-1)
        self.pareto_set = self.pareto_set[BATCH_IDX, SORT_IDX]
        if self.test:
            self.sols = self.sols[BATCH_IDX, SORT_IDX]


    def show_PE(self):
        inf_mask = self.pareto_set == self.pareto_sols_num
        self.set_num = (~inf_mask.any(-1)).long().sum(-1)

        # 由于在更新过程中，当前个体可能占优多个解
        # 如帕累托集合有: p1:[3.8, 10.5], p2:[3.9, 10.4], p3: [10.2, 4.1]， 帕累托个数为3
        # 如解o1:[3.7, 10.2] 占优p1,p2
        # 更新后变为：p1:[3.7, 10.2], p2:[3.7, 10.2], p3:[10.2, 4.1]
        # 之后会用inf去重，则变成：p1:[inf, inf], p2:[3.7, 10.2], p3:[10.2, 4.1]， 帕累托个数为2
        # 此时之前根据帕累托个数2来找前2个会遗漏掉p3，因此需要先排序，再返回帕累托集合

        # sort pareto_set
        self.sort()

        if self.test:
            return self.pareto_set, self.set_num, self.sols
        else:
            return self.pareto_set, self.set_num, None


    # 当前帕累托集合：self.pareto_set, shape: batch_size, max_pareto_set_num, obj_dim
    # 当前帕累托集合对应的路径：self.sols
    # 其中max_pareto_set_num表示当前可记录的最大帕累托个数
    # 迭代搜索过程中的新的人口：sols，对应目标空间下的值：objs
    # 当前帕累托集合的个数：self.set_num:
    #
    # 当前帕累托集中可替换的节点的值设置为infs，即新的人口在判断中总是会占优该点：self.infs:
    def update_PE(self, objs, sols=None):
        # objs.shape: batch_size, pop_size, obj_dim
        # sols.shape: batch_size, pop_size, graph_size
        #
        next_objs = objs.clone()
        if self.test:
            next_sols = sols.clone()
        bs, ps, obj_dim = next_objs.size()

        # init pareto set
        if self.pareto_set.dim() == 2:
            # old
            # self.pareto_set.shape: batch_size, pop_size, obj_dim
            self.pareto_set = self.pareto_set[None, :, :].expand(bs, -1, -1).to(next_objs.device)
            # PE len
            self.set_num = self.set_num[None, :].expand(bs, -1).to(next_objs.device)
            self.infs = self.infs.to(next_objs.device)
            if self.test:
                self.intinfs = self.intinfs.to(next_sols.device)
                self.sols = self.sols[None, :, :].expand(bs, -1, -1).to(next_sols.device)

        # update_loc = torch.ones((bs, ps)) * -1

        # find non-dominated sols
        # for

        # 按人口来遍历查找当前个体是否占优整个帕累托集合
        for pi in range(ps):
            # 当前个体跟所有帕累托集合的大小关系
            pareto_mask = next_objs[:, pi][:, None, :].expand(-1, self.pareto_sols_num, -1) < self.pareto_set
            #
            # check obj then check existing non-dominated
            # non_dominated mask

            # nd_mask = pareto_mask.any(-1) | self_mask.any(-1)
            # nd_mask = nd_mask.all(-1)

            # update idx where can put

            # 占优mask，记录当前个体是否占优一个以上的帕累托解
            nd_mask = pareto_mask.any(-1).all(-1)

            # 位置mask，记录当前个体可更新替换的帕累托解的索引位置
            idx_mask = pareto_mask.all(-1)
            # check if all sols non_dominated except inf

            # next_idx: update or put
            # TODO don't use []
            # 当前个体替换的位置
            next_pareto_idx = [idx_mask[i].nonzero()[0] for i in range(bs)]
            next_pareto_idx = torch.stack(next_pareto_idx, 0)
            # next_pareto_idx = next_pareto_idx[nd_mask]

            # 新的帕累托占优解
            tmp_value = self.pareto_set.scatter(1, next_pareto_idx[:, :, None].expand(-1, -1, obj_dim),
                                                next_objs[:, pi][:, None, :])
            # kk = self.pareto_set[:, :, k].masked_scatter_(nd_mask[:, None].expand(-1, self.pareto_sols_num), tmp_value)

            # 更新当前帕累托集合，更新所有实例中占优的个体
            self.pareto_set = torch.where(nd_mask[:, None, None].expand(-1, self.pareto_sols_num, obj_dim),
                                          tmp_value, self.pareto_set)
            if self.test:
                tmp_sols = self.sols.scatter(1, next_pareto_idx[:, :, None].expand(-1, -1, self.size),
                                             next_sols[:, pi][:, None, :])
                # kk = self.pareto_set[:, :, k].masked_scatter_(nd_mask[:, None].expand(-1, self.pareto_sols_num), tmp_value)
                self.sols = torch.where(nd_mask[:, None, None].expand(-1, self.pareto_sols_num, self.size),
                                        tmp_sols, self.sols)

            # tmp_value = self.pareto_set.scatter(1, next_pareto_idx, next_objs[:, pi][:, k][:, None])
            # # kk = self.pareto_set[:, :, k].masked_scatter_(nd_mask[:, None].expand(-1, self.pareto_sols_num), tmp_value)
            # self.pareto_set = torch.where(nd_mask[:, None].expand(-1, self.pareto_sols_num), tmp_value, self.pareto_set[:, :, k])

            # other multiple non-dominated delete
            # 由于更新时，当前个体可能同时占优多个帕累托解，因此需要继续检查是否还有占优别的解
            while True:
                pareto_mask = next_objs[:, pi][:, None, :].expand(-1, self.pareto_sols_num, -1) < self.pareto_set
                # self_mask = next_objs[:, pi][:, None,:].expand(-1, self.pareto_sols_num, -1) == self.pareto_set

                # 找到可更新的位置
                inf_mask = self.pareto_set == self.pareto_sets_max_num
                inf_mask = inf_mask.all(-1)
                idx_mask = pareto_mask.all(-1)

                # 除去inf这些必定可以放置的位置外，还有占优别的解，即说明可以继续更新
                update_mask = (idx_mask & ~inf_mask).any(-1)

                # 所有实例的当前个体已经没有占优的解
                if update_mask.any() == False:
                    break

                # 占优的解的位置
                idx_mask = idx_mask & ~inf_mask

                # protect not put idx
                # 由于部分实例的个体已经没有多的占优的解了，
                # 批量更新时为了避免没有可更新的位置，确保当前个体肯定能更新。仅用于代码辅助，而不会实际更新对应的解
                idx_mask[:, -1] = True

                next_pareto_idx = [idx_mask[i].nonzero()[0] for i in range(bs)]
                next_pareto_idx = torch.stack(next_pareto_idx, 0)
                # next_pareto_idx = next_pareto_idx[nd_mask]

                tmp_value = self.pareto_set.scatter(1, next_pareto_idx[:, :, None].expand(-1, -1, obj_dim),
                                                    self.infs[:, :, None].expand(-1, -1, obj_dim))
                # kk = self.pareto_set[:, :, k].masked_scatter_(nd_mask[:, None].expand(-1, self.pareto_sols_num), tmp_value)
                self.pareto_set = torch.where(update_mask[:, None, None].expand(-1, self.pareto_sols_num, obj_dim),
                                              tmp_value, self.pareto_set)

                if self.test:
                    tmp_sols = self.sols.scatter(1, next_pareto_idx[:, :, None].expand(-1, -1, self.size),
                                                 self.intinfs[:, :, None].expand(-1, -1, self.size))
                    # kk = self.pareto_set[:, :, k].masked_scatter_(nd_mask[:, None].expand(-1, self.pareto_sols_num), tmp_value)
                    self.sols = torch.where(update_mask[:, None, None].expand(-1, self.pareto_sols_num, self.size),
                                            tmp_sols, self.sols)
