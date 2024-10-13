# Released under MIT license
# Copyright (c) 2022 finetuneanon (NovelAI/Anlatan LLC)

import numpy as np
import pickle
import time

def get_prng(seed):
    return np.random.RandomState(seed)

class BucketManager:
    def __init__(self, bucket_file, valid_ids=None, max_size=(768,512), divisible=64, step_size=8, min_dim=256, base_res=(512,512), bsz=1, world_size=1, global_rank=0, max_ar_error=4, seed=42, dim_limit=1024, debug=False):
        with open(bucket_file, "rb") as fh:
            self.res_map = pickle.load(fh)
        if valid_ids is not None:
            new_res_map = {}
            valid_ids = set(valid_ids)
            for k, v in self.res_map.items():
                if k in valid_ids:
                    new_res_map[k] = v
            self.res_map = new_res_map
        self.max_size = max_size
        self.f = 8
        self.max_tokens = (max_size[0]/self.f) * (max_size[1]/self.f)
        self.div = divisible
        self.min_dim = min_dim
        self.dim_limit = dim_limit
        self.base_res = base_res
        self.bsz = bsz
        self.world_size = world_size
        self.global_rank = global_rank
        self.max_ar_error = max_ar_error
        self.prng = get_prng(seed)
        epoch_seed = self.prng.tomaxint() % (2**32-1)
        self.epoch_prng = get_prng(epoch_seed) # separate prng for sharding use for increased thread resilience
        self.epoch = None
        self.left_over = {}
        self.batch_total = None
        self.batch_delivered = None

        self.debug = debug

        self.gen_buckets()
        self.assign_buckets()
        self.start_epoch()

    def gen_buckets(self):
        if self.debug:
            timer = time.perf_counter()
        resolutions = []
        aspects = []
        w = self.min_dim
        while (w/self.f) * (self.min_dim/self.f) <= self.max_tokens and w <= self.dim_limit:
            h = self.min_dim
            got_base = False
            while (w/self.f) * ((h+self.div)/self.f) <= self.max_tokens and (h+self.div) <= self.dim_limit:
                if w == self.base_res[0] and h == self.base_res[1]:
                    got_base = True
                h += self.div
            if (w != self.base_res[0] or h != self.base_res[1]) and got_base:
                resolutions.append(self.base_res)
                aspects.append(1)
            resolutions.append((w, h))
            aspects.append(float(w)/float(h))
            w += self.div
        h = self.min_dim
        while (h/self.f) * (self.min_dim/self.f) <= self.max_tokens and h <= self.dim_limit:
            w = self.min_dim
            got_base = False
            while (h/self.f) * ((w+self.div)/self.f) <= self.max_tokens and (w+self.div) <= self.dim_limit:
                if w == self.base_res[0] and h == self.base_res[1]:
                    got_base = True
                w += self.div
            resolutions.append((w, h))
            aspects.append(float(w)/float(h))
            h += self.div
        res_map = {}
        for i, res in enumerate(resolutions):
            res_map[res] = aspects[i]
        self.resolutions = sorted(res_map.keys(), key=lambda x: x[0] * 4096 - x[1])
        self.aspects = np.array(list(map(lambda x: res_map[x], self.resolutions)))
        self.resolutions = np.array(self.resolutions)
        if self.debug:
            timer = time.perf_counter() - timer
            print(f"resolutions:\n{self.resolutions}")
            print(f"aspects:\n{self.aspects}")
            print(f"gen_buckets: {timer:.5f}s")

    def assign_buckets(self):
        if self.debug:
            timer = time.perf_counter()
        self.buckets = {}
        self.aspect_errors = []
        skipped = 0
        skip_list = []
        for post_id in self.res_map.keys():
            w, h = self.res_map[post_id]
            aspect = float(w)/float(h)
            bucket_id = np.abs(np.log(self.aspects) - np.log(aspect)).argmin()
            if bucket_id not in self.buckets:
                self.buckets[bucket_id] = []
            error = abs(self.aspects[bucket_id] - aspect)
            if error < self.max_ar_error:
                self.buckets[bucket_id].append(post_id)
                if self.debug:
                    self.aspect_errors.append(error)
            else:
                skipped += 1
                skip_list.append(post_id)
        for post_id in skip_list:
            del self.res_map[post_id]
        if self.debug:
            timer = time.perf_counter() - timer
            self.aspect_errors = np.array(self.aspect_errors)
            print(f"skipped images: {skipped}")
            print(f"aspect error: mean {self.aspect_errors.mean()}, median {np.median(self.aspect_errors)}, max {self.aspect_errors.max()}")
            for bucket_id in reversed(sorted(self.buckets.keys(), key=lambda b: len(self.buckets[b]))):
                print(f"bucket {bucket_id}: {self.resolutions[bucket_id]}, aspect {self.aspects[bucket_id]:.5f}, entries {len(self.buckets[bucket_id])}")
            print(f"assign_buckets: {timer:.5f}s")

    def start_epoch(self, world_size=None, global_rank=None):
        if self.debug:
            timer = time.perf_counter()
        if world_size is not None:
            self.world_size = world_size
        if global_rank is not None:
            self.global_rank = global_rank

        # select ids for this epoch/rank
        index = np.array(sorted(list(self.res_map.keys())))
        index_len = index.shape[0]
        index = self.epoch_prng.permutation(index)
        index = index[:index_len - (index_len % (self.bsz * self.world_size))]
        #print("perm", self.global_rank, index[0:16])
        index = index[self.global_rank::self.world_size]
        self.batch_total = index.shape[0] // self.bsz
        assert(index.shape[0] % self.bsz == 0)
        index = set(index)

        self.epoch = {}
        self.left_over = {}
        self.batch_delivered = 0
        for bucket_id in sorted(self.buckets.keys()):
            if len(self.buckets[bucket_id]) > 0:
                self.epoch[bucket_id] = np.array([post_id for post_id in self.buckets[bucket_id] if post_id in index], dtype=np.int64)
                self.prng.shuffle(self.epoch[bucket_id])
                self.epoch[bucket_id] = list(self.epoch[bucket_id])
                overhang = len(self.epoch[bucket_id]) % self.bsz
                if overhang != 0:
                    if bucket_id not in self.left_over:
                        self.left_over[bucket_id] = []
                    self.left_over[bucket_id].extend(self.epoch[bucket_id][:overhang])
                    self.epoch[bucket_id] = self.epoch[bucket_id][overhang:]

        if self.debug:
            timer = time.perf_counter() - timer
            count = 0
            for bucket_id in self.epoch.keys():
                count += len(self.epoch[bucket_id])
            print(f"correct item count: {count == len(index)} ({count} of {len(index)})")
            print(f"start_epoch: {timer:.5f}s")

    def get_batch(self):
        if self.debug:
            timer = time.perf_counter()
        if self.epoch is None or self.batch_total == self.batch_delivered:
            self.start_epoch()

        found_batch = False
        batch_data = None
        resolution = self.base_res
        while not found_batch:
            bucket_ids = list(self.epoch.keys())
            bucket_probs = [len(self.epoch[bucket_id]) for bucket_id in bucket_ids]

            left_over_bucket_ids = list(self.left_over.keys())
            left_over_bucket_probs = [len(self.left_over[bucket_id]) for bucket_id in left_over_bucket_ids]

            all_bucket_ids = bucket_ids + left_over_bucket_ids
            all_bucket_probs = bucket_probs + left_over_bucket_probs

            if len(all_bucket_probs) == 0:
                # No buckets left, start new epoch
                self.start_epoch()
                continue

            all_bucket_probs = np.array(all_bucket_probs, dtype=np.float32)
            all_bucket_probs = all_bucket_probs / all_bucket_probs.sum()
            all_bucket_ids = np.array(all_bucket_ids, dtype=np.int64)

            chosen_id = int(self.prng.choice(all_bucket_ids, 1, p=all_bucket_probs)[0])

            if chosen_id in self.epoch:
                if len(self.epoch[chosen_id]) >= self.bsz:
                    batch_data = self.epoch[chosen_id][:self.bsz]
                    self.epoch[chosen_id] = self.epoch[chosen_id][self.bsz:]
                    resolution = tuple(self.resolutions[chosen_id])
                    found_batch = True
                    if len(self.epoch[chosen_id]) == 0:
                        del self.epoch[chosen_id]
                else:
                    # Move leftovers to left_over dict
                    if chosen_id not in self.left_over:
                        self.left_over[chosen_id] = []
                    self.left_over[chosen_id].extend(self.epoch[chosen_id])
                    del self.epoch[chosen_id]
            elif chosen_id in self.left_over:
                if len(self.left_over[chosen_id]) >= self.bsz:
                    batch_data = self.left_over[chosen_id][:self.bsz]
                    self.left_over[chosen_id] = self.left_over[chosen_id][self.bsz:]
                    resolution = tuple(self.resolutions[chosen_id])
                    found_batch = True
                    if len(self.left_over[chosen_id]) == 0:
                        del self.left_over[chosen_id]
                else:
                    # Not enough images to form a batch, keep them for the next epoch
                    del self.left_over[chosen_id]
            else:
                # Should not happen
                assert False, "Chosen bucket ID not found in epoch or leftovers"

        if self.debug:
            timer = time.perf_counter() - timer
            print(f"bucket probs: " + ", ".join(map(lambda x: f"{x:.2f}", list(bucket_probs*100))))
            print(f"chosen id: {chosen_id}")
            print(f"batch data: {batch_data}")
            print(f"resolution: {resolution}")
            print(f"get_batch: {timer:.5f}s")

        self.batch_delivered += 1
        return (batch_data, resolution)

    def generator(self):
        if self.batch_delivered >= self.batch_total:
            self.start_epoch()
        while self.batch_delivered < self.batch_total:
            yield self.get_batch()

if __name__ == "__main__":
    # prepare a pickle with mapping of dataset IDs to resolutions called resolutions.pkl to use this
    with open("resolutions.pkl", "rb") as fh:
        ids = list(pickle.load(fh).keys())

    counts = np.zeros((len(ids),)).astype(np.int64)
    id_map = {}
    for i, post_id in enumerate(ids):
        id_map[post_id] = i

    bm = BucketManager("resolutions.pkl", debug=True, bsz=8, world_size=8, global_rank=3)
    print("got: " + str(bm.get_batch()))
    print("got: " + str(bm.get_batch()))
    print("got: " + str(bm.get_batch()))
    print("got: " + str(bm.get_batch()))
    print("got: " + str(bm.get_batch()))
    print("got: " + str(bm.get_batch()))
    print("got: " + str(bm.get_batch()))

    bm = BucketManager("resolutions.pkl", bsz=8, world_size=1, global_rank=0, valid_ids=ids[0:16])
    for _ in range(16):
        bm.get_batch()
    print("got from future epoch: " + str(bm.get_batch()))

    bms = []
    for rank in range(16):
        bm = BucketManager("resolutions.pkl", bsz=8, world_size=16, global_rank=rank)
        bms.append(bm)
    for epoch in range(5):
        print(f"epoch {epoch}")
        for i, bm in enumerate(bms):
            print(f"bm {i}")
            first = True
            for ids, res in bm.generator():
                if first and i == 0:
                    #print(ids)
                    first = False
                for post_id in ids:
                    counts[id_map[post_id]] += 1
        print(np.bincount(counts))
