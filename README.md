# NovelAI Aspect Ratio Bucketing

Training with aspect ratio bucketing can greatly improve the quality of outputs, so we have decided to release the bucketing code under a permissive MIT license.

This repository provides an implementation of aspect ratio bucketing for training generative image models as described in our [blogpost](https://blog.novelai.net/novelai-improvements-on-stable-diffusion-e10d38db82ac). The relevant section of the post is reproduced below.

## Description

One common issue of existing image generation models is that they are very prone to producing images with unnatural crops. This is due to the fact that these models are trained to produce square images. However, most photos and artworks are not square. However, the model can only work on images of the same size at the same time, and during training, it is common practice to operate on multiple training samples at once to optimize the efficiency of the GPUs used. As a compromise, square images are chosen, and during training, only the center of each image is cropped out and then shown to the image generation model as a training example.

![Knight wearing a crown with darkened regions removed by the center crop](https://cdn.discordapp.com/attachments/864987405487833158/1028721357128749207/knightcrop.png)

For example, humans are often generated without feet or heads, and swords consist of only a blade with a hilt and point outside the frame.
As we are creating an image generation model to accompany our storytelling experience, it is important that our model is able to produce proper, uncropped characters, and generated knights should not be holding a metallic-looking straight line extending to infinity.

Another issue with training on cropped images is that it can lead to a mismatch between the text and the image.

For example, an image with a `crown` tag will often no longer contain a crown after a center crop is applied and the monarch has been, thereby, decapitated.

We found that using random crops instead of center crops only slightly improves these issues.

Using Stable Diffusion with variable image sizes is possible, although it can be noticed that going too far beyond the native resolution of 512x512 tends to introduce repeated image elements, and very low resolutions produce indiscernible images.

Still, this indicated to us that training the model on variable sized images should be possible. Training on single, variable sized samples would be trivial, but also extremely slow and more liable to training instability due to the lack of regularization provided by the use of mini batches.

### Custom Batch Generation

As no existing solution for this problem seems to exist, we have implemented custom batch generation code for our dataset that allows the creation of batches where every item in the batch has the same size, but the image size of batches may differ.

We do this through a method we call aspect ratio bucketing. An alternative approach would be to use a fixed image size, scale each image to fit within this fixed size and apply padding that is masked out during training. Since this leads to unnecessary computation during training, we have not chosen to follow this alternative approach.

In the following, we describe the original idea behind our custom batch generation scheme for aspect ratio bucketing.

First, we have to define which buckets we want to sort the images of our dataset into. For this purpose, we define a maximum image size of 512x768 with a maximum dimension size of 1024. Since the maximum image size is 512x768, which is larger than 512x512 and requires more VRAM, per-GPU batch size has to be lowered, which can be compensated through gradient accumulation.

We generate buckets by applying the following algorithm:

* Set the width to 256.
*  While the width is less than or equal to 1024:
    * Find the largest height such that height is less than or equal to 1024 and that width multiplied by height is less than or equal to 512 * 768.
    * Add the resolution given by height and width as a bucket.
    * Increase the width by 64.

The same is repeated with width and height exchanged. Duplicated buckets are pruned from the list, and an additional bucket sized 512x512 is added.

Next, we assign images to their corresponding buckets. For this purpose, we first store the bucket resolutions in a NumPy array and calculate the aspect ratio of each resolution. For each image in the dataset, we then retrieve its resolution and calculate the aspect ratio. The image aspect ratio is subtracted from the array of bucket aspect ratios, allowing us to efficiently select the closest bucket according to the absolute value of the difference between aspect ratios:

```
image_bucket = argmin(abs(bucket_aspects — image_aspect))
```

The image’s bucket number is stored associated with its item ID in the dataset. If the image’s aspect ratio is very extreme and too different from even the best-fitting bucket, the image is pruned from the dataset.

Since we train on multiple GPUs, before each epoch, we shard the dataset to ensure that each GPU works on a distinct subset of equal size. To do this, we first copy the list of item IDs in the dataset and shuffle them. If this copied list is not divisible by the number of GPUs multiplied by the batch size, the list is trimmed, and the last items are dropped to make it divisible.

We then select a distinct subset of `1/world_size*bsz` item IDs according to the global rank of the current process. The rest of the custom batch generation will be described as seen from any single of these processes and operate on the subset of dataset item IDs.

For the current shard, lists for each bucket are created by iterating over the list of shuffled dataset item IDs and assigning the ID to the list corresponding to the bucket that was assigned to the image.

Once all images are processed, we iterate over the lists for each bucket. If its length is not divisible by the batch size, we remove the last elements on the list as necessary to make it divisible and add them to a separate catch-all bucket. As the overall shard size is guaranteed to contain a number of elements divisible by the batch size, doing is guaranteed to produce a catch-all bucket with a length divisible by the batch size as well.

When a batch is requested, we draw randomly draw a bucket from a weighted distribution. The bucket weights are set as the size of the bucket divided by the size of all remaining buckets. This ensures that even with buckets of widely varying sizes, the custom batch generation does not introduce strong bias when during training, an image shows up according to image size. If buckets were chosen without weighting, small buckets would empty out early during the training process, and only the biggest buckets would remain towards the end of training. Weighting buckets by size avoids this.

A batch of items is finally taken from the chosen bucket. The items taken are removed from the bucket. If the bucket is now empty, it is deleted for the rest of the epoch. The chosen item IDs and the chosen bucket’s resolution are now passed to an image-loading function.

### Image Loading

*Note that image loading code is not part of this release but should be relatively easy to implement.*

Each item ID’s image is loaded and processed to fit within the bucket resolution. For fitting the image, two approaches are possible.

First, the image could be simply rescaled. This would lead to a slight distortion of the image. For this reason, we have opted for the second approach:

The image is scaled, while preserving its aspect ratio, in such a way that it:

* Either fits the bucket resolution exactly if the aspect ratio happens to match
* or it extends past the bucket resolution on one dimension while fitting it exactly on the other.

In the latter case, a random crop is applied.

As we found that the mean aspect ratio error per image is only 0.033, these random crops only remove very little of the actual image, usually less than 32 pixels.

The loaded and processed images are finally returned as the image part of the batch.

### Sample Output

```
resolutions:
[[ 256 1024] [ 320 1024] [ 384 1024] [ 384  960] [ 384  896] [ 448  832] [ 512  768] [ 512  704]
 [ 512  512] [ 576  640] [ 640  576] [ 704  512] [ 768  512] [ 832  448] [ 896  384] [ 960  384]
 [1024  384] [1024  320] [1024  256]]
aspects:
[0.25       0.3125     0.375      0.4        0.42857143 0.53846154
 0.66666667 0.72727273 1.         0.9        1.11111111 1.375
 1.5        1.85714286 2.33333333 2.5        2.66666667 3.2
 4.        ]
gen_buckets: 0.00012s
skipped images: 344
aspect error: mean 0.03291049121627481, median 0.022727272727272707, max 3.991769547325103
bucket 7: [512 704], aspect 0.72727, entries 2189784
bucket 6: [512 768], aspect 0.66667, entries 661814
bucket 11: [704 512], aspect 1.37500, entries 586743
bucket 8: [512 512], aspect 1.00000, entries 429811
bucket 9: [576 640], aspect 0.90000, entries 417592
bucket 5: [448 832], aspect 0.53846, entries 247869
bucket 10: [640 576], aspect 1.11111, entries 231463
bucket 12: [768 512], aspect 1.50000, entries 209557
bucket 13: [832 448], aspect 1.85714, entries 173294
bucket 4: [384 896], aspect 0.42857, entries 42191
bucket 1: [ 320 1024], aspect 0.31250, entries 38428
bucket 2: [ 384 1024], aspect 0.37500, entries 24377
bucket 0: [ 256 1024], aspect 0.25000, entries 16618
bucket 14: [896 384], aspect 2.33333, entries 15046
bucket 3: [384 960], aspect 0.40000, entries 12010
bucket 16: [1024  384], aspect 2.66667, entries 4512
bucket 17: [1024  320], aspect 3.20000, entries 3704
bucket 15: [960 384], aspect 2.50000, entries 3478
bucket 18: [1024  256], aspect 4.00000, entries 2326
assign_buckets: 15.47429s
```

## Author

finetuneanon (NovelAI/Anlatan LLC)
