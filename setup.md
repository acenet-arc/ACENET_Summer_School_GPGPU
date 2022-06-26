---
layout: page
title: Setup
permalink: /setup/
---

This lesson will work best if done on a real cluster, like Compute Canada's Béluga, Cedar, or Graham.
If you haven't got an account on those, our virtual training cluster will also be fine
although some of the profiling exercises might not work as described.

~~~
ssh userXX@pcs2.ace-net.training
~~~
{: .bash}

To set up your environment, load the NVidia HPC Software Development Kit and CUDA with:

~~~
module load StdEnv/2020
module load nvhpc/20.7
~~~
{: .bash}

You should have access to the compiler and tools now. Test this with:

~~~
which nvcc
which nvprof
~~~
{: .bash}
