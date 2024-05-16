---
layout: page
title: Setup
permalink: /setup/
---

This lesson will work best if done on a real cluster, like Alliance's Béluga, Cedar, Graham or Narval.
If you haven't got an account on those, our virtual training cluster will also be fine
although some of the profiling exercises might not work as described.

~~~
ssh userXX@pcs2023-3.ace-net.training
~~~
{: .language-bash}

To set up your environment, load the NVidia HPC Software Development Kit and CUDA with:

~~~
module purge
module load StdEnv/2023 cuda/12.2
~~~
{: .language-bash}

You should have access to the compiler and tools now. Test this with:

~~~
which nvcc
which nvprof
~~~
{: .language-bash}
