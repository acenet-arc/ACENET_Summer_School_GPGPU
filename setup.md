---
layout: page
title: Setup
permalink: /setup/
---

This lesson will work best if done on a real cluster, like Alliance's Béluga, Cedar, Graham or Narval.
If you haven't got an account on those, our virtual training cluster will also be fine
although some of the profiling exercises might not work as described.

~~~
ssh userXX@pcsii.ace-net.training
~~~
{: .language-bash}

To set up your environment, load the GCC Compiler Collection and CUDA with:

~~~
module purge
module load  StdEnv/2020 gcc/9.3.0 cuda/11.4
~~~
{: .language-bash}

You should have access to the compiler and tools now. Test this with:

~~~
which nvcc
which nvprof
~~~
{: .language-bash}

> ## On real clusters we can use `StdEnv/2023` with `cuda/12.2` instead
> All of our real clusters have been upgraded to also support `StdEnv/2023` with `cuda/12.2` 
> instead of the older `cuda/11.4` that is still required on the virtual training clusters.
> ~~~
> module purge
> module load  StdEnv/2023 cuda/12.2
> ~~~
> {: .language-bash}
{: .callout }
