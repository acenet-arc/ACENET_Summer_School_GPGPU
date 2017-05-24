---
layout: page
title: Setup
permalink: /setup/
---

To get access to one of the graphics machines, you will need to connect to the mahone cluster first.

~~~
ssh username@mahone.ace-net.ca
~~~
{: .bash}

From there, you will need to connect to one of the graphics machines.

~~~
ssh clg01.smu.acenet.ca
OR
ssh clg02.smu.acenet.ca
OR
ssh clg03.smu.acenet.ca
~~~
{: .bash}

To set up your environment correctly, you will need to run the following commands first.

~~~
module purge

module load gcc cuda
~~~
{: .bash}

You should have access to the compiler and tools now.
